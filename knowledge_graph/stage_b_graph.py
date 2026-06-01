#!/usr/bin/env python3
"""
Stage B graph builder for the CIA unredacted corpus.

This produces the `graphs/` outputs from `extraction/` and consumes Stage A1
entity canonicalization outputs when available.

Primary input contract:
- `entity_resolution/entity_canonical_map.csv`
- `entity_resolution/entities_resolved.csv` (optional)
- `entity_resolution/entity_alias_map_for_notebook_b.csv` (fallback)

If no A1 outputs are present, Stage B falls back to mechanical normalization
only. Stage B is not responsible for entity resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from graph_labeling import make_theme_family, make_topic_label
except ImportError:
    from knowledge_graph.graph_labeling import make_theme_family, make_topic_label

try:
    import igraph as ig
except Exception as exc:
    raise RuntimeError("Please install igraph: pip install igraph") from exc

try:
    import leidenalg as la
except Exception as exc:
    raise RuntimeError("Please install leidenalg: pip install leidenalg") from exc

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:
    raise RuntimeError("Please install scikit-learn: pip install scikit-learn") from exc


KNOWLEDGE_GRAPH_ROOT = Path(__file__).resolve().parent
DEFAULT_ENTITY_DIRECTORY_NAME = "entity_directory"
DEFAULT_ENTITY_ALIAS_FILENAME = "entity_aliases.csv"
GENERATED_ALIAS_FILENAME = "entity_aliases_generated.csv"
ENTITY_CANONICAL_MAP_FILENAME = "entity_canonical_map.csv"
ENTITY_ALIAS_MAP_FOR_B_FILENAME = "entity_alias_map_for_notebook_b.csv"
ENTITIES_RESOLVED_FILENAME = "entities_resolved.csv"

#
# Main Stage B tuning knobs:
# - `DEFAULT_MIN_DEGREE_FOR_COMMUNITY_GRAPH`
#     Filters very low-degree nodes out of the clustering view.
#     Increase to remove more peripheral noise; decrease to keep more weakly
#     connected structure. This is usually a secondary tuning lever.
# - `DEFAULT_LEIDEN_RESOLUTION`
#     Controls community granularity in the recursive hierarchy.
#     Lower values give fewer, larger groups; higher values give more, smaller
#     groups. This is the main clustering-shape knob.
#     On the CIA full run, ~1.2 still produced level-1 communities that were
#     too broad for analyst use, so level 1 is now tuned more aggressively via
#     `LEIDEN_RESOLUTION_BY_LEVEL`.
# - `DEFAULT_MAX_HIERARCHY_LEVELS`
#     Controls how many recursive hierarchy layers Stage B attempts to build.
#     The analyst-facing workflow currently exposes two levels downstream
#     (communities -> narratives), so the default is 2.
DEFAULT_MIN_DEGREE_FOR_COMMUNITY_GRAPH = 2
DEFAULT_LEIDEN_RESOLUTION = 1.20
DEFAULT_MAX_HIERARCHY_LEVELS = 2
LEIDEN_RESOLUTION_BY_LEVEL = {
    1: 2.40,
    2: 0.95,
}

# Event clustering threshold for merging extracted event mentions into
# `EventCluster` nodes.
# - Increase to require stronger textual similarity before events merge.
# - Decrease to merge more aggressively.
# This is an important upstream knob because overly loose event clustering can
# create giant generic event hubs that distort the hierarchy.
DEFAULT_EVENT_CLUSTER_SIMILARITY_THRESHOLD = 0.64

META_EDGE_MIN_WEIGHT_BY_LEVEL = {
    # Minimum meta-edge score kept at each recursive hierarchy level after
    # mutual-strength reweighting.
    #
    # The score is based on both:
    # - relative strength of the edge compared with each endpoint's total
    #   meta-graph connectivity, and
    # - reciprocal rank agreement between the two endpoints.
    #
    # Increase a level's threshold to make that hierarchy layer sparser and
    # more selective. Decrease it to keep more cross-group structure.
    #
    # Level 1 is the main narrative-connectivity knob. Values just above 0.06
    # cut off several otherwise meaningful community links; 0.055 keeps those
    # lighter but still mutually-supported links without reverting to the
    # earlier blob-like meta-graph.
    1: 0.055,
    2: 0.15,
    3: 0.20,
}

EDGE_TYPE_WEIGHTS = {
    "CO_OCCURRENCE": 0.15,
    "SHARED_ENTITY_EVENT_LINK": 0.35,
    "CANONICAL_EVENT_PARTICIPANT": 1.00,
    "EVENT_PARTICIPANT": 1.00,
    "CANONICAL_EVENT_LOCATION": 0.80,
    "EVENT_LOCATION": 0.80,
    "AFFILIATION": 0.80,
    "COMMUNICATION": 0.70,
    "OPPOSITION": 0.90,
    "SUPPORT": 0.70,
    "REQUEST": 0.60,
    "META_LINK": 1.00,
}

GENERIC_CLUSTER_HUBS = {
    "CANONICAL_ENTITY:UNITED_STATES",
    "CANONICAL_ENTITY:SOVIET_UNION",
    "CANONICAL_ENTITY:CHINA",
    "CANONICAL_ENTITY:UNITED_NATIONS",
    "CANONICAL_ENTITY:GOVERNMENT",
    "CANONICAL_ENTITY:PRESIDENT",
    "CANONICAL_ENTITY:PREMIER",
    "CANONICAL_ENTITY:EMBASSY",
    "CANONICAL_ENTITY:MINISTRY",
    "CANONICAL_ENTITY:CENTRAL_INTELLIGENCE_BULLETIN",
}

GENERIC_META_TOKENS = {
    "UNITED STATES",
    "SOVIET UNION",
    "CHINA",
    "UNITED NATIONS",
    "GOVERNMENT",
    "PRESIDENT",
    "PREMIER",
    "EMBASSY",
    "MINISTRY",
    "CENTRAL INTELLIGENCE BULLETIN",
}

GENERIC_HUB_EDGE_FACTOR = 0.25
MAX_COOCCURRENCE_EDGES_PER_CHUNK = 10
DISPLAY_LABEL_SEPARATOR = " / "
TOP_LABELS_SEPARATOR = " | "

COMPACT_LABEL_STOPWORDS = {
    "BULLETIN",
    "CURRENT",
    "INTELLIGENCE",
    "COUNTRY",
    "GOVERNMENT",
    "OFFICIALS",
    "OFFICIAL",
    "PARTY",
    "ARMY",
    "STATE",
    "GENERAL",
    "REGIME",
    "GROUP",
    "MOVEMENT",
    "MILITARY",
    "NATIONAL",
    "COMMUNIST",
    "COMMISSION",
    "CONFERENCE",
    "INTERNATIONAL",
    "DOCUMENT",
    "REPORT",
}

A1_CANONICAL_MAP: Dict[str, str] = {}
A1_CANONICAL_KEY_MAP: Dict[str, str] = {}
A1_CANONICAL_METHOD_MAP: Dict[str, str] = {}
A1_CANONICAL_CONFIDENCE_MAP: Dict[str, float] = {}
A1_CANONICAL_MAP_LOADED = False


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def norm_space(text: object) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def normalize_label_basic(text: object) -> str:
    normalized = norm_space(text)
    normalized = normalized.replace("&amp;", "&")
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = normalized.upper()
    normalized = re.sub(r"[\"'`]+", "", normalized)
    normalized = re.sub(r"[.,;:()\[\]{}]", " ", normalized)
    normalized = re.sub(r"[-_/]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^(THE|A|AN)\s+", "", normalized)
    normalized = normalized.replace("U S A", "USA")
    normalized = normalized.replace("U S", "US")
    normalized = normalized.replace("U N", "UN")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def keyify(text: object) -> str:
    normalized = normalize_label_basic(text)
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "UNKNOWN"


def stable_hash(text: object, length: int = 10) -> str:
    payload = norm_space(text) or "UNKNOWN"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length].upper()


def split_trailing_parenthetical(label: object) -> Tuple[str, str]:
    text = norm_space(label)
    match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", text)
    if not match:
        return text, ""
    return norm_space(match.group(1)), norm_space(match.group(2))


def informative_descriptor_tokens(text: object) -> List[str]:
    tokens: List[str] = []
    seen: set[str] = set()
    for token in re.split(r"[^A-Z0-9]+", normalize_label_basic(text)):
        if not token or token.isdigit() or len(token) < 3:
            continue
        if token in COMPACT_LABEL_STOPWORDS:
            continue
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def compact_canonical_label_base(label: object) -> str:
    base, _ = split_trailing_parenthetical(label)
    return base or norm_space(label)


def compact_canonical_entity_label(label: object, force_disambiguate: bool = False) -> str:
    full_label = norm_space(label)
    if not full_label:
        return "UNKNOWN ENTITY"

    base, descriptor = split_trailing_parenthetical(full_label)
    base = base or full_label
    if not force_disambiguate:
        return base

    descriptor_tokens = informative_descriptor_tokens(descriptor)
    if descriptor_tokens:
        return f"{base} [{' / '.join(descriptor_tokens[:2])}]"

    sense_match = re.search(r"\[SENSE\s+(\d+)\]", full_label, flags=re.IGNORECASE)
    if sense_match:
        return f"{base} [SENSE {sense_match.group(1)}]"

    return f"{base} [{stable_hash(full_label, length=4)}]"


def compact_event_label(label: object, max_chars: int = 80) -> str:
    text = norm_space(label)
    if not text:
        return "EVENT"

    parts = [norm_space(part) for part in text.split("|") if norm_space(part)]
    head = parts[0] if parts else text
    if len(head) <= max_chars:
        return head
    return head[: max_chars - 3].rstrip() + "..."


def compact_misc_label(label: object, max_chars: int = 80) -> str:
    text = norm_space(label)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def downstream_canonical_node_key(canonical_label: object, canonical_key_full: object = "") -> str:
    full_label = norm_space(canonical_label) or "UNKNOWN ENTITY"
    base = compact_canonical_label_base(full_label)
    anchor = keyify(base)[:48] or "ENTITY"
    stable_source = norm_space(canonical_key_full) or full_label
    return f"CANONICAL_ENTITY:{anchor}__{stable_hash(stable_source)}"


def apply_concise_node_labels(nodes: pd.DataFrame) -> pd.DataFrame:
    if len(nodes) == 0:
        return nodes.copy()

    out = nodes.copy()
    if "label_full" not in out.columns:
        out["label_full"] = out["label"].map(norm_space)
    else:
        out["label_full"] = out["label_full"].map(norm_space)

    out["display_label"] = out["label_full"]

    canonical_mask = out["node_type"].astype(str) == "CanonicalEntity"
    if canonical_mask.any():
        canonical_bases = out.loc[canonical_mask, "label_full"].map(compact_canonical_label_base)
        base_counts = canonical_bases.value_counts()
        for idx in out.index[canonical_mask]:
            full_label = out.at[idx, "label_full"]
            base = compact_canonical_label_base(full_label)
            force_disambiguate = int(base_counts.get(base, 0)) > 1
            out.at[idx, "display_label"] = compact_canonical_entity_label(
                full_label,
                force_disambiguate=force_disambiguate,
            )

        duplicate_mask = canonical_mask & out["display_label"].duplicated(keep=False)
        if duplicate_mask.any():
            for idx in out.index[duplicate_mask]:
                full_label = out.at[idx, "label_full"]
                current = norm_space(out.at[idx, "display_label"])
                out.at[idx, "display_label"] = f"{current} [{stable_hash(full_label, length=4)}]"

    event_mask = out["node_type"].astype(str) == "EventCluster"
    if event_mask.any():
        out.loc[event_mask, "display_label"] = out.loc[event_mask, "label_full"].map(compact_event_label)

    other_mask = ~(canonical_mask | event_mask)
    if other_mask.any():
        out.loc[other_mask, "display_label"] = out.loc[other_mask, "label_full"].map(compact_misc_label)

    out["label"] = out["display_label"]
    return out


def summarized_label_counts(group: pd.DataFrame, label_col: str = "label") -> Counter:
    counts: Counter = Counter()
    weight_col = "member_count" if "member_count" in group.columns else "size" if "size" in group.columns else None

    if "top_labels" in group.columns:
        for _, row in group.iterrows():
            weight = int(row.get(weight_col, 1)) if weight_col else 1
            pieces = [
                norm_space(part)
                for part in re.split(r"\s+\|\s+|;", safe_str(row.get("top_labels", "")))
                if norm_space(part)
            ]
            for piece in pieces:
                counts[piece] += max(1, weight)
        if counts:
            return counts

    for _, row in group.iterrows():
        label = norm_space(row.get(label_col, ""))
        if label:
            weight = int(row.get(weight_col, 1)) if weight_col else 1
            counts[label] += max(1, weight)
    return counts


def join_display_labels(labels: Sequence[str], max_items: int) -> str:
    cleaned = [norm_space(label) for label in labels if norm_space(label)]
    return DISPLAY_LABEL_SEPARATOR.join(cleaned[:max_items])


def join_top_labels(labels: Sequence[str], max_items: int) -> str:
    cleaned = [norm_space(label) for label in labels if norm_space(label)]
    return TOP_LABELS_SEPARATOR.join(cleaned[:max_items])


def make_hierarchy_display_label(labels: Sequence[str], fallback: str) -> str:
    top_labels_text = join_top_labels(labels, max_items=8)
    short_fallback = join_display_labels(labels, max_items=3) if labels else fallback
    return make_topic_label(top_labels_text, fallback=short_fallback or fallback)


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns {missing}. Columns are: {df.columns.tolist()}")


def make_node_key(prefix: str, label: str) -> str:
    return f"{prefix}:{keyify(label)}"


def make_event_text(row: pd.Series) -> str:
    pieces: List[str] = []
    for column in ["label", "event_label", "description", "summary", "evidence", "type", "date"]:
        if column in row.index:
            value = norm_space(row.get(column, ""))
            if value:
                pieces.append(value)
    return " | ".join(pieces) or "EVENT"


def ensure_weight_col(edges: pd.DataFrame) -> pd.DataFrame:
    if "weight" not in edges.columns:
        if "confidence" in edges.columns:
            edges["weight"] = pd.to_numeric(edges["confidence"], errors="coerce").fillna(0.7)
        else:
            edges["weight"] = 0.7
    else:
        edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0.7)
    return edges


def ensure_rel_col(edges: pd.DataFrame) -> pd.DataFrame:
    if "rel" not in edges.columns:
        if "relation" in edges.columns:
            edges["rel"] = edges["relation"]
        else:
            edges["rel"] = "RELATED"
    edges["rel"] = edges["rel"].astype(str).str.strip().str.upper()
    return edges


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def resolve_entity_directory(project_dir: Path, entity_dir_arg: Optional[Path] = None) -> Path:
    if entity_dir_arg is not None:
        return Path(entity_dir_arg).expanduser().resolve()
    return project_dir / DEFAULT_ENTITY_DIRECTORY_NAME


def load_entity_alias_table(entity_dir: Path) -> pd.DataFrame:
    alias_path = entity_dir / DEFAULT_ENTITY_ALIAS_FILENAME
    if not alias_path.exists():
        return pd.DataFrame(columns=["alias", "canonical_label", "notes"])

    alias_df = pd.read_csv(alias_path)
    if len(alias_df) == 0:
        return pd.DataFrame(columns=["alias", "canonical_label", "notes"])

    alias_col = first_existing_col(alias_df, ["alias", "variant", "label"])
    canonical_col = first_existing_col(alias_df, ["canonical_label", "canonical", "canonical_name"])
    notes_col = first_existing_col(alias_df, ["notes", "note", "comment"])

    if alias_col is None or canonical_col is None:
        raise KeyError(
            f"Alias table must contain alias and canonical columns. Columns are: {alias_df.columns.tolist()}"
        )

    out = pd.DataFrame(
        {
            "alias": alias_df[alias_col].map(norm_space),
            "canonical_label": alias_df[canonical_col].map(norm_space),
            "notes": alias_df[notes_col].map(norm_space) if notes_col else "",
        }
    )
    out = out[(out["alias"] != "") & (out["canonical_label"] != "")].drop_duplicates()
    return out.reset_index(drop=True)


def load_combined_entity_alias_table(project_dir: Path, entity_dir: Path) -> pd.DataFrame:
    seed_aliases = load_entity_alias_table(entity_dir)
    generated_path = project_dir / "entity_resolution" / GENERATED_ALIAS_FILENAME
    if not generated_path.exists():
        return seed_aliases

    generated_aliases = pd.read_csv(generated_path)
    if len(generated_aliases) == 0:
        return seed_aliases

    alias_col = first_existing_col(generated_aliases, ["alias", "variant", "label"])
    canonical_col = first_existing_col(generated_aliases, ["canonical_label", "canonical", "canonical_name"])
    notes_col = first_existing_col(generated_aliases, ["notes", "note", "comment"])
    if alias_col is None or canonical_col is None:
        return seed_aliases

    generated = pd.DataFrame(
        {
            "alias": generated_aliases[alias_col].map(norm_space),
            "canonical_label": generated_aliases[canonical_col].map(norm_space),
            "notes": generated_aliases[notes_col].map(norm_space) if notes_col else "",
        }
    )
    generated = generated[(generated["alias"] != "") & (generated["canonical_label"] != "")].drop_duplicates()
    if len(seed_aliases) == 0:
        return generated.reset_index(drop=True)

    seed_aliases = seed_aliases.copy()
    seed_aliases["alias_key"] = seed_aliases["alias"].map(normalize_label_basic)
    generated["alias_key"] = generated["alias"].map(normalize_label_basic)
    combined = pd.concat([seed_aliases, generated], ignore_index=True)
    combined = combined.drop_duplicates(subset=["alias_key"], keep="last").drop(columns=["alias_key"])
    return combined.reset_index(drop=True)


def build_entity_alias_map(alias_df: pd.DataFrame) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for _, row in alias_df.iterrows():
        alias_map[normalize_label_basic(row["alias"])] = normalize_label_basic(row["canonical_label"])
    return alias_map


def canonicalize_entity_label(label: object, alias_map: Dict[str, str]) -> str:
    raw = normalize_label_basic(label)
    if not raw:
        return ""
    return alias_map.get(raw, raw)


def load_a1_entity_resolution(project_dir: Path, entity_dir: Path) -> Dict[str, Dict]:
    project_dir = Path(project_dir)
    er_dir = project_dir / "entity_resolution"
    preferred = er_dir / ENTITY_CANONICAL_MAP_FILENAME
    fallback = er_dir / ENTITY_ALIAS_MAP_FOR_B_FILENAME
    generated = er_dir / GENERATED_ALIAS_FILENAME

    path: Optional[Path] = None
    if preferred.exists():
        path = preferred
    elif fallback.exists():
        path = fallback
    elif generated.exists():
        path = generated

    if path is not None:
        df = pd.read_csv(path)
        if len(df) == 0:
            return {
                "canonical": {},
                "canonical_key": {},
                "method": {},
                "confidence": {},
                "loaded": False,
                "path": str(path),
            }

        if path.name in {ENTITY_CANONICAL_MAP_FILENAME, ENTITY_ALIAS_MAP_FOR_B_FILENAME}:
            original_col = first_existing_col(df, ["original_normalized_label", "alias", "variant", "label"])
            canonical_col = first_existing_col(df, ["canonical_label", "canonical", "canonical_name"])
            if original_col is None or canonical_col is None:
                raise KeyError(
                    f"A1 map {path} must contain original_normalized_label/alias and canonical_label. "
                    f"Columns found: {df.columns.tolist()}"
                )

            df = df.copy()
            df["original_normalized_label"] = df[original_col].map(normalize_label_basic)
            df["canonical_label"] = df[canonical_col].map(norm_space)
            source_key_col = "canonical_key" if "canonical_key" in df.columns else None
            df["canonical_key_full"] = df[source_key_col].map(norm_space) if source_key_col else ""
            df["canonical_key"] = df.apply(
                lambda row: downstream_canonical_node_key(row["canonical_label"], row["canonical_key_full"]),
                axis=1,
            )
            if "method" not in df.columns:
                df["method"] = "a1_map"
            if "confidence" not in df.columns:
                df["confidence"] = 1.0
        else:
            alias_col = first_existing_col(df, ["alias", "variant", "label"])
            canonical_col = first_existing_col(df, ["canonical_label", "canonical", "canonical_name"])
            if alias_col is None or canonical_col is None:
                raise KeyError(
                    f"Generated alias table {path} must contain alias and canonical_label columns. "
                    f"Columns found: {df.columns.tolist()}"
                )
            df = df.copy()
            df["original_normalized_label"] = df[alias_col].map(normalize_label_basic)
            df["canonical_label"] = df[canonical_col].map(norm_space)
            df["canonical_key_full"] = ""
            df["canonical_key"] = df["canonical_label"].map(downstream_canonical_node_key)
            df["method"] = df[first_existing_col(df, ["review_status", "source"])] if first_existing_col(df, ["review_status", "source"]) else "a1_generated_alias"
            if first_existing_col(df, ["openai_confidence", "final_score"]):
                df["confidence"] = pd.to_numeric(df[first_existing_col(df, ["openai_confidence", "final_score"])], errors="coerce").fillna(1.0)
            else:
                df["confidence"] = 1.0

        canonical = dict(zip(df["original_normalized_label"], df["canonical_label"]))
        canonical_key = dict(zip(df["original_normalized_label"], df["canonical_key"].astype(str)))
        method = dict(zip(df["original_normalized_label"], df["method"].astype(str)))
        confidence = dict(zip(df["original_normalized_label"], pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0).astype(float)))

        print("Loaded Stage A1 entity-resolution map:", path)
        print("A1 canonical map rows:", len(df))
        print("A1 unique canonical labels:", df["canonical_label"].nunique())
        if "method" in df.columns:
            print("A1 method counts:")
            print(df["method"].astype(str).value_counts().head(10))
        return {
            "canonical": canonical,
            "canonical_key": canonical_key,
            "method": method,
            "confidence": confidence,
            "loaded": True,
            "path": str(path),
        }

    print("Stage A1 entity-resolution map not found; Stage B will fall back to mechanical normalization only.")
    return {
        "canonical": {},
        "canonical_key": {},
        "method": {},
        "confidence": {},
        "loaded": False,
        "path": "",
    }


def set_a1_entity_resolution_maps(project_dir: Path, entity_dir: Path) -> None:
    global A1_CANONICAL_MAP, A1_CANONICAL_KEY_MAP
    global A1_CANONICAL_METHOD_MAP, A1_CANONICAL_CONFIDENCE_MAP
    global A1_CANONICAL_MAP_LOADED

    maps = load_a1_entity_resolution(project_dir, entity_dir)
    A1_CANONICAL_MAP = maps["canonical"]
    A1_CANONICAL_KEY_MAP = maps["canonical_key"]
    A1_CANONICAL_METHOD_MAP = maps["method"]
    A1_CANONICAL_CONFIDENCE_MAP = maps["confidence"]
    A1_CANONICAL_MAP_LOADED = bool(maps["loaded"])


def canonicalize_entity_label_a1(label: object) -> str:
    normalized = normalize_label_basic(label)
    if not normalized:
        return "UNKNOWN_ENTITY"
    return A1_CANONICAL_MAP.get(normalized, normalized)


def canonical_key_for_entity_label(label: object) -> str:
    normalized = normalize_label_basic(label)
    canonical_label = canonicalize_entity_label_a1(label)
    return A1_CANONICAL_KEY_MAP.get(normalized, downstream_canonical_node_key(canonical_label))


def canonical_method_for_entity_label(label: object) -> str:
    normalized = normalize_label_basic(label)
    return A1_CANONICAL_METHOD_MAP.get(normalized, "a1_missing_fallback_normalization")


def canonical_confidence_for_entity_label(label: object) -> float:
    normalized = normalize_label_basic(label)
    return float(A1_CANONICAL_CONFIDENCE_MAP.get(normalized, 1.0))


def strip_leading_article(label: str) -> str:
    stripped = re.sub(r"^(THE|A|AN)\s+", "", label).strip()
    return stripped or label


def acronym_from_label(label: str) -> str:
    tokens = [
        token
        for token in re.split(r"[\s\-]+", label)
        if token and token not in {"OF", "THE", "AND", "IN", "TO", "FOR", "A", "AN"}
    ]
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens if token and token[0].isalpha())


def looks_like_acronym(label: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", label)
    return compact == label and 2 <= len(label) <= 6


def build_entity_label_frequencies(
    entities: pd.DataFrame,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    type_col = first_existing_col(entities, ["type", "entity_type", "broad_type"])
    if label_col is None or len(entities) == 0:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "canonical_label_current",
                "mention_count",
                "unique_docs",
                "source_types",
                "example_mentions",
                "in_alias_dictionary",
            ]
        )

    rows: List[Dict[str, object]] = []
    for _, row in entities.iterrows():
        raw_label = norm_space(row.get(label_col, ""))
        if not raw_label:
            continue
        normalized_label = normalize_label_basic(raw_label)
        rows.append(
            {
                "normalized_label": normalized_label,
                "raw_label": raw_label,
                "doc_id": safe_str(row.get("doc_id", "")),
                "source_type": norm_space(row.get(type_col, "")) if type_col else "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "canonical_label_current",
                "mention_count",
                "unique_docs",
                "source_types",
                "example_mentions",
                "in_alias_dictionary",
            ]
        )

    tmp = pd.DataFrame(rows)
    grouped_rows: List[Dict[str, object]] = []

    for normalized_label, group in tmp.groupby("normalized_label"):
        examples = pd.unique(group["raw_label"].astype(str)).tolist()[:10]
        source_types = sorted({value for value in group["source_type"].astype(str) if value})
        grouped_rows.append(
            {
                "normalized_label": normalized_label,
                "canonical_label_current": canonicalize_entity_label(normalized_label, alias_map),
                "mention_count": int(len(group)),
                "unique_docs": int(group["doc_id"].astype(str).nunique()),
                "source_types": "; ".join(source_types[:10]),
                "example_mentions": "; ".join(examples),
                "in_alias_dictionary": normalized_label in alias_map,
            }
        )

    out = pd.DataFrame(grouped_rows)
    out = out.sort_values(
        ["mention_count", "unique_docs", "normalized_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return out


def build_alias_review_candidates(
    label_frequencies: pd.DataFrame,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    if len(label_frequencies) == 0:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "suggested_canonical_label",
                "reason",
                "mention_count",
                "unique_docs",
                "source_types",
                "example_mentions",
            ]
        )

    known_labels = set(label_frequencies["normalized_label"].astype(str)) | set(alias_map.values())
    mention_count_map = dict(
        zip(
            label_frequencies["normalized_label"].astype(str),
            label_frequencies["mention_count"].astype(int),
        )
    )
    labels = sorted(known_labels)
    candidate_rows: List[Dict[str, object]] = []

    for _, row in label_frequencies.iterrows():
        normalized_label = safe_str(row["normalized_label"])
        if not normalized_label or normalized_label in alias_map:
            continue

        suggested = ""
        reason = ""

        stripped = strip_leading_article(normalized_label)
        if stripped != normalized_label and stripped in known_labels:
            suggested = stripped
            reason = "leading_article"
        elif normalized_label.startswith("GOVERNMENT OF "):
            tail = normalized_label[len("GOVERNMENT OF ") :].strip()
            if tail in known_labels:
                suggested = tail
                reason = "government_of"
        elif looks_like_acronym(normalized_label):
            matches = [label for label in labels if label != normalized_label and acronym_from_label(label) == normalized_label]
            if matches:
                suggested = max(matches, key=lambda label: (mention_count_map.get(label, 0), len(label)))
                reason = "acronym_match"

        if suggested and suggested != normalized_label:
            candidate_rows.append(
                {
                    "normalized_label": normalized_label,
                    "suggested_canonical_label": suggested,
                    "reason": reason,
                    "mention_count": int(row["mention_count"]),
                    "unique_docs": int(row["unique_docs"]),
                    "source_types": safe_str(row["source_types"]),
                    "example_mentions": safe_str(row["example_mentions"]),
                }
            )

    out = pd.DataFrame(candidate_rows)
    if len(out) == 0:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "suggested_canonical_label",
                "reason",
                "mention_count",
                "unique_docs",
                "source_types",
                "example_mentions",
            ]
        )

    out = out.sort_values(
        ["mention_count", "unique_docs", "normalized_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return out


def load_extraction_tables(project_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extraction_dir = project_dir / "extraction"
    entity_resolution_dir = project_dir / "entity_resolution"
    if not extraction_dir.exists():
        raise FileNotFoundError(f"Extraction directory not found: {extraction_dir}")

    paths = {
        "entities": extraction_dir / "entities.csv",
        "events": extraction_dir / "events.csv",
        "claims": extraction_dir / "claims.csv",
        "relations": extraction_dir / "relations.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} file: {path}")

    resolved_entities_path = entity_resolution_dir / ENTITIES_RESOLVED_FILENAME
    if resolved_entities_path.exists():
        entities = pd.read_csv(resolved_entities_path)
        print("Loaded resolved entities from:", resolved_entities_path)
    else:
        entities = pd.read_csv(paths["entities"])
    events = pd.read_csv(paths["events"])
    claims = pd.read_csv(paths["claims"])
    relations = pd.read_csv(paths["relations"])

    print("Loaded tables:")
    print("entities", entities.shape)
    print("events", events.shape)
    print("claims", claims.shape)
    print("relations", relations.shape)

    return entities, events, claims, relations


def add_local_keys(
    entities: pd.DataFrame,
    events: pd.DataFrame,
    claims: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name, df in [("entities", entities), ("events", events), ("claims", claims)]:
        require_columns(df, ["doc_id", "paragraph_id"], name)
        if "local_id" not in df.columns:
            if "id" in df.columns:
                df["local_id"] = df["id"]
            else:
                df["local_id"] = [f"{name[:1]}{idx}" for idx in range(len(df))]
        df["mention_key"] = (
            df["doc_id"].astype(str)
            + "::"
            + df["paragraph_id"].astype(str)
            + "::"
            + df["local_id"].astype(str)
        )
    return entities, events, claims


def build_canonical_entities(
    entities: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    type_col = first_existing_col(entities, ["type", "entity_type", "broad_type"])
    desc_col = first_existing_col(entities, ["description", "summary", "evidence"])
    canonical_label_col = first_existing_col(entities, ["final_canonical_label", "canonical_label"])
    canonical_key_col = first_existing_col(entities, ["final_canonical_key", "canonical_key"])
    canonical_method_col = first_existing_col(entities, ["canonical_method", "final_method"])
    canonical_confidence_col = first_existing_col(entities, ["canonical_confidence", "final_confidence"])
    if label_col is None:
        raise KeyError(f"Could not find entity label column. Columns: {entities.columns.tolist()}")

    rows: List[Dict[str, object]] = []
    provenance_rows: List[Dict[str, object]] = []
    mention_to_canonical: Dict[str, str] = {}

    for _, row in entities.iterrows():
        label = norm_space(row.get(label_col, ""))
        if not label:
            continue
        canonical_label_full = (
            norm_space(row.get(canonical_label_col, "")) if canonical_label_col else ""
        ) or canonicalize_entity_label_a1(label)
        canonical_key_full = norm_space(row.get(canonical_key_col, "")) if canonical_key_col else ""
        node_key = downstream_canonical_node_key(canonical_label_full, canonical_key_full) or canonical_key_for_entity_label(label)
        canonical_method = (
            norm_space(row.get(canonical_method_col, "")) if canonical_method_col else ""
        ) or canonical_method_for_entity_label(label)
        try:
            confidence_value = row.get(canonical_confidence_col, np.nan) if canonical_confidence_col else np.nan
            canonical_confidence = float(confidence_value if not pd.isna(confidence_value) else canonical_confidence_for_entity_label(label))
        except Exception:
            canonical_confidence = canonical_confidence_for_entity_label(label)
        mention_to_canonical[str(row["mention_key"])] = node_key
        rows.append(
            {
                "node_key": node_key,
                "label": canonical_label_full,
                "label_full": canonical_label_full,
                "node_type": "CanonicalEntity",
                "source_type": norm_space(row.get(type_col, "Entity")) if type_col else "Entity",
                "example_mention": label,
                "a1_method": canonical_method,
                "a1_confidence": canonical_confidence,
                "a1_map_loaded": A1_CANONICAL_MAP_LOADED,
            }
        )
        provenance_rows.append(
            {
                "node_key": node_key,
                "canonical_label": canonical_label_full,
                "canonical_label_full": canonical_label_full,
                "canonical_key_full": canonical_key_full,
                "doc_id": safe_str(row.get("doc_id", "")),
                "paragraph_id": safe_str(row.get("paragraph_id", "")),
                "mention_key": safe_str(row.get("mention_key", "")),
                "raw_label": label,
                "source_type": norm_space(row.get(type_col, "Entity")) if type_col else "Entity",
                "description": norm_space(row.get(desc_col, "")) if desc_col else "",
                "canonical_method": canonical_method,
                "canonical_confidence": canonical_confidence,
            }
        )

    nodes = pd.DataFrame(rows)
    provenance = pd.DataFrame(provenance_rows)
    if len(nodes) == 0:
        return (
            pd.DataFrame(
                columns=[
                    "node_key",
                    "label",
                    "label_full",
                    "node_type",
                    "source_type",
                    "example_mention",
                    "a1_method",
                    "a1_confidence",
                    "a1_map_loaded",
                    "mention_count",
                    "doc_count",
                    "doc_ids_sample",
                ]
            ),
            mention_to_canonical,
            pd.DataFrame(
                columns=[
                    "node_key",
                    "canonical_label",
                    "canonical_label_full",
                    "canonical_label_display",
                    "canonical_key_full",
                    "doc_id",
                    "paragraph_id",
                    "mention_key",
                    "raw_label",
                    "source_type",
                    "description",
                    "canonical_method",
                    "canonical_confidence",
                ]
            ),
        )

    nodes = nodes.groupby("node_key", as_index=False).agg(
        label=("label", "first"),
        label_full=("label_full", "first"),
        node_type=("node_type", "first"),
        source_type=("source_type", lambda values: "; ".join(sorted(set(map(str, values)))[:5])),
        example_mention=("example_mention", "first"),
        a1_method=("a1_method", lambda values: "; ".join(sorted(set(map(str, values)))[:5])),
        a1_confidence=("a1_confidence", "max"),
        a1_map_loaded=("a1_map_loaded", "first"),
        mention_count=("node_key", "size"),
        doc_count=("node_key", lambda values: 0),
    )
    if len(provenance):
        doc_summary = provenance.groupby("node_key").agg(
            doc_count=("doc_id", lambda values: pd.Series(values).astype(str).nunique()),
            doc_ids_sample=("doc_id", lambda values: "; ".join(sorted(pd.Series(values).astype(str).dropna().unique().tolist())[:20])),
        ).reset_index()
        nodes = nodes.drop(columns=["doc_count"]).merge(doc_summary, on="node_key", how="left")
    else:
        nodes["doc_count"] = 0
        nodes["doc_ids_sample"] = ""
    nodes = apply_concise_node_labels(nodes)
    if len(provenance):
        display_label_map = dict(zip(nodes["node_key"].astype(str), nodes["label"].astype(str)))
        provenance["canonical_label_display"] = provenance["node_key"].astype(str).map(display_label_map).fillna("")
        provenance["canonical_label"] = provenance["canonical_label_display"]
    print("Canonical entity nodes:", nodes.shape)
    return nodes, mention_to_canonical, provenance


def build_event_clusters(events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if len(events) == 0:
        return pd.DataFrame(columns=["node_key", "label", "node_type", "source_type", "mention_count"]), {}

    events = events.copy()
    events["event_text_for_cluster"] = events.apply(make_event_text, axis=1)

    if len(events) < 2:
        cluster_key = "EVENT_CLUSTER:0"
        return (
            pd.DataFrame(
                [
                    {
                        "node_key": cluster_key,
                        "label": events.iloc[0]["event_text_for_cluster"][:120],
                        "label_full": events.iloc[0]["event_text_for_cluster"][:120],
                        "node_type": "EventCluster",
                        "source_type": "Event",
                        "mention_count": 1,
                    }
                ]
            ),
            {str(events.iloc[0]["mention_key"]): cluster_key},
        )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(events["event_text_for_cluster"].fillna("EVENT"))
    except ValueError:
        cluster_rows = []
        mention_to_event_cluster = {}
        for idx, (_, row) in enumerate(events.iterrows()):
            cluster_key = f"EVENT_CLUSTER:{idx}"
            cluster_rows.append(
                {
                    "node_key": cluster_key,
                    "label": safe_str(row["event_text_for_cluster"])[:160],
                    "label_full": safe_str(row["event_text_for_cluster"])[:160],
                    "node_type": "EventCluster",
                    "source_type": "Event",
                    "mention_count": 1,
                }
            )
            mention_to_event_cluster[str(row["mention_key"])] = cluster_key
        return pd.DataFrame(cluster_rows), mention_to_event_cluster

    similarity = cosine_similarity(matrix)

    parent = list(range(len(events)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    threshold = DEFAULT_EVENT_CLUSTER_SIMILARITY_THRESHOLD
    max_neighbors = 5
    for idx in range(len(events)):
        sims = similarity[idx]
        candidates = np.argsort(-sims)[1 : max_neighbors + 1]
        for candidate in candidates:
            if sims[candidate] >= threshold:
                union(idx, int(candidate))

    root_to_members: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(events)):
        root_to_members[find(idx)].append(idx)

    cluster_rows = []
    mention_to_event_cluster: Dict[str, str] = {}
    for cluster_idx, members in enumerate(root_to_members.values()):
        cluster_key = f"EVENT_CLUSTER:{cluster_idx}"
        texts = events.iloc[members]["event_text_for_cluster"].tolist()
        label = sorted(texts, key=lambda text: (len(text), text))[0][:160]
        cluster_rows.append(
            {
                "node_key": cluster_key,
                "label": label,
                "label_full": label,
                "node_type": "EventCluster",
                "source_type": "Event",
                "mention_count": len(members),
            }
        )
        for member in members:
            mention_to_event_cluster[str(events.iloc[member]["mention_key"])] = cluster_key

    event_nodes = pd.DataFrame(cluster_rows)
    print("Event cluster nodes:", event_nodes.shape)
    return event_nodes, mention_to_event_cluster


def build_relation_edges(
    relations: pd.DataFrame,
    entities: pd.DataFrame,
    events: pd.DataFrame,
    mention_to_canonical: Dict[str, str],
    mention_to_event_cluster: Dict[str, str],
) -> pd.DataFrame:
    edges = relations.copy()
    edges = ensure_rel_col(edges)
    edges = ensure_weight_col(edges)

    source_col = first_existing_col(edges, ["source", "source_id", "source_local_id", "head", "subject_id", "from_id"])
    target_col = first_existing_col(edges, ["target", "target_id", "target_local_id", "tail", "object_id", "to_id"])

    if source_col is None or target_col is None:
        print("WARNING: Could not identify relation source/target columns. Relation edges skipped.")
        return pd.DataFrame(columns=["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    local_lookup: Dict[Tuple[str, str, str], str] = {}
    for df, mapping in [(entities, mention_to_canonical), (events, mention_to_event_cluster)]:
        if "local_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            key = (
                safe_str(row.get("doc_id", "")),
                safe_str(row.get("paragraph_id", "")),
                safe_str(row.get("local_id", "")),
            )
            mention_key = safe_str(row.get("mention_key", ""))
            if mention_key in mapping:
                local_lookup[key] = mapping[mention_key]

    edge_rows: List[Dict[str, object]] = []
    for _, row in edges.iterrows():
        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        source_raw = safe_str(row.get(source_col, ""))
        target_raw = safe_str(row.get(target_col, ""))

        source_node = local_lookup.get((doc_id, paragraph_id, source_raw))
        target_node = local_lookup.get((doc_id, paragraph_id, target_raw))

        if not source_node and source_raw:
            source_node = canonical_key_for_entity_label(source_raw)
        if not target_node and target_raw:
            target_node = canonical_key_for_entity_label(target_raw)

        if not source_node or not target_node or source_node == target_node:
            continue

        left, right = canonical_pair(source_node, target_node)
        edge_rows.append(
            {
                "source_global": left,
                "target_global": right,
                "rel": safe_str(row.get("rel", "RELATED")).upper(),
                "weight": float(row.get("weight", 0.7)),
                "support_count": 1,
                "doc_ids": doc_id,
            }
        )

    if not edge_rows:
        return pd.DataFrame(columns=["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    out = pd.DataFrame(edge_rows)
    out = out.groupby(["source_global", "target_global", "rel"], as_index=False).agg(
        weight=("weight", "sum"),
        support_count=("support_count", "sum"),
        doc_ids=("doc_ids", lambda values: ";".join(sorted(set(value for value in map(str, values) if value))[:20])),
    )
    return out


def build_cooccurrence_edges(
    entities: pd.DataFrame,
    mention_to_canonical: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    cols = ["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"]
    if len(entities) == 0:
        return pd.DataFrame(columns=cols)

    for (doc_id, paragraph_id), group in entities.groupby(["doc_id", "paragraph_id"]):
        nodes = []
        for _, row in group.iterrows():
            mention_key = safe_str(row.get("mention_key", ""))
            node = mention_to_canonical.get(mention_key)
            if node:
                nodes.append(node)

        nodes = sorted(set(nodes))
        if len(nodes) < 2:
            continue

        candidate_pairs = []
        for left_idx in range(len(nodes)):
            for right_idx in range(left_idx + 1, len(nodes)):
                left = nodes[left_idx]
                right = nodes[right_idx]
                hub_penalty = int(left in GENERIC_CLUSTER_HUBS) + int(right in GENERIC_CLUSTER_HUBS)
                candidate_pairs.append((hub_penalty, left, right))

        candidate_pairs = sorted(candidate_pairs, key=lambda item: item[0])[:MAX_COOCCURRENCE_EDGES_PER_CHUNK]
        for _, left, right in candidate_pairs:
            rows.append(
                {
                    "source_global": left,
                    "target_global": right,
                    "rel": "CO_OCCURRENCE",
                    "weight": 0.5,
                    "support_count": 1,
                    "doc_ids": safe_str(doc_id),
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    out = out.groupby(["source_global", "target_global", "rel"], as_index=False).agg(
        weight=("weight", "sum"),
        support_count=("support_count", "sum"),
        doc_ids=("doc_ids", lambda values: ";".join(sorted(set(map(str, values)))[:20])),
    )
    return out


def build_event_entity_edges(
    events: pd.DataFrame,
    entities: pd.DataFrame,
    mention_to_canonical: Dict[str, str],
    mention_to_event_cluster: Dict[str, str],
) -> pd.DataFrame:
    if len(events) == 0 or len(entities) == 0:
        return pd.DataFrame(columns=["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    entity_by_paragraph: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for _, row in entities.iterrows():
        mention_key = safe_str(row.get("mention_key", ""))
        node = mention_to_canonical.get(mention_key)
        if node:
            entity_by_paragraph[(safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))].add(node)

    rows: List[Dict[str, object]] = []
    for _, row in events.iterrows():
        mention_key = safe_str(row.get("mention_key", ""))
        event_node = mention_to_event_cluster.get(mention_key)
        if not event_node:
            continue

        paragraph_key = (safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
        for entity_node in entity_by_paragraph.get(paragraph_key, set()):
            left, right = canonical_pair(event_node, entity_node)
            rows.append(
                {
                    "source_global": left,
                    "target_global": right,
                    "rel": "SHARED_ENTITY_EVENT_LINK",
                    "weight": 0.7,
                    "support_count": 1,
                    "doc_ids": safe_str(row.get("doc_id", "")),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    out = pd.DataFrame(rows)
    out = out.groupby(["source_global", "target_global", "rel"], as_index=False).agg(
        weight=("weight", "sum"),
        support_count=("support_count", "sum"),
        doc_ids=("doc_ids", lambda values: ";".join(sorted(set(map(str, values)))[:20])),
    )
    return out


def combine_edges(edge_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in edge_frames if frame is not None and len(frame)]
    cols = ["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"]
    if not frames:
        return pd.DataFrame(columns=cols)

    all_edges = pd.concat(frames, ignore_index=True)
    all_edges = all_edges[all_edges["source_global"] != all_edges["target_global"]].copy()
    out = all_edges.groupby(["source_global", "target_global", "rel"], as_index=False).agg(
        weight=("weight", "sum"),
        support_count=("support_count", "sum"),
        doc_ids=("doc_ids", lambda values: ";".join(sorted(set(";".join(map(str, values)).split(";")))[:30])),
    )
    return out


def apply_clustering_edge_weights(edges: pd.DataFrame) -> pd.DataFrame:
    if edges is None or len(edges) == 0:
        return pd.DataFrame(columns=["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    out = edges.copy()
    out["rel"] = out["rel"].astype(str).str.upper()
    out["edge_type_factor"] = out["rel"].map(EDGE_TYPE_WEIGHTS).fillna(0.50)

    out["hub_factor"] = 1.0
    hub_mask = out["source_global"].astype(str).isin(GENERIC_CLUSTER_HUBS) | out["target_global"].astype(str).isin(
        GENERIC_CLUSTER_HUBS
    )
    out.loc[hub_mask, "hub_factor"] = GENERIC_HUB_EDGE_FACTOR

    out["weight_raw"] = out["weight"].astype(float)
    out["weight"] = out["weight_raw"] * out["edge_type_factor"] * out["hub_factor"]
    out["weight"] = out["weight"].clip(lower=0.001)
    return out


def compute_degree(edges: pd.DataFrame) -> pd.DataFrame:
    degree = Counter()
    for _, row in edges.iterrows():
        degree[safe_str(row["source_global"])] += 1
        degree[safe_str(row["target_global"])] += 1
    return pd.DataFrame([{"node_key": node_key, "degree": count} for node_key, count in degree.items()])


def make_cluster_graph_view(
    nodes_full: pd.DataFrame,
    edges_full: pd.DataFrame,
    min_degree: int = DEFAULT_MIN_DEGREE_FOR_COMMUNITY_GRAPH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    degree_df = compute_degree(edges_full)
    if len(degree_df) == 0:
        return nodes_full.iloc[0:0].copy(), edges_full.iloc[0:0].copy(), degree_df

    keep_nodes = set(degree_df.loc[degree_df["degree"] >= min_degree, "node_key"].astype(str))
    nodes_cluster = nodes_full[nodes_full["node_key"].astype(str).isin(keep_nodes)].copy()
    edges_cluster = edges_full[
        edges_full["source_global"].astype(str).isin(keep_nodes)
        & edges_full["target_global"].astype(str).isin(keep_nodes)
    ].copy()

    print("Full graph nodes:", nodes_full.shape)
    print("Full graph edges:", edges_full.shape)
    print("Cluster graph nodes:", nodes_cluster.shape)
    print("Cluster graph edges:", edges_cluster.shape)
    if len(degree_df):
        print("Degree summary:")
        print(degree_df["degree"].describe())
        print("Top degree nodes:")
        print(degree_df.sort_values("degree", ascending=False).head(20))

    return nodes_cluster, edges_cluster, degree_df


def build_igraph(nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[ig.Graph, Dict[str, int], Dict[int, str]]:
    node_keys = nodes["node_key"].astype(str).tolist()
    node_to_idx = {node_key: idx for idx, node_key in enumerate(node_keys)}
    idx_to_node = {idx: node_key for node_key, idx in node_to_idx.items()}

    edge_tuples = []
    weights = []
    relations = []
    for _, row in edges.iterrows():
        source = safe_str(row["source_global"])
        target = safe_str(row["target_global"])
        if source in node_to_idx and target in node_to_idx and source != target:
            edge_tuples.append((node_to_idx[source], node_to_idx[target]))
            weights.append(float(row.get("weight", 1.0)))
            relations.append(safe_str(row.get("rel", "RELATED")))

    graph = ig.Graph(n=len(node_keys), edges=edge_tuples, directed=False)
    graph.vs["node_key"] = node_keys
    if "label" in nodes.columns:
        label_map = dict(zip(nodes["node_key"].astype(str), nodes["label"].astype(str)))
        graph.vs["label"] = [label_map.get(node_key, node_key) for node_key in node_keys]
    if "node_type" in nodes.columns:
        type_map = dict(zip(nodes["node_key"].astype(str), nodes["node_type"].astype(str)))
        graph.vs["node_type"] = [type_map.get(node_key, "") for node_key in node_keys]
    graph.es["weight"] = weights
    graph.es["rel"] = relations
    return graph, node_to_idx, idx_to_node


def run_leiden(graph: ig.Graph, resolution: float = DEFAULT_LEIDEN_RESOLUTION, seed: int = 42) -> List[int]:
    if graph.vcount() == 0:
        return []
    if graph.ecount() == 0:
        return list(range(graph.vcount()))

    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights=graph.es["weight"] if "weight" in graph.es.attributes() else None,
        resolution_parameter=resolution,
        seed=seed,
    )
    return list(partition.membership)


def summarize_communities(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    membership: List[int],
    level_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = nodes.copy()
    tmp[f"{level_name}_id"] = membership
    node_to_comm = tmp[["node_key", f"{level_name}_id"]].copy()

    label_col = "label" if "label" in tmp.columns else "node_key"
    type_col = "node_type" if "node_type" in tmp.columns else None
    mention_col = "mention_count" if "mention_count" in tmp.columns else "member_count" if "member_count" in tmp.columns else None

    group_lookup = dict(zip(node_to_comm["node_key"].astype(str), node_to_comm[f"{level_name}_id"]))
    internal_strength: Dict[str, float] = defaultdict(float)
    if edges is not None and len(edges):
        for _, row in edges.iterrows():
            source = safe_str(row.get("source_global", ""))
            target = safe_str(row.get("target_global", ""))
            if not source or not target:
                continue
            source_group = group_lookup.get(source)
            target_group = group_lookup.get(target)
            if source_group is None or target_group is None or source_group != target_group:
                continue
            weight = float(row.get("weight", 1.0))
            internal_strength[source] += weight
            internal_strength[target] += weight

    rows: List[Dict[str, object]] = []
    for community_id, group in tmp.groupby(f"{level_name}_id"):
        scored_rows = []
        for _, row in group.iterrows():
            node_key = safe_str(row.get("node_key", ""))
            label = norm_space(row.get(label_col, ""))
            if not label:
                continue
            internal = float(internal_strength.get(node_key, 0.0))
            mention_bonus = 0.0
            if mention_col:
                try:
                    mention_bonus = math.log1p(float(row.get(mention_col, 0.0) or 0.0)) * 0.05
                except Exception:
                    mention_bonus = 0.0
            scored_rows.append((internal + mention_bonus, label))

        ranked_labels = []
        seen_labels: set[str] = set()
        for _score, label in sorted(scored_rows, key=lambda item: (-item[0], item[1])):
            if label not in seen_labels:
                ranked_labels.append(label)
                seen_labels.add(label)
        if not ranked_labels:
            label_counts = summarized_label_counts(group, label_col=label_col)
            ranked_labels = [label for label, _ in label_counts.most_common()]
        non_generic_labels = [label for label in ranked_labels if label.upper().strip() not in GENERIC_META_TOKENS]
        if not non_generic_labels:
            non_generic_labels = ranked_labels
        top_labels = non_generic_labels[:8]
        fallback_label = f"{level_name.upper()} {community_id}"
        display_label = make_hierarchy_display_label(top_labels, fallback=fallback_label) if top_labels else fallback_label
        analyst_label = make_theme_family(join_top_labels(top_labels, max_items=8), fallback=display_label)
        node_types = group[type_col].astype(str).value_counts().to_dict() if type_col else {}
        rows.append(
            {
                f"{level_name}_id": community_id,
                "size": len(group),
                "top_labels": join_top_labels(top_labels, max_items=8),
                "top_types": json.dumps(node_types),
                "display_label": display_label,
                "analyst_label": analyst_label,
            }
        )

    summary = pd.DataFrame(rows).sort_values("size", ascending=False)
    return node_to_comm, summary


def build_meta_graph_from_partition(
    lower_nodes: pd.DataFrame,
    lower_edges: pd.DataFrame,
    node_to_group: pd.DataFrame,
    group_col: str,
    next_node_prefix: str,
    min_meta_edge_weight: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["source_global", "target_global", "rel", "weight", "support_count", "doc_ids"]

    if lower_nodes is None or len(lower_nodes) == 0:
        return (
            pd.DataFrame(columns=["node_key", "label", "node_type", "member_count", "top_labels"]),
            pd.DataFrame(columns=cols),
        )

    node_to_group_id = dict(zip(node_to_group["node_key"].astype(str), node_to_group[group_col]))
    label_col = "label" if "label" in lower_nodes.columns else "node_key"
    lower_tmp = lower_nodes.copy()
    lower_tmp["group_id"] = lower_tmp["node_key"].astype(str).map(node_to_group_id)
    lower_tmp = lower_tmp.dropna(subset=["group_id"]).copy()

    if len(lower_tmp) == 0:
        return (
            pd.DataFrame(columns=["node_key", "label", "node_type", "member_count", "top_labels"]),
            pd.DataFrame(columns=cols),
        )

    group_neighbor_sets: Dict[object, set[object]] = defaultdict(set)
    for _, row in lower_edges.iterrows():
        source = safe_str(row["source_global"])
        target = safe_str(row["target_global"])
        group_source = node_to_group_id.get(source)
        group_target = node_to_group_id.get(target)
        if group_source is None or group_target is None or group_source == group_target:
            continue
        group_neighbor_sets[group_source].add(group_target)
        group_neighbor_sets[group_target].add(group_source)

    total_groups = max(1, lower_tmp["group_id"].nunique())
    group_idf: Dict[object, float] = {}
    for group_id in lower_tmp["group_id"].unique():
        df = len(group_neighbor_sets.get(group_id, set()))
        group_idf[group_id] = math.log((1 + total_groups) / (1 + df)) + 1.0

    meta_edge_rows: List[Dict[str, object]] = []
    for _, row in lower_edges.iterrows():
        source = safe_str(row["source_global"])
        target = safe_str(row["target_global"])
        group_source = node_to_group_id.get(source)
        group_target = node_to_group_id.get(target)
        if group_source is None or group_target is None or group_source == group_target:
            continue

        left, right = canonical_pair(f"{next_node_prefix}:{group_source}", f"{next_node_prefix}:{group_target}")
        base_weight = float(row.get("weight", 1.0))
        support_count = int(row.get("support_count", 1))
        idf_factor = min(group_idf.get(group_source, 1.0), group_idf.get(group_target, 1.0))
        support_factor = math.log1p(max(1, support_count))
        final_weight = base_weight * idf_factor * support_factor
        meta_edge_rows.append(
            {
                "source_global": left,
                "target_global": right,
                "rel": "META_LINK",
                "weight": final_weight,
                "support_count": support_count,
                "doc_ids": safe_str(row.get("doc_ids", "")),
            }
        )

    group_labels: List[Dict[str, object]] = []
    for group_id, group in lower_tmp.groupby("group_id"):
        label_counts = summarized_label_counts(group, label_col=label_col)
        ranked_labels = [label for label, _ in label_counts.most_common()]
        non_generic = [label for label in ranked_labels if label.upper().strip() not in GENERIC_META_TOKENS]
        if not non_generic:
            non_generic = ranked_labels
        top_labels = non_generic[:8]
        fallback_label = f"{next_node_prefix}:{group_id}"
        display_label = make_hierarchy_display_label(top_labels, fallback=fallback_label) if top_labels else fallback_label
        group_labels.append(
            {
                "node_key": f"{next_node_prefix}:{group_id}",
                "label": display_label,
                "node_type": next_node_prefix,
                "member_count": len(group),
                "top_labels": join_top_labels(top_labels, max_items=8),
            }
        )

    meta_nodes = pd.DataFrame(group_labels)
    if not meta_edge_rows:
        return meta_nodes, pd.DataFrame(columns=cols)

    meta_edges = pd.DataFrame(meta_edge_rows).groupby(["source_global", "target_global", "rel"], as_index=False).agg(
        weight=("weight", "sum"),
        support_count=("support_count", "sum"),
        doc_ids=("doc_ids", lambda values: ";".join(sorted(set(";".join(map(str, values)).split(";")))[:30])),
    )
    meta_edges["weight_raw_meta"] = meta_edges["weight"].astype(float)
    meta_edges["weight_log_meta"] = np.log1p(meta_edges["weight_raw_meta"])

    strength_by_node: Dict[str, float] = defaultdict(float)
    for _, row in meta_edges.iterrows():
        source = safe_str(row["source_global"])
        target = safe_str(row["target_global"])
        weight = float(row["weight_log_meta"])
        strength_by_node[source] += weight
        strength_by_node[target] += weight

    rank_by_node: Dict[str, Dict[int, int]] = {}
    tmp = meta_edges.reset_index(drop=True)
    neighbor_rows: Dict[str, List[Tuple[float, str, int]]] = defaultdict(list)
    for idx, row in tmp.iterrows():
        source = safe_str(row["source_global"])
        target = safe_str(row["target_global"])
        weight = float(row["weight_log_meta"])
        neighbor_rows[source].append((weight, target, idx))
        neighbor_rows[target].append((weight, source, idx))

    for node_key, values in neighbor_rows.items():
        ranked = sorted(values, key=lambda item: (-item[0], item[1]))
        rank_by_node[node_key] = {edge_idx: rank + 1 for rank, (_, _, edge_idx) in enumerate(ranked)}

    mutual_strengths: List[float] = []
    rank_scores: List[float] = []
    combined_scores: List[float] = []
    for idx, row in tmp.iterrows():
        source = safe_str(row["source_global"])
        target = safe_str(row["target_global"])
        weight = float(row["weight_log_meta"])
        source_strength = max(strength_by_node.get(source, 0.0), 1e-9)
        target_strength = max(strength_by_node.get(target, 0.0), 1e-9)
        mutual_strength = weight / math.sqrt(source_strength * target_strength)

        source_rank = rank_by_node[source][idx]
        target_rank = rank_by_node[target][idx]
        rank_score = 1.0 / math.sqrt(float(source_rank * target_rank))

        combined_score = math.sqrt(mutual_strength * rank_score)
        mutual_strengths.append(mutual_strength)
        rank_scores.append(rank_score)
        combined_scores.append(combined_score)

    meta_edges = tmp
    meta_edges["weight_mutual"] = mutual_strengths
    meta_edges["weight_rank"] = rank_scores
    meta_edges["weight_combined"] = combined_scores
    meta_edges["weight"] = meta_edges["weight_combined"]
    meta_edges = meta_edges[meta_edges["weight"] >= min_meta_edge_weight].copy()
    return meta_nodes, meta_edges


def build_recursive_hierarchy(
    level0_nodes: pd.DataFrame,
    level0_edges: pd.DataFrame,
    max_levels: int = DEFAULT_MAX_HIERARCHY_LEVELS,
    resolution: float = DEFAULT_LEIDEN_RESOLUTION,
) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}
    current_nodes = level0_nodes.copy()
    current_edges = level0_edges.copy()

    outputs["level0_nodes"] = current_nodes.copy()
    outputs["level0_edges"] = current_edges.copy()

    for level in range(1, max_levels + 1):
        print(f"\n--- Hierarchy level {level} ---")
        if len(current_nodes) == 0:
            print("No nodes left; stopping hierarchy.")
            break

        graph, _, _ = build_igraph(current_nodes, current_edges)
        level_resolution = LEIDEN_RESOLUTION_BY_LEVEL.get(level, resolution)
        print(f"Leiden resolution for level {level}: {level_resolution}")
        membership = run_leiden(graph, resolution=level_resolution, seed=42 + level)

        node_to_comm, summary = summarize_communities(current_nodes, current_edges, membership, f"level{level}")
        outputs[f"node_to_level{level}"] = node_to_comm.copy()
        outputs[f"level{level}_summary"] = summary.copy()

        print(f"Level {level} nodes:", len(current_nodes))
        print(f"Level {level} communities:", summary.shape[0])
        if len(summary):
            print(summary.head(10))

        if summary.shape[0] <= 1:
            break

        next_prefix = f"LEVEL{level}_COMMUNITY"
        min_meta_edge_weight = META_EDGE_MIN_WEIGHT_BY_LEVEL.get(level, 0.25)
        print(f"Meta-edge threshold for level {level}: {min_meta_edge_weight}")

        meta_nodes, meta_edges = build_meta_graph_from_partition(
            current_nodes,
            current_edges,
            node_to_comm,
            f"level{level}_id",
            next_prefix,
            min_meta_edge_weight=min_meta_edge_weight,
        )
        outputs[f"level{level}_meta_nodes"] = meta_nodes.copy()
        outputs[f"level{level}_meta_edges"] = meta_edges.copy()

        current_nodes = meta_nodes.copy()
        current_edges = meta_edges.copy()

        if len(current_edges) == 0:
            print("Meta graph has no edges; stopping hierarchy.")
            break

    return outputs


def save_outputs(
    graph_dir: Path,
    nodes_full: pd.DataFrame,
    edges_full: pd.DataFrame,
    nodes_cluster: pd.DataFrame,
    edges_cluster: pd.DataFrame,
    degree_df: pd.DataFrame,
    canonical_entity_mentions: pd.DataFrame,
    hierarchy_outputs: Dict[str, pd.DataFrame],
) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)

    nodes_full.to_csv(graph_dir / "canonical_nodes_full.csv", index=False)
    edges_full.to_csv(graph_dir / "analysis_edges_full.csv", index=False)
    nodes_cluster.to_csv(graph_dir / "canonical_nodes_cluster.csv", index=False)
    edges_cluster.to_csv(graph_dir / "analysis_edges_cluster.csv", index=False)
    degree_df.to_csv(graph_dir / "degree_diagnostics.csv", index=False)

    nodes_cluster.to_csv(graph_dir / "canonical_nodes.csv", index=False)
    edges_cluster.to_csv(graph_dir / "analysis_edges.csv", index=False)
    canonical_entity_mentions.to_csv(graph_dir / "canonical_entity_mentions.csv", index=False)

    for name, df in hierarchy_outputs.items():
        df.to_csv(graph_dir / f"{name}.csv", index=False)

    try:
        graph, _, _ = build_igraph(nodes_cluster, edges_cluster)
        if "node_to_level1" in hierarchy_outputs:
            level1_map = dict(
                zip(
                    hierarchy_outputs["node_to_level1"]["node_key"].astype(str),
                    hierarchy_outputs["node_to_level1"]["level1_id"],
                )
            )
            graph.vs["community"] = [int(level1_map.get(node_key, -1)) for node_key in graph.vs["node_key"]]
        graph.write_graphml(str(graph_dir / "analysis_graph_cluster.graphml"))
    except Exception as exc:
        print("WARNING: Could not write GraphML:", exc)

    print("\nSaved outputs to:", graph_dir)
    for path in sorted(graph_dir.glob("*.csv")):
        print(" -", path.name)


def write_canonicalization_reports(
    graph_dir: Path,
    alias_df: pd.DataFrame,
    label_frequencies: pd.DataFrame,
    alias_candidates: pd.DataFrame,
) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    alias_df.to_csv(graph_dir / "entity_alias_dictionary_snapshot.csv", index=False)
    label_frequencies.to_csv(graph_dir / "entity_label_frequencies.csv", index=False)
    alias_candidates.to_csv(graph_dir / "entity_alias_review_candidates.csv", index=False)


def main(
    project_dir_arg: Optional[Path] = None,
    entity_dir_arg: Optional[Path] = None,
    min_degree: int = DEFAULT_MIN_DEGREE_FOR_COMMUNITY_GRAPH,
    resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    max_levels: int = DEFAULT_MAX_HIERARCHY_LEVELS,
):
    if project_dir_arg is None:
        parser = argparse.ArgumentParser(description="Build Matryoshka graphs for the CIA corpus.")
        parser.add_argument("PROJECT_DIR", help="Project folder containing extraction/ outputs.")
        parser.add_argument(
            "--entity-dir",
            default=None,
            help="Legacy ignored argument retained for compatibility. Stage B uses Stage A1 outputs or mechanical normalization only.",
        )
        parser.add_argument(
            "--min-degree",
            type=int,
            default=DEFAULT_MIN_DEGREE_FOR_COMMUNITY_GRAPH,
            help="Minimum node degree for the cluster graph.",
        )
        parser.add_argument(
            "--resolution",
            type=float,
            default=DEFAULT_LEIDEN_RESOLUTION,
            help="Leiden resolution parameter.",
        )
        parser.add_argument(
            "--max-levels",
            type=int,
            default=DEFAULT_MAX_HIERARCHY_LEVELS,
            help="Maximum recursive hierarchy depth.",
        )
        args = parser.parse_args()
        project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
        entity_dir_arg = Path(args.entity_dir).expanduser().resolve() if args.entity_dir else None
        min_degree = args.min_degree
        resolution = args.resolution
        max_levels = args.max_levels
    else:
        project_dir = Path(project_dir_arg).expanduser().resolve()

    graph_dir = project_dir / "graphs"
    entity_dir = resolve_entity_directory(project_dir, entity_dir_arg)
    set_a1_entity_resolution_maps(project_dir, entity_dir)

    print("Project directory:", project_dir)
    print("Graph directory:", graph_dir)
    print("Entity directory:", entity_dir)
    print("A1 map loaded:", A1_CANONICAL_MAP_LOADED)
    print("Canonical labels available:", len(A1_CANONICAL_MAP))
    print("min_degree:", min_degree)
    print("resolution:", resolution)
    print("max_levels:", max_levels)

    entities, events, claims, relations = load_extraction_tables(project_dir)
    entities, events, claims = add_local_keys(entities, events, claims)

    canonical_entity_nodes, mention_to_canonical, canonical_entity_mentions = build_canonical_entities(entities)
    event_cluster_nodes, mention_to_event_cluster = build_event_clusters(events)

    nodes_full = pd.concat([canonical_entity_nodes, event_cluster_nodes], ignore_index=True)
    nodes_full = nodes_full.drop_duplicates(subset=["node_key"]).copy()

    relation_edges = build_relation_edges(
        relations,
        entities,
        events,
        mention_to_canonical,
        mention_to_event_cluster,
    )
    cooccurrence_edges = build_cooccurrence_edges(entities, mention_to_canonical)
    event_entity_edges = build_event_entity_edges(events, entities, mention_to_canonical, mention_to_event_cluster)
    edges_full_raw = combine_edges([relation_edges, cooccurrence_edges, event_entity_edges])

    all_edge_nodes = (
        set(edges_full_raw["source_global"].astype(str)) | set(edges_full_raw["target_global"].astype(str))
        if len(edges_full_raw)
        else set()
    )
    known_nodes = set(nodes_full["node_key"].astype(str))
    missing = sorted(all_edge_nodes - known_nodes)
    if missing:
        extra_nodes = pd.DataFrame(
            [
                {
                    "node_key": node_key,
                    "label": node_key.split(":", 1)[-1].replace("_", " "),
                    "label_full": node_key.split(":", 1)[-1].replace("_", " "),
                    "node_type": "CanonicalEntity" if node_key.startswith("CANONICAL_ENTITY") else "EventCluster",
                    "source": "edge_endpoint_fallback",
                }
                for node_key in missing
            ]
        )
        nodes_full = pd.concat([nodes_full, extra_nodes], ignore_index=True).drop_duplicates(subset=["node_key"]).copy()

    nodes_full = apply_concise_node_labels(nodes_full)
    if len(canonical_entity_mentions):
        node_label_map = dict(zip(nodes_full["node_key"].astype(str), nodes_full["label"].astype(str)))
        node_label_full_map = dict(zip(nodes_full["node_key"].astype(str), nodes_full["label_full"].astype(str)))
        canonical_entity_mentions["canonical_label_display"] = (
            canonical_entity_mentions["node_key"].astype(str).map(node_label_map).fillna("")
        )
        canonical_entity_mentions["canonical_label_full"] = (
            canonical_entity_mentions["node_key"].astype(str).map(node_label_full_map).fillna(
                canonical_entity_mentions.get("canonical_label_full", "")
            )
        )
        canonical_entity_mentions["canonical_label"] = canonical_entity_mentions["canonical_label_display"]

    edges_for_clustering = apply_clustering_edge_weights(edges_full_raw)
    nodes_cluster, edges_cluster, degree_df = make_cluster_graph_view(nodes_full, edges_for_clustering, min_degree=min_degree)
    hierarchy_outputs = build_recursive_hierarchy(nodes_cluster, edges_cluster, max_levels=max_levels, resolution=resolution)

    save_outputs(
        graph_dir,
        nodes_full,
        edges_full_raw,
        nodes_cluster,
        edges_cluster,
        degree_df,
        canonical_entity_mentions,
        hierarchy_outputs,
    )

    return {
        "nodes_full": nodes_full,
        "edges_full_raw": edges_full_raw,
        "nodes_cluster": nodes_cluster,
        "edges_cluster": edges_cluster,
        "degree_df": degree_df,
        "canonical_entity_mentions": canonical_entity_mentions,
        "hierarchy_outputs": hierarchy_outputs,
    }


if __name__ == "__main__":
    main()
