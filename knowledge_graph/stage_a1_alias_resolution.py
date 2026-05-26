#!/usr/bin/env python3
"""
Stage A1 alias-resolution worklist builder.

This step sits between Stage A extraction and Stage B graph building.

It builds a structured alias resolution pass from `extraction/entities.csv`
using:
1. Optional project-local alias seeding from `entity_directory/entity_aliases.csv`
2. Deterministic normalization rules
3. Lexical similarity scoring
4. Context-aware similarity scoring from co-mentions and descriptions
5. Optional OpenAI adjudication for ambiguous high-value candidates

Outputs are written to `PROJECT_DIR/entity_resolution/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


KNOWLEDGE_GRAPH_ROOT = Path(__file__).resolve().parent
DEFAULT_ENTITY_DIRECTORY_NAME = "entity_directory"
DEFAULT_ENTITY_ALIAS_FILENAME = "entity_aliases.csv"
DEFAULT_OUTPUT_DIRNAME = "entity_resolution"

TOKEN_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "AT",
    "BY",
    "FOR",
    "FROM",
    "IN",
    "INTO",
    "OF",
    "ON",
    "OR",
    "THE",
    "TO",
    "WITH",
}

PERSON_TITLE_TOKENS = {
    "ADMIRAL",
    "AMBASSADOR",
    "BRIGADIER",
    "CAPTAIN",
    "CHAIRMAN",
    "COL",
    "COLONEL",
    "COMMANDER",
    "DR",
    "DOCTOR",
    "GENERAL",
    "GENERALISSIMO",
    "GEN",
    "KING",
    "MAJOR",
    "MISTER",
    "MR",
    "MRS",
    "MS",
    "PRESIDENT",
    "PREMIER",
    "PRIME",
    "PRINCE",
    "PRINCESS",
    "PROFESSOR",
    "QUEEN",
    "SECRETARY",
    "SIR",
}

DIRECTIONAL_TOKEN_MAP = {
    "NORTH": "NORTH",
    "NORTHERN": "NORTH",
    "SOUTH": "SOUTH",
    "SOUTHERN": "SOUTH",
    "EAST": "EAST",
    "EASTERN": "EAST",
    "WEST": "WEST",
    "WESTERN": "WEST",
}

BRANCH_TOKEN_MAP = {
    "AIR": "AIR",
    "AIRFORCE": "AIR",
    "ARMY": "ARMY",
    "MARINE": "MARINE",
    "NAVAL": "NAVY",
    "NAVY": "NAVY",
}

LOCATION_MARKER_TOKENS = {
    "CAPITAL",
    "CITY",
    "ISLAND",
    "PREFECTURE",
    "PROVINCE",
    "REGION",
    "STATE",
    "TERRITORY",
}

GOVERNMENT_MARKER_TOKENS = {
    "GOVERNMENT",
    "KINGDOM",
    "REGIME",
    "REPUBLIC",
}

ORGANIZATION_MARKER_TOKENS = {
    "AGENCY",
    "BUREAU",
    "COMMISSION",
    "COMMITTEE",
    "CONFERENCE",
    "CONGRESS",
    "COUNCIL",
    "DELEGATION",
    "DEPARTMENT",
    "MINISTRY",
    "MISSION",
    "PRESIDIUM",
    "SECRETARIAT",
}

POLITICAL_MARKER_TOKENS = {
    "BLOC",
    "FRONT",
    "MOVEMENT",
    "NATIONALISTS",
    "PARTY",
    "UNION",
}

MILITARY_MARKER_TOKENS = {
    "AIRBORNE",
    "ARMY",
    "BATTALION",
    "DIVISION",
    "FORCE",
    "FORCES",
    "GUERRILLAS",
    "IRREGULARS",
    "MILITARY",
    "NAVY",
    "REGIMENT",
    "TROOPS",
}

ROLE_MARKER_TOKENS = {
    "ASSISTANT",
    "CHAIRMAN",
    "CHIEF",
    "COUNSELOR",
    "DELEGATE",
    "DEPUTY",
    "DIRECTOR",
    "ENVOY",
    "MINISTER",
    "REPRESENTATIVE",
    "SECRETARY",
}

GENERATED_ALIAS_FILENAME = "entity_aliases_generated.csv"

TYPE_COMPATIBILITY_GROUPS = [
    {"COUNTRY", "GOVERNMENT", "LOCATION"},
    {"ORGANIZATION", "POLITICALGROUP", "MILITARY"},
    {"PERSON"},
    {"DOCUMENT"},
    {"RELIGIOUSGROUP"},
    {"OTHER"},
]

SAFE_RULE_ACTIONS = {
    "leading_article",
    "government_of",
    "parenthetical_base_match",
    "acronym_match",
    "person_title_variant",
    "surface_variant",
    "token_bag_match",
}

OPENAI_PROMPT_VERSION = "stage_a1_alias_resolution_v1"
OPENAI_ACCEPT_CONFIDENCE_THRESHOLD = 0.80

NEGATIVE_DISTINGUISHER_TOKENS = {
    "ACTING",
    "AIR",
    "ARMY",
    "DEPUTY",
    "EAST",
    "FORCE",
    "INTERIM",
    "JOINT",
    "MARINE",
    "NAVAL",
    "NAVY",
    "NORTH",
    "SOUTH",
    "UNDER",
    "VICE",
    "WEST",
}


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", safe_str(value)).strip()


def normalize_label_basic(text: object) -> str:
    normalized = norm_space(text)
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = normalized.upper()
    normalized = re.sub(r"[\"'`]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def keyify(text: object) -> str:
    normalized = normalize_label_basic(text)
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "UNKNOWN"


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


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


def strip_leading_article(label: str) -> str:
    stripped = re.sub(r"^(THE|A|AN)\s+", "", label).strip()
    return stripped or label


def strip_trailing_parenthetical(label: str) -> str:
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    return stripped or label


def acronym_from_label(label: str) -> str:
    tokens = [
        token
        for token in re.split(r"[\s\-]+", label)
        if token and token not in TOKEN_STOPWORDS
    ]
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens if token and token[0].isalpha())


def looks_like_acronym(label: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", label)
    return compact == label and 2 <= len(label) <= 6


def token_set(label: str) -> Set[str]:
    return {
        token
        for token in re.split(r"[^A-Z0-9]+", normalize_label_basic(label))
        if token and token not in TOKEN_STOPWORDS and len(token) > 1
    }


def token_list(label: str) -> List[str]:
    return [
        token
        for token in re.split(r"[^A-Z0-9]+", normalize_label_basic(label))
        if token
    ]


def singularize_token(token: str) -> str:
    if token.endswith("IES") and len(token) > 4:
        return token[:-3] + "Y"
    if token.endswith("S") and len(token) > 4 and not token.endswith(("SS", "US", "IS")):
        return token[:-1]
    return token


def variant_surface_key(label: str) -> str:
    return " ".join(token for token in token_list(label) if token not in {"A", "AN", "THE"})


def token_bag_key(label: str) -> Tuple[str, ...]:
    return tuple(sorted(token for token in token_list(label) if token not in {"A", "AN", "THE"}))


def strip_person_titles(label: str) -> str:
    tokens = token_list(label)
    while tokens and tokens[0] in PERSON_TITLE_TOKENS:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def is_person_name_like(label: str, source_types: Set[str]) -> bool:
    stripped = strip_person_titles(label)
    tokens = token_list(stripped)
    if not tokens or len(tokens) > 5:
        return False
    if "PERSON" not in source_types and not (token_list(label) and token_list(label)[0] in PERSON_TITLE_TOKENS):
        return False
    if "OF" in tokens:
        return False
    if token_set(stripped) & (
        LOCATION_MARKER_TOKENS
        | GOVERNMENT_MARKER_TOKENS
        | ORGANIZATION_MARKER_TOKENS
        | POLITICAL_MARKER_TOKENS
        | MILITARY_MARKER_TOKENS
        | ROLE_MARKER_TOKENS
    ):
        return False
    return True


def person_core_key(label: str, source_types: Set[str]) -> str:
    if not is_person_name_like(label, source_types):
        return ""
    return variant_surface_key(strip_person_titles(label))


def directional_signature(label: str) -> Set[str]:
    return {
        DIRECTIONAL_TOKEN_MAP[token]
        for token in token_list(label)
        if token in DIRECTIONAL_TOKEN_MAP
    }


def branch_signature(label: str) -> Set[str]:
    tokens = token_list(label)
    signature = {
        BRANCH_TOKEN_MAP[token]
        for token in tokens
        if token in BRANCH_TOKEN_MAP
    }
    if "AIR" in tokens and "FORCE" in tokens:
        signature.add("AIR")
    return signature


def infer_structural_family(label: str, source_types: Set[str]) -> str:
    tokens = token_set(label)

    if "DOCUMENT" in source_types:
        return "document"
    if tokens & LOCATION_MARKER_TOKENS or "LOCATION" in source_types:
        return "location"
    if is_person_name_like(label, source_types):
        return "person_name"
    if tokens & GOVERNMENT_MARKER_TOKENS:
        return "government"
    if tokens & MILITARY_MARKER_TOKENS:
        return "military"
    if tokens & POLITICAL_MARKER_TOKENS:
        return "political_group"
    if tokens & ORGANIZATION_MARKER_TOKENS:
        return "organization"
    if tokens & ROLE_MARKER_TOKENS:
        return "role"
    if "COUNTRY" in source_types:
        return "country"
    if "GOVERNMENT" in source_types:
        return "government"
    if "MILITARY" in source_types:
        return "military"
    if "POLITICALGROUP" in source_types:
        return "political_group"
    if "ORGANIZATION" in source_types:
        return "organization"
    if "RELIGIOUSGROUP" in source_types:
        return "religious_group"
    if "PERSON" in source_types:
        return "person"
    return "other"


def structural_conflict_reason(
    alias_label: str,
    candidate_label: str,
    alias_types: Set[str],
    candidate_types: Set[str],
) -> str:
    alias_family = infer_structural_family(alias_label, alias_types)
    candidate_family = infer_structural_family(candidate_label, candidate_types)

    person_families = {"person", "person_name"}
    entity_families = {"country", "document", "government", "location", "military", "organization", "political_group", "religious_group", "role"}

    if alias_family in person_families and candidate_family in entity_families:
        return "person_vs_nonperson"
    if candidate_family in person_families and alias_family in entity_families:
        return "person_vs_nonperson"

    if {alias_family, candidate_family} == {"country", "government"}:
        return "country_vs_government"
    if {alias_family, candidate_family} == {"location", "government"}:
        return "location_vs_government"
    if {alias_family, candidate_family} == {"location", "country"}:
        return "location_vs_country"
    if {alias_family, candidate_family} == {"organization", "military"}:
        return "organization_vs_military"
    if {alias_family, candidate_family} == {"political_group", "military"}:
        return "political_group_vs_military"
    if {alias_family, candidate_family} == {"location", "military"}:
        return "location_vs_military"

    alias_dirs = directional_signature(alias_label)
    candidate_dirs = directional_signature(candidate_label)
    if alias_dirs and candidate_dirs and alias_dirs != candidate_dirs:
        return "directional_conflict"

    alias_branches = branch_signature(alias_label)
    candidate_branches = branch_signature(candidate_label)
    if alias_branches and candidate_branches and alias_branches != candidate_branches:
        return "branch_conflict"

    return ""


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def cooccurrence_jaccard(text_a: str, text_b: str) -> float:
    return jaccard(
        [item.strip() for item in safe_str(text_a).split(";") if item.strip()],
        [item.strip() for item in safe_str(text_b).split(";") if item.strip()],
    )


def type_set_from_text(text: object) -> Set[str]:
    raw = safe_str(text)
    if not raw:
        return set()
    return {normalize_label_basic(part) for part in raw.split(";") if normalize_label_basic(part)}


def type_group(type_name: str) -> Optional[int]:
    normalized = normalize_label_basic(type_name)
    for idx, group in enumerate(TYPE_COMPATIBILITY_GROUPS):
        if normalized in group:
            return idx
    return None


def type_compatibility(source_types: Set[str], target_types: Set[str]) -> Tuple[float, str]:
    if not source_types or not target_types:
        return 0.45, "missing_type_info"

    if source_types & target_types:
        return 1.0, "shared_type"

    source_groups = {type_group(value) for value in source_types}
    target_groups = {type_group(value) for value in target_types}
    source_groups.discard(None)
    target_groups.discard(None)

    if source_groups and target_groups and source_groups & target_groups:
        return 0.82, "same_type_group"

    return 0.0, "type_mismatch"


def load_entities(project_dir: Path) -> pd.DataFrame:
    path = project_dir / "extraction" / "entities.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage A entity output: {path}")

    entities = pd.read_csv(path)
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    if label_col is None:
        raise KeyError(f"Could not find entity label column in {path}. Columns: {entities.columns.tolist()}")

    print("Loaded entities:", entities.shape)
    return entities


def build_label_catalog(
    entities: pd.DataFrame,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    type_col = first_existing_col(entities, ["type", "entity_type", "broad_type"])
    desc_col = first_existing_col(entities, ["description", "summary", "evidence"])

    rows: List[Dict[str, object]] = []
    paragraph_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for _, row in entities.iterrows():
        raw_label = norm_space(row.get(label_col, ""))
        normalized_label = normalize_label_basic(raw_label)
        if not normalized_label:
            continue

        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        source_type = norm_space(row.get(type_col, "")) if type_col else ""
        description = norm_space(row.get(desc_col, "")) if desc_col else ""

        rows.append(
            {
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "raw_label": raw_label,
                "normalized_label": normalized_label,
                "canonical_label_current": canonicalize_entity_label(normalized_label, alias_map),
                "source_type": source_type,
                "description": description,
            }
        )
        paragraph_groups[(doc_id, paragraph_id)].append(normalized_label)

    if not rows:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "canonical_label_current",
                "mention_count",
                "unique_docs",
                "unique_paragraphs",
                "source_types",
                "primary_type",
                "example_mentions",
                "example_descriptions",
                "top_co_mentions",
                "context_text",
                "in_alias_dictionary",
            ]
        )

    tmp = pd.DataFrame(rows)

    co_mentions: Dict[str, Counter] = defaultdict(Counter)
    for labels in paragraph_groups.values():
        unique_labels = sorted(set(labels))
        for left_idx, left in enumerate(unique_labels):
            for right_idx, right in enumerate(unique_labels):
                if left_idx == right_idx:
                    continue
                co_mentions[left][right] += 1

    grouped_rows: List[Dict[str, object]] = []
    for normalized_label, group in tmp.groupby("normalized_label"):
        example_mentions = pd.unique(group["raw_label"].astype(str)).tolist()[:10]
        example_descriptions = [value for value in pd.unique(group["description"].astype(str)).tolist() if value][:8]
        type_counter = Counter(value for value in group["source_type"].astype(str) if value)
        source_types = sorted(type_counter.keys())
        top_co_mentions = [
            label
            for label, _count in co_mentions.get(normalized_label, Counter()).most_common(12)
        ]
        context_bits = [
            normalized_label,
            " ".join(source_types[:8]),
            " ".join(top_co_mentions[:10]),
            " ".join(example_descriptions[:6]),
        ]
        grouped_rows.append(
            {
                "normalized_label": normalized_label,
                "canonical_label_current": canonicalize_entity_label(normalized_label, alias_map),
                "mention_count": int(len(group)),
                "unique_docs": int(group["doc_id"].astype(str).nunique()),
                "unique_paragraphs": int(group[["doc_id", "paragraph_id"]].astype(str).drop_duplicates().shape[0]),
                "source_types": "; ".join(source_types[:10]),
                "primary_type": type_counter.most_common(1)[0][0] if type_counter else "",
                "example_mentions": "; ".join(example_mentions),
                "example_descriptions": "; ".join(example_descriptions),
                "top_co_mentions": "; ".join(top_co_mentions),
                "context_text": " || ".join(bit for bit in context_bits if bit).strip(),
                "in_alias_dictionary": normalized_label in alias_map,
            }
        )

    catalog = pd.DataFrame(grouped_rows)
    catalog = catalog.sort_values(
        ["mention_count", "unique_docs", "normalized_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return catalog


def choose_canonical_label(alias_row: pd.Series, candidate_row: pd.Series) -> str:
    candidate_canonical = safe_str(candidate_row.get("canonical_label_current", ""))
    if candidate_canonical:
        return candidate_canonical
    candidate_label = safe_str(candidate_row.get("normalized_label", ""))
    if candidate_label:
        return candidate_label
    return safe_str(alias_row.get("normalized_label", ""))


def deterministic_candidates_for_label(
    row: pd.Series,
    catalog_lookup: Dict[str, pd.Series],
    labels: Set[str],
    surface_key_lookup: Dict[str, List[str]],
    token_bag_lookup: Dict[Tuple[str, ...], List[str]],
    person_core_lookup: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    normalized_label = safe_str(row["normalized_label"])
    candidates: List[Tuple[str, str]] = []

    stripped = strip_leading_article(normalized_label)
    if stripped != normalized_label and stripped in labels:
        candidates.append((stripped, "leading_article"))

    if normalized_label.startswith("GOVERNMENT OF "):
        tail = normalized_label[len("GOVERNMENT OF ") :].strip()
        if tail and tail in labels:
            candidates.append((tail, "government_of"))

    base = strip_trailing_parenthetical(normalized_label)
    if base != normalized_label and base in labels:
        candidates.append((base, "parenthetical_base_match"))

    if looks_like_acronym(normalized_label):
        matches = [
            label
            for label in labels
            if label != normalized_label and acronym_from_label(label) == normalized_label
        ]
        if matches:
            best = max(
                matches,
                key=lambda label: (
                    int(catalog_lookup[label]["mention_count"]),
                    len(label),
                ),
            )
            candidates.append((best, "acronym_match"))

    surface_matches = [
        label
        for label in surface_key_lookup.get(variant_surface_key(normalized_label), [])
        if label != normalized_label
    ]
    for label in surface_matches:
        candidates.append((label, "surface_variant"))

    bag_matches = [
        label
        for label in token_bag_lookup.get(token_bag_key(normalized_label), [])
        if label != normalized_label
    ]
    for label in bag_matches:
        candidates.append((label, "token_bag_match"))

    source_types = type_set_from_text(row.get("source_types", ""))
    person_key = person_core_key(normalized_label, source_types)
    if person_key:
        matches = [
            label
            for label in person_core_lookup.get(person_key, [])
            if label != normalized_label
        ]
        for label in matches:
            candidates.append((label, "person_title_variant"))

    deduped: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for label, reason in candidates:
        key = (label, reason)
        if key not in seen:
            seen.add(key)
            deduped.append((label, reason))
    return deduped


def should_skip_candidate_direction(alias_label: str, candidate_label: str) -> bool:
    if not alias_label or not candidate_label or alias_label == candidate_label:
        return False

    if strip_leading_article(candidate_label) == alias_label and candidate_label != alias_label:
        return True

    if strip_trailing_parenthetical(candidate_label) == alias_label and candidate_label != alias_label:
        return True

    if looks_like_acronym(candidate_label) and acronym_from_label(alias_label) == candidate_label:
        return True

    return False


def canonical_preference_key(row: pd.Series) -> Tuple[int, int, int, int, int, str]:
    label = safe_str(row.get("normalized_label", ""))
    source_types = type_set_from_text(row.get("source_types", ""))
    family = infer_structural_family(label, source_types)
    family_rank = {
        "country": 4,
        "government": 4,
        "organization": 4,
        "political_group": 4,
        "location": 4,
        "person_name": 3,
        "military": 3,
        "role": 2,
        "person": 2,
        "other": 1,
        "document": 1,
        "religious_group": 1,
    }.get(family, 0)
    return (
        int(row.get("mention_count", 0)),
        int(row.get("unique_docs", 0)),
        family_rank,
        0 if looks_like_acronym(label) else 1,
        len(token_set(label)),
        label,
    )


def should_keep_candidate_direction(alias_row: pd.Series, candidate_row: pd.Series) -> bool:
    alias_key = canonical_preference_key(alias_row)
    candidate_key = canonical_preference_key(candidate_row)
    return candidate_key > alias_key


def build_similarity_models(
    catalog: pd.DataFrame,
    top_k_neighbors: int,
) -> Tuple[TfidfVectorizer, np.ndarray, NearestNeighbors, TfidfVectorizer, np.ndarray, NearestNeighbors]:
    label_texts = catalog["normalized_label"].fillna("UNKNOWN").astype(str).tolist()
    label_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False, min_df=1)
    label_matrix = label_vectorizer.fit_transform(label_texts)
    label_neighbors = NearestNeighbors(
        n_neighbors=min(top_k_neighbors + 1, len(catalog)),
        metric="cosine",
        algorithm="brute",
    )
    label_neighbors.fit(label_matrix)

    context_texts = catalog["context_text"].fillna("UNKNOWN").astype(str).tolist()
    context_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
    context_matrix = context_vectorizer.fit_transform(context_texts)
    context_neighbors = NearestNeighbors(
        n_neighbors=min(top_k_neighbors + 1, len(catalog)),
        metric="cosine",
        algorithm="brute",
    )
    context_neighbors.fit(context_matrix)

    return (
        label_vectorizer,
        label_matrix,
        label_neighbors,
        context_vectorizer,
        context_matrix,
        context_neighbors,
    )


def neighbor_index_map(
    matrix: np.ndarray,
    neighbors: NearestNeighbors,
) -> Dict[int, List[int]]:
    distances, indices = neighbors.kneighbors(matrix)
    out: Dict[int, List[int]] = {}
    for row_idx in range(len(indices)):
        out[row_idx] = [int(idx) for idx in indices[row_idx] if int(idx) != row_idx]
    return out


def sparse_cosine(matrix: np.ndarray, left_idx: int, right_idx: int) -> float:
    return float(matrix[left_idx].multiply(matrix[right_idx]).sum())


def score_candidate_pair(
    alias_row: pd.Series,
    candidate_row: pd.Series,
    label_matrix: np.ndarray,
    context_matrix: np.ndarray,
    alias_idx: int,
    candidate_idx: int,
    rule_reason: Optional[str],
) -> Dict[str, object]:
    alias_label = safe_str(alias_row["normalized_label"])
    candidate_label = safe_str(candidate_row["normalized_label"])
    alias_types = type_set_from_text(alias_row.get("source_types", ""))
    candidate_types = type_set_from_text(candidate_row.get("source_types", ""))
    alias_family = infer_structural_family(alias_label, alias_types)
    candidate_family = infer_structural_family(candidate_label, candidate_types)
    hard_conflict_reason = structural_conflict_reason(alias_label, candidate_label, alias_types, candidate_types)

    lexical_cosine = sparse_cosine(label_matrix, alias_idx, candidate_idx)
    context_cosine = sparse_cosine(context_matrix, alias_idx, candidate_idx)
    sequence_ratio = SequenceMatcher(None, alias_label, candidate_label).ratio()
    token_jaccard = jaccard(token_set(alias_label), token_set(candidate_label))
    alias_tokens = token_set(alias_label)
    candidate_tokens = token_set(candidate_label)
    co_mention_score = cooccurrence_jaccard(alias_row.get("top_co_mentions", ""), candidate_row.get("top_co_mentions", ""))
    type_score, type_note = type_compatibility(
        alias_types,
        candidate_types,
    )

    subset_bonus = 0.05 if token_set(alias_label) and (
        alias_tokens.issubset(candidate_tokens)
        or candidate_tokens.issubset(alias_tokens)
    ) else 0.0

    negative_diff = (alias_tokens ^ candidate_tokens) & NEGATIVE_DISTINGUISHER_TOKENS
    negative_penalty = 0.18 if negative_diff else 0.0

    freq_delta = max(
        -0.05,
        min(
            0.05,
            (math.log1p(float(candidate_row.get("mention_count", 1))) - math.log1p(float(alias_row.get("mention_count", 1)))) / 12.0,
        ),
    )

    final_score = (
        0.36 * lexical_cosine
        + 0.18 * sequence_ratio
        + 0.14 * token_jaccard
        + 0.18 * context_cosine
        + 0.08 * co_mention_score
        + 0.06 * type_score
        + subset_bonus
        + freq_delta
        - negative_penalty
    )

    if rule_reason == "leading_article":
        final_score = max(final_score, 0.96)
    elif rule_reason == "government_of":
        final_score = max(final_score, 0.95)
    elif rule_reason == "parenthetical_base_match":
        final_score = max(final_score, 0.91)
    elif rule_reason == "acronym_match":
        final_score = max(final_score, 0.90)
    elif rule_reason == "person_title_variant":
        final_score = max(final_score, 0.95)
    elif rule_reason in {"surface_variant", "token_bag_match"}:
        final_score = max(final_score, 0.94)

    if hard_conflict_reason:
        final_score = min(final_score, 0.24)

    final_score = max(0.0, min(1.0, final_score))

    return {
        "alias_family": alias_family,
        "matched_label_family": candidate_family,
        "lexical_cosine": round(lexical_cosine, 4),
        "context_cosine": round(context_cosine, 4),
        "sequence_ratio": round(sequence_ratio, 4),
        "token_jaccard": round(token_jaccard, 4),
        "co_mention_jaccard": round(co_mention_score, 4),
        "type_compatibility_score": round(type_score, 4),
        "type_compatibility_note": type_note,
        "negative_distinguishers": ";".join(sorted(negative_diff)),
        "hard_conflict_reason": hard_conflict_reason,
        "final_score": round(final_score, 4),
    }


def initial_recommended_action(
    final_score: float,
    rule_reason: Optional[str],
    type_score: float,
    alias_mentions: int,
    candidate_mentions: int,
    lexical_cosine: float,
    sequence_ratio: float,
    token_jaccard: float,
    hard_conflict_reason: str,
    openai_min_score: float,
    review_min_score: float,
) -> str:
    if hard_conflict_reason:
        return "reject_structural_mismatch"
    if rule_reason in SAFE_RULE_ACTIONS and type_score >= 0.45:
        return "accept_rule"
    if final_score >= 0.92 and type_score >= 0.82 and candidate_mentions >= alias_mentions:
        return "accept_heuristic"
    if (
        final_score >= openai_min_score
        and type_score > 0.0
        and lexical_cosine >= 0.76
        and sequence_ratio >= 0.68
        and token_jaccard >= 0.40
    ):
        return "needs_openai_review"
    if final_score >= review_min_score and type_score > 0.0:
        return "needs_openai_review"
    return "reject_low_score"


def build_candidate_worklist(
    catalog: pd.DataFrame,
    alias_map: Dict[str, str],
    top_k_neighbors: int,
    max_candidates_per_label: int,
    openai_min_score: float,
    review_min_score: float,
) -> pd.DataFrame:
    if len(catalog) == 0:
        return pd.DataFrame()

    catalog = catalog.copy().reset_index(drop=True)
    catalog["catalog_idx"] = np.arange(len(catalog))
    catalog_lookup = {
        safe_str(row["normalized_label"]): row
        for _, row in catalog.iterrows()
    }
    labels = set(catalog_lookup.keys())
    surface_key_lookup: Dict[str, List[str]] = defaultdict(list)
    token_bag_lookup: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    person_core_lookup: Dict[str, List[str]] = defaultdict(list)
    label_to_idx = {
        safe_str(row["normalized_label"]): int(row["catalog_idx"])
        for _, row in catalog.iterrows()
    }
    for _, row in catalog.iterrows():
        normalized_label = safe_str(row["normalized_label"])
        source_types = type_set_from_text(row.get("source_types", ""))
        surface_key_lookup[variant_surface_key(normalized_label)].append(normalized_label)
        token_bag_lookup[token_bag_key(normalized_label)].append(normalized_label)
        person_key = person_core_key(normalized_label, source_types)
        if person_key:
            person_core_lookup[person_key].append(normalized_label)

    (
        _label_vectorizer,
        label_matrix,
        label_neighbors,
        _context_vectorizer,
        context_matrix,
        context_neighbors,
    ) = build_similarity_models(catalog, top_k_neighbors=top_k_neighbors)

    label_neighbor_map = neighbor_index_map(label_matrix, label_neighbors)
    context_neighbor_map = neighbor_index_map(context_matrix, context_neighbors)

    candidate_rows: List[Dict[str, object]] = []

    for _, alias_row in catalog.iterrows():
        alias_label = safe_str(alias_row["normalized_label"])
        if not alias_label or alias_label in alias_map:
            continue

        alias_idx = int(alias_row["catalog_idx"])
        deterministic = deterministic_candidates_for_label(
            alias_row,
            catalog_lookup,
            labels,
            surface_key_lookup,
            token_bag_lookup,
            person_core_lookup,
        )
        neighbor_indices = set(label_neighbor_map.get(alias_idx, [])) | set(context_neighbor_map.get(alias_idx, []))

        candidate_map: Dict[str, Dict[str, object]] = {}

        for candidate_label, rule_reason in deterministic:
            candidate_row = catalog_lookup.get(candidate_label)
            if candidate_row is None:
                continue
            candidate_map[candidate_label] = {
                "candidate_row": candidate_row,
                "rule_reason": rule_reason,
            }

        for candidate_idx in neighbor_indices:
            candidate_row = catalog.iloc[int(candidate_idx)]
            candidate_label = safe_str(candidate_row["normalized_label"])
            if candidate_label == alias_label:
                continue
            if not should_keep_candidate_direction(alias_row, candidate_row):
                continue
            existing = candidate_map.get(candidate_label)
            if existing is None:
                candidate_map[candidate_label] = {
                    "candidate_row": candidate_row,
                    "rule_reason": None,
                }

        scored_rows: List[Dict[str, object]] = []
        for candidate_label, payload in candidate_map.items():
            candidate_row = payload["candidate_row"]
            candidate_idx = label_to_idx[candidate_label]
            rule_reason = payload["rule_reason"]

            if should_skip_candidate_direction(alias_label, candidate_label):
                continue
            if not should_keep_candidate_direction(alias_row, candidate_row):
                continue

            metrics = score_candidate_pair(
                alias_row=alias_row,
                candidate_row=candidate_row,
                label_matrix=label_matrix,
                context_matrix=context_matrix,
                alias_idx=alias_idx,
                candidate_idx=candidate_idx,
                rule_reason=rule_reason,
            )

            final_score = float(metrics["final_score"])
            type_score = float(metrics["type_compatibility_score"])
            suggested_canonical = choose_canonical_label(alias_row, candidate_row)
            action = initial_recommended_action(
                final_score=final_score,
                rule_reason=rule_reason,
                type_score=type_score,
                alias_mentions=int(alias_row["mention_count"]),
                candidate_mentions=int(candidate_row["mention_count"]),
                lexical_cosine=float(metrics["lexical_cosine"]),
                sequence_ratio=float(metrics["sequence_ratio"]),
                token_jaccard=float(metrics["token_jaccard"]),
                hard_conflict_reason=safe_str(metrics["hard_conflict_reason"]),
                openai_min_score=openai_min_score,
                review_min_score=review_min_score,
            )
            if action == "reject_low_score":
                continue
            if action == "reject_structural_mismatch":
                continue

            scored_rows.append(
                {
                    "normalized_label": alias_label,
                    "suggested_canonical_label": suggested_canonical,
                    "matched_existing_label": safe_str(candidate_row["normalized_label"]),
                    "candidate_source": "deterministic" if rule_reason else "similarity",
                    "heuristic_reason": rule_reason or "lexical_context_similarity",
                    "mention_count": int(alias_row["mention_count"]),
                    "unique_docs": int(alias_row["unique_docs"]),
                    "unique_paragraphs": int(alias_row["unique_paragraphs"]),
                    "source_types": safe_str(alias_row["source_types"]),
                    "primary_type": safe_str(alias_row["primary_type"]),
                    "example_mentions": safe_str(alias_row["example_mentions"]),
                    "example_descriptions": safe_str(alias_row["example_descriptions"]),
                    "top_co_mentions": safe_str(alias_row["top_co_mentions"]),
                    "matched_label_mention_count": int(candidate_row["mention_count"]),
                    "matched_label_unique_docs": int(candidate_row["unique_docs"]),
                    "matched_label_source_types": safe_str(candidate_row["source_types"]),
                    "matched_label_example_mentions": safe_str(candidate_row["example_mentions"]),
                    "matched_label_example_descriptions": safe_str(candidate_row["example_descriptions"]),
                    **metrics,
                    "recommended_action": action,
                }
            )

        if not scored_rows:
            continue

        ranked = sorted(
            scored_rows,
            key=lambda row: (
                row["final_score"],
                row["matched_label_mention_count"],
                len(row["matched_existing_label"]),
            ),
            reverse=True,
        )[:max_candidates_per_label]

        for rank, candidate in enumerate(ranked, start=1):
            candidate["candidate_rank"] = rank
            candidate_rows.append(candidate)

    out = pd.DataFrame(candidate_rows)
    if len(out) == 0:
        return out

    out = out.sort_values(
        ["recommended_action", "mention_count", "final_score", "normalized_label", "candidate_rank"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    return out


def adjudication_key(row: pd.Series) -> str:
    raw = "||".join(
        [
            OPENAI_PROMPT_VERSION,
            safe_str(row.get("normalized_label", "")),
            safe_str(row.get("suggested_canonical_label", "")),
            safe_str(row.get("matched_existing_label", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def openai_prompt(row: pd.Series) -> str:
    return f"""
You are adjudicating whether two extracted entity labels from historical intelligence documents
should be merged into the same canonical entity.

Be conservative.
Merge only if they clearly refer to the same real-world entity or institution.
Do not merge entities that are merely related, in the same country, or part of the same event.
If you are uncertain, choose keep_separate.

Return only valid JSON:
{{
  "decision": "merge|keep_separate",
  "confidence": 0.0,
  "canonical_label": "best canonical label if merge, else empty",
  "reason": "short explanation"
}}

Alias candidate:
- normalized_label: {safe_str(row.get("normalized_label", ""))}
- mention_count: {safe_str(row.get("mention_count", ""))}
- unique_docs: {safe_str(row.get("unique_docs", ""))}
- source_types: {safe_str(row.get("source_types", ""))}
- example_mentions: {safe_str(row.get("example_mentions", ""))}
- example_descriptions: {safe_str(row.get("example_descriptions", ""))}
- top_co_mentions: {safe_str(row.get("top_co_mentions", ""))}

Candidate canonical target:
- matched_existing_label: {safe_str(row.get("matched_existing_label", ""))}
- suggested_canonical_label: {safe_str(row.get("suggested_canonical_label", ""))}
- mention_count: {safe_str(row.get("matched_label_mention_count", ""))}
- unique_docs: {safe_str(row.get("matched_label_unique_docs", ""))}
- source_types: {safe_str(row.get("matched_label_source_types", ""))}
- example_mentions: {safe_str(row.get("matched_label_example_mentions", ""))}
- example_descriptions: {safe_str(row.get("matched_label_example_descriptions", ""))}

Heuristic signals:
- heuristic_reason: {safe_str(row.get("heuristic_reason", ""))}
- lexical_cosine: {safe_str(row.get("lexical_cosine", ""))}
- context_cosine: {safe_str(row.get("context_cosine", ""))}
- sequence_ratio: {safe_str(row.get("sequence_ratio", ""))}
- token_jaccard: {safe_str(row.get("token_jaccard", ""))}
- co_mention_jaccard: {safe_str(row.get("co_mention_jaccard", ""))}
- type_compatibility_score: {safe_str(row.get("type_compatibility_score", ""))}
- type_compatibility_note: {safe_str(row.get("type_compatibility_note", ""))}
- final_score: {safe_str(row.get("final_score", ""))}
""".strip()


def call_openai_json(client: OpenAI, prompt: str, model: str, max_retries: int = 4) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "developer",
                        "content": "Return only valid JSON matching the requested schema.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            sleep_seconds = min(60, (2 ** attempt) + random.random())
            print(f"OpenAI adjudication error attempt {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}")
            print(f"Sleeping {sleep_seconds:.1f} seconds...")
            time.sleep(sleep_seconds)
    if last_error is None:
        raise RuntimeError("OpenAI adjudication failed without exception details.")
    raise last_error


def run_openai_adjudication(
    candidates: pd.DataFrame,
    out_dir: Path,
    model: str,
    max_candidates: Optional[int],
    min_mentions: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(candidates) == 0:
        return pd.DataFrame(
            columns=[
                "adjudication_key",
                "normalized_label",
                "matched_existing_label",
                "decision",
                "confidence",
                "canonical_label",
                "reason",
            ]
        )

    if OpenAI is None:
        raise ImportError("OpenAI package is not installed. Run: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    path = out_dir / "openai_adjudications.jsonl"
    error_path = out_dir / "openai_adjudication_errors.jsonl"
    existing_rows = load_jsonl(path)
    existing_map = {
        safe_str(row.get("adjudication_key", "")): row
        for row in existing_rows
        if safe_str(row.get("adjudication_key", ""))
    }

    queue = candidates.copy()
    queue = queue[
        queue["recommended_action"].eq("needs_openai_review")
        & (queue["mention_count"] >= min_mentions)
    ].copy()
    if len(queue) == 0:
        return pd.DataFrame(existing_rows)

    queue["adjudication_key"] = queue.apply(adjudication_key, axis=1)
    queue["openai_priority"] = np.where(queue["final_score"] >= 0.78, 0, 1)
    queue = queue.sort_values(
        ["openai_priority", "mention_count", "unique_docs", "final_score", "candidate_rank"],
        ascending=[True, False, False, False, True],
    )
    if max_candidates is not None:
        queue = queue.head(max_candidates)
    queue.drop(columns=["openai_priority"], inplace=True, errors="ignore")
    queue.to_csv(out_dir / "alias_openai_review_queue.csv", index=False)

    pending_queue = queue[~queue["adjudication_key"].isin(existing_map)].copy()
    print(
        "OpenAI adjudication queue:",
        {
            "total": int(len(queue)),
            "cached": int(len(queue) - len(pending_queue)),
            "pending": int(len(pending_queue)),
        },
    )
    if len(pending_queue) == 0:
        return pd.DataFrame(existing_map.values())

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        max_retries=2,
    )

    iterator = pending_queue.iterrows()
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(pending_queue),
            desc="OpenAI adjudications",
            unit="pair",
        )

    for _, row in iterator:
        key = safe_str(row["adjudication_key"])

        prompt = openai_prompt(row)
        try:
            response = call_openai_json(client, prompt=prompt, model=model)
        except Exception as exc:
            error_record = {
                "adjudication_key": key,
                "normalized_label": safe_str(row.get("normalized_label", "")),
                "matched_existing_label": safe_str(row.get("matched_existing_label", "")),
                "model": model,
                "error_type": type(exc).__name__,
                "error": safe_str(exc),
            }
            append_jsonl(error_path, error_record)
            print(f"Skipping OpenAI adjudication for {row.get('normalized_label', '')} -> {row.get('matched_existing_label', '')}: {type(exc).__name__}: {exc}")
            continue
        record = {
            "adjudication_key": key,
            "normalized_label": safe_str(row.get("normalized_label", "")),
            "matched_existing_label": safe_str(row.get("matched_existing_label", "")),
            "suggested_canonical_label": safe_str(row.get("suggested_canonical_label", "")),
            "model": model,
            "decision": safe_str(response.get("decision", "")),
            "confidence": float(response.get("confidence", 0.0) or 0.0),
            "canonical_label": safe_str(response.get("canonical_label", "")),
            "reason": safe_str(response.get("reason", "")),
        }
        append_jsonl(path, record)
        existing_map[key] = record

    return pd.DataFrame(existing_map.values())


def merge_openai_results(
    candidates: pd.DataFrame,
    adjudications: pd.DataFrame,
) -> pd.DataFrame:
    if len(candidates) == 0:
        return candidates

    out = candidates.copy()
    out["adjudication_key"] = out.apply(adjudication_key, axis=1)
    for col in ["openai_decision", "openai_confidence", "openai_reason", "openai_canonical_label"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    if len(adjudications) == 0:
        out["openai_decision"] = ""
        out["openai_confidence"] = np.nan
        out["openai_reason"] = ""
        out["openai_canonical_label"] = ""
        return out

    adjudications = adjudications.rename(
        columns={
            "decision": "openai_decision",
            "confidence": "openai_confidence",
            "reason": "openai_reason",
            "canonical_label": "openai_canonical_label",
        }
    )
    out = out.merge(
        adjudications[
            [
                "adjudication_key",
                "openai_decision",
                "openai_confidence",
                "openai_reason",
                "openai_canonical_label",
            ]
        ],
        on="adjudication_key",
        how="left",
    )

    out["openai_decision"] = out["openai_decision"].fillna("")
    out["openai_reason"] = out["openai_reason"].fillna("")
    out["openai_canonical_label"] = out["openai_canonical_label"].fillna("")

    def revised_action(row: pd.Series) -> str:
        current = safe_str(row.get("recommended_action", ""))
        decision = safe_str(row.get("openai_decision", ""))
        confidence = float(row.get("openai_confidence", 0.0) or 0.0)

        if current == "needs_openai_review":
            if decision == "merge" and confidence >= OPENAI_ACCEPT_CONFIDENCE_THRESHOLD:
                return "accept_openai"
            if decision == "keep_separate" and confidence >= OPENAI_ACCEPT_CONFIDENCE_THRESHOLD:
                return "reject_openai"
            if decision in {"merge", "keep_separate"}:
                return "reject_openai_low_confidence"
            return "pending_openai"
        return current

    out["recommended_action_final"] = out.apply(revised_action, axis=1)
    out["suggested_canonical_label_final"] = np.where(
        out["openai_decision"].eq("merge") & out["openai_canonical_label"].fillna("").astype(str).ne(""),
        out["openai_canonical_label"],
        out["suggested_canonical_label"],
    )
    return out


def build_openai_review_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    if len(candidates) == 0:
        return candidates.copy()

    queue = candidates[candidates["recommended_action"].eq("needs_openai_review")].copy()
    if len(queue) == 0:
        return queue

    queue["review_pair_key"] = queue.apply(
        lambda row: "||".join(
            sorted(
                [
                    safe_str(row.get("normalized_label", "")),
                    safe_str(row.get("matched_existing_label", "")),
                ]
            )
        ),
        axis=1,
    )
    queue = (
        queue.sort_values(
            ["mention_count", "unique_docs", "final_score", "candidate_rank", "normalized_label"],
            ascending=[False, False, False, True, True],
        )
        .drop_duplicates(subset=["review_pair_key", "candidate_rank"], keep="first")
        .drop(columns=["review_pair_key"], errors="ignore")
        .reset_index(drop=True)
    )
    return queue


def build_generated_alias_table(
    seed_alias_df: pd.DataFrame,
    accepted_aliases: pd.DataFrame,
) -> pd.DataFrame:
    seed = seed_alias_df.copy()
    if len(seed) == 0:
        seed = pd.DataFrame(columns=["alias", "canonical_label", "notes"])
    if "notes" not in seed.columns:
        seed["notes"] = ""
    seed["source"] = "seed"
    seed["review_status"] = "seed"
    seed["final_score"] = np.nan

    auto = accepted_aliases.copy()
    if len(auto) == 0:
        auto = pd.DataFrame(
            columns=[
                "alias",
                "canonical_label",
                "notes",
                "source",
                "review_status",
                "final_score",
            ]
        )
    else:
        auto = auto.rename(columns={"heuristic_reason": "notes"})
        auto["source"] = "stage_a1_auto"
        auto["notes"] = auto["notes"].fillna("")
        if "final_score" not in auto.columns:
            auto["final_score"] = np.nan
        auto = auto.loc[:, ["alias", "canonical_label", "notes", "source", "review_status", "final_score"]]

    combined = pd.concat(
        [
            seed.loc[:, ["alias", "canonical_label", "notes", "source", "review_status", "final_score"]],
            auto,
        ],
        ignore_index=True,
    )
    combined["alias_key"] = combined["alias"].map(normalize_label_basic)
    combined = combined.sort_values(
        ["source", "review_status", "final_score"],
        ascending=[True, True, False],
        na_position="last",
    )
    combined = combined.drop_duplicates(subset=["alias_key"], keep="first").drop(columns=["alias_key"])
    return combined.reset_index(drop=True)


def write_outputs(
    out_dir: Path,
    seed_alias_df: pd.DataFrame,
    label_catalog: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label_catalog.to_csv(out_dir / "label_catalog.csv", index=False)
    finalized_candidates = candidates.copy()
    review_candidates = build_openai_review_queue(finalized_candidates)
    finalized_candidates.to_csv(out_dir / "alias_candidates_all.csv", index=False)
    review_candidates.to_csv(out_dir / "alias_candidates_review.csv", index=False)
    review_candidates.to_csv(out_dir / "alias_openai_review_queue.csv", index=False)
    final_candidates = finalized_candidates.copy()
    final_candidates.to_csv(out_dir / "alias_candidates_final.csv", index=False)

    auto_accepted_candidates = finalized_candidates[
        finalized_candidates["recommended_action_final"].isin({"accept_rule", "accept_heuristic"})
    ].copy()
    auto_accepted_candidates.to_csv(out_dir / "alias_auto_accepted.csv", index=False)

    openai_accepted_candidates = finalized_candidates[
        finalized_candidates["recommended_action_final"].eq("accept_openai")
    ].copy()
    openai_accepted_candidates.to_csv(out_dir / "alias_openai_accepted.csv", index=False)

    openai_rejected_candidates = finalized_candidates[
        finalized_candidates["recommended_action_final"].isin({"reject_openai", "reject_openai_low_confidence"})
    ].copy()
    openai_rejected_candidates.to_csv(out_dir / "alias_openai_rejected.csv", index=False)

    openai_unresolved_candidates = finalized_candidates[
        finalized_candidates["recommended_action_final"].eq("pending_openai")
    ].copy()
    openai_unresolved_candidates.to_csv(out_dir / "alias_openai_unresolved.csv", index=False)

    empty_manual_review = finalized_candidates.iloc[0:0].copy()
    empty_manual_review.to_csv(out_dir / "alias_manual_review.csv", index=False)

    accepted = finalized_candidates[
        finalized_candidates["recommended_action_final"].isin({"accept_rule", "accept_heuristic", "accept_openai"})
    ].copy()
    if len(accepted):
        accepted = (
            accepted.sort_values(
                ["normalized_label", "openai_confidence", "final_score", "candidate_rank"],
                ascending=[True, False, False, True],
                na_position="last",
            )
            .drop_duplicates(subset=["normalized_label"], keep="first")
            .copy()
        )
        label_stats = {
            safe_str(row["normalized_label"]): {
                "mention_count": int(row.get("mention_count", 0)),
                "token_count": len(token_set(safe_str(row["normalized_label"]))),
            }
            for _, row in label_catalog.iterrows()
        }

        parent: Dict[str, str] = {}

        def find(value: str) -> str:
            parent.setdefault(value, value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: str, right: str) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        target_support: Counter = Counter()
        for _, row in accepted.iterrows():
            alias = safe_str(row["normalized_label"])
            target = safe_str(row["suggested_canonical_label_final"])
            if alias and target:
                union(alias, target)
                target_support[target] += 1

        groups: Dict[str, Set[str]] = defaultdict(set)
        for _, row in accepted.iterrows():
            alias = safe_str(row["normalized_label"])
            target = safe_str(row["suggested_canonical_label_final"])
            if alias:
                groups[find(alias)].add(alias)
            if target:
                groups[find(target)].add(target)

        resolved_canonical: Dict[str, str] = {}
        for members in groups.values():
            ranked = sorted(
                members,
                key=lambda label: (
                    target_support.get(label, 0),
                    label_stats.get(label, {}).get("mention_count", 0),
                    0 if looks_like_acronym(label) else 1,
                    label_stats.get(label, {}).get("token_count", 0),
                    len(label),
                ),
                reverse=True,
            )
            winner = ranked[0]
            for member in members:
                resolved_canonical[member] = winner

        accepted["canonical_label_resolved"] = accepted["suggested_canonical_label_final"].map(
            lambda label: resolved_canonical.get(safe_str(label), safe_str(label))
        )
        accepted_aliases = (
            accepted.sort_values(
                ["normalized_label", "final_score", "candidate_rank"],
                ascending=[True, False, True],
            )
            .drop_duplicates(subset=["normalized_label"])
            .loc[:, ["normalized_label", "canonical_label_resolved", "recommended_action_final", "heuristic_reason", "final_score", "openai_decision", "openai_confidence"]]
            .rename(
                columns={
                    "normalized_label": "alias",
                    "canonical_label_resolved": "canonical_label",
                    "recommended_action_final": "review_status",
                }
            )
        )
        accepted_aliases = accepted_aliases[accepted_aliases["alias"].astype(str) != accepted_aliases["canonical_label"].astype(str)].copy()
    else:
        accepted_aliases = pd.DataFrame(
            columns=[
                "alias",
                "canonical_label",
                "review_status",
                "heuristic_reason",
                "final_score",
                "openai_decision",
                "openai_confidence",
            ]
        )
    accepted_aliases.to_csv(out_dir / "entity_aliases_proposed.csv", index=False)
    generated_aliases = build_generated_alias_table(seed_alias_df, accepted_aliases)
    generated_aliases.to_csv(out_dir / GENERATED_ALIAS_FILENAME, index=False)

    summary = {
        "label_catalog_rows": int(len(label_catalog)),
        "candidate_rows": int(len(review_candidates)),
        "candidate_rows_all": int(len(finalized_candidates)),
        "candidate_rows_final": int(len(final_candidates)),
        "proposed_alias_rows": int(len(accepted_aliases)),
        "generated_alias_rows": int(len(generated_aliases)),
        "auto_accepted_rows": int(len(auto_accepted_candidates)),
        "openai_review_queue_rows": int(len(review_candidates)),
        "openai_accepted_rows": int(len(openai_accepted_candidates)),
        "openai_rejected_rows": int(len(openai_rejected_candidates)),
        "openai_unresolved_rows": int(len(openai_unresolved_candidates)),
        "manual_review_rows": 0,
        "actions": (
            finalized_candidates["recommended_action_final"].value_counts(dropna=False).to_dict()
            if "recommended_action_final" in finalized_candidates.columns
            else {}
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage A1 alias-resolution worklists from extraction outputs.")
    parser.set_defaults(run_openai=True)
    parser.add_argument("PROJECT_DIR", help="Project directory containing extraction/ and optional entity_directory/")
    parser.add_argument("--entity-dir", default=None, help="Optional explicit alias-directory path. If omitted and no project entity_directory exists, A1 starts with an empty alias seed.")
    parser.add_argument("--top-k-neighbors", type=int, default=10, help="Nearest-neighbor pool size for lexical/context search.")
    parser.add_argument("--max-candidates-per-label", type=int, default=3, help="Maximum candidate suggestions to keep per unresolved label before OpenAI adjudication.")
    parser.add_argument("--openai-min-score", type=float, default=0.78, help="Higher-priority heuristic score threshold for OpenAI adjudication ordering.")
    parser.add_argument("--review-min-score", type=float, default=0.60, help="Minimum heuristic score required to keep a candidate for OpenAI adjudication.")
    parser.add_argument("--run-openai", dest="run_openai", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-openai", dest="run_openai", action="store_false", help="Skip OpenAI adjudication and use heuristics only.")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for adjudication when --run-openai is enabled.")
    parser.add_argument("--max-openai-candidates", type=int, default=None, help="Optional cap on candidates to adjudicate with OpenAI in a single run. Omit to adjudicate all surviving candidates.")
    parser.add_argument("--openai-min-mentions", type=int, default=1, help="Minimum alias mention count before a candidate is eligible for OpenAI adjudication.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.PROJECT_DIR).expanduser().resolve()
    entity_dir = resolve_entity_directory(project_dir, Path(args.entity_dir).expanduser().resolve() if args.entity_dir else None)
    out_dir = project_dir / DEFAULT_OUTPUT_DIRNAME

    alias_df = load_entity_alias_table(entity_dir)
    alias_map = build_entity_alias_map(alias_df)
    print("Loaded alias seed rows:", len(alias_df))

    entities = load_entities(project_dir)
    label_catalog = build_label_catalog(entities, alias_map)
    print("Label catalog rows:", len(label_catalog))

    candidates = build_candidate_worklist(
        catalog=label_catalog,
        alias_map=alias_map,
        top_k_neighbors=args.top_k_neighbors,
        max_candidates_per_label=args.max_candidates_per_label,
        openai_min_score=args.openai_min_score,
        review_min_score=args.review_min_score,
    )
    if len(candidates) == 0:
        candidates["recommended_action_final"] = []
        write_outputs(
            out_dir,
            alias_df,
            label_catalog,
            candidates,
        )
        return

    candidates["recommended_action_final"] = candidates["recommended_action"]
    candidates["suggested_canonical_label_final"] = candidates["suggested_canonical_label"]
    candidates["openai_decision"] = ""
    candidates["openai_confidence"] = np.nan
    candidates["openai_reason"] = ""

    if args.run_openai:
        adjudications = run_openai_adjudication(
            candidates=candidates,
            out_dir=out_dir,
            model=args.openai_model,
            max_candidates=args.max_openai_candidates,
            min_mentions=args.openai_min_mentions,
        )
        candidates = merge_openai_results(candidates, adjudications)

    write_outputs(
        out_dir,
        alias_df,
        label_catalog,
        candidates,
    )


if __name__ == "__main__":
    main()
