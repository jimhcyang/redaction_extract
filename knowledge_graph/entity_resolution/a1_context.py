"""
Stage A1: build mention-level and label-level context catalogs from Stage A outputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .common import (
    ENTITY_INVENTORY_FILENAME,
    MENTION_CATALOG_FILENAME,
    build_entity_alias_map,
    canonicalize_entity_label,
    entity_resolution_dir,
    first_existing_col,
    infer_structural_family,
    iter_progress,
    label_ambiguity_score,
    load_entities,
    load_seed_alias_table,
    looks_like_acronym,
    mean_pairwise_jaccard,
    norm_space,
    normalize_label_basic,
    resolve_entity_directory,
    safe_str,
    token_set,
    type_set_from_text,
    _sample_text_list,
)


def build_entity_inventory(label_catalog: pd.DataFrame) -> pd.DataFrame:
    if len(label_catalog) == 0:
        return pd.DataFrame(
            columns=[
                "normalized_label",
                "canonical_seed_label",
                "mention_count",
                "doc_count",
                "entity_type",
                "surface_forms",
                "sample_context",
                "is_generic",
                "acronym",
            ]
        )

    inventory = label_catalog.copy()
    inventory["is_generic"] = inventory.apply(
        lambda row: infer_structural_family(
            safe_str(row.get("normalized_label", "")),
            type_set_from_text(row.get("source_types", "")),
        )
        in {"government", "location", "organization", "role"},
        axis=1,
    )
    inventory["acronym"] = inventory["normalized_label"].map(looks_like_acronym)
    inventory = inventory.rename(
        columns={
            "canonical_label_current": "canonical_seed_label",
            "unique_docs": "doc_count",
            "primary_type": "entity_type",
            "example_mentions": "surface_forms",
            "context_text": "sample_context",
        }
    )
    keep_cols = [
        "normalized_label",
        "canonical_seed_label",
        "mention_count",
        "doc_count",
        "entity_type",
        "surface_forms",
        "sample_context",
        "is_generic",
        "acronym",
    ]
    return inventory.loc[:, keep_cols].reset_index(drop=True)


def load_stage_a_context_tables(project_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extraction_dir = project_dir / "extraction"
    entities = load_entities(project_dir)

    def maybe_read_csv(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    events = maybe_read_csv(extraction_dir / "events.csv")
    claims = maybe_read_csv(extraction_dir / "claims.csv")
    input_docs = maybe_read_csv(project_dir / "input_documents.csv")
    return entities, events, claims, input_docs


def build_mention_catalog(
    entities: pd.DataFrame,
    events: pd.DataFrame,
    claims: pd.DataFrame,
    input_docs: pd.DataFrame,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    type_col = first_existing_col(entities, ["type", "entity_type", "broad_type"])
    desc_col = first_existing_col(entities, ["description", "summary", "evidence"])
    if label_col is None:
        raise KeyError(f"Could not find entity label column in entities. Columns: {entities.columns.tolist()}")

    input_subject_map: Dict[str, str] = {}
    input_date_map: Dict[str, str] = {}
    if len(input_docs) and "doc_id" in input_docs.columns:
        subject_col = first_existing_col(input_docs, ["subject", "document_title", "title"])
        date_col = first_existing_col(input_docs, ["date", "date_display"])
        if subject_col is not None:
            input_subject_map = dict(zip(input_docs["doc_id"].astype(str), input_docs[subject_col].map(norm_space)))
        if date_col is not None:
            input_date_map = dict(zip(input_docs["doc_id"].astype(str), input_docs[date_col].map(norm_space)))

    event_label_col = first_existing_col(events, ["label", "event_label", "event_text_for_cluster"])
    event_desc_col = first_existing_col(events, ["description", "summary"])
    claim_text_col = first_existing_col(claims, ["claim_text", "label", "text"])

    events_by_paragraph: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    if len(events) and event_label_col is not None:
        for _, row in iter_progress(events.iterrows(), total=len(events), desc="A1 event context"):
            key = (safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
            label = norm_space(row.get(event_label_col, ""))
            desc = norm_space(row.get(event_desc_col, "")) if event_desc_col else ""
            text = " - ".join(part for part in [label, desc] if part)
            if text:
                events_by_paragraph[key].append(text)

    claims_by_paragraph: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    if len(claims) and claim_text_col is not None:
        for _, row in iter_progress(claims.iterrows(), total=len(claims), desc="A1 claim context"):
            key = (safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
            text = norm_space(row.get(claim_text_col, ""))
            if text:
                claims_by_paragraph[key].append(text)

    entity_labels_by_paragraph: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for _, row in iter_progress(entities.iterrows(), total=len(entities), desc="A1 entity context"):
        key = (safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))
        raw_label = norm_space(row.get(label_col, ""))
        if raw_label:
            entity_labels_by_paragraph[key].append(raw_label)

    rows: List[Dict[str, object]] = []
    for idx, row in iter_progress(entities.iterrows(), total=len(entities), desc="A1 mention catalog"):
        raw_label = norm_space(row.get(label_col, ""))
        normalized_label = normalize_label_basic(raw_label)
        if not normalized_label:
            continue

        doc_id = safe_str(row.get("doc_id", ""))
        paragraph_id = safe_str(row.get("paragraph_id", ""))
        local_id = safe_str(row.get("local_id", row.get("id", f"e{idx}")))
        source_type = norm_space(row.get(type_col, "")) if type_col else ""
        description = norm_space(row.get(desc_col, "")) if desc_col else ""
        paragraph_key = (doc_id, paragraph_id)

        sibling_entities = [
            value
            for value in _sample_text_list(entity_labels_by_paragraph.get(paragraph_key, []), limit=12)
            if normalize_label_basic(value) != normalized_label
        ]
        sibling_events = _sample_text_list(events_by_paragraph.get(paragraph_key, []), limit=6)
        sibling_claims = _sample_text_list(claims_by_paragraph.get(paragraph_key, []), limit=4)
        subject = input_subject_map.get(doc_id, "")
        date = input_date_map.get(doc_id, "")

        context_parts = [
            subject,
            date,
            source_type,
            description,
            "entities: " + "; ".join(sibling_entities[:8]) if sibling_entities else "",
            "events: " + "; ".join(sibling_events[:4]) if sibling_events else "",
            "claims: " + "; ".join(sibling_claims[:2]) if sibling_claims else "",
        ]
        context_text = " || ".join(part for part in context_parts if part)

        rows.append(
            {
                "mention_key": f"{doc_id}::{paragraph_id}::{local_id}",
                "doc_id": doc_id,
                "paragraph_id": paragraph_id,
                "local_id": local_id,
                "raw_label": raw_label,
                "normalized_label": normalized_label,
                "canonical_label_current": canonicalize_entity_label(normalized_label, alias_map),
                "source_type": source_type,
                "description": description,
                "doc_subject": subject,
                "doc_date": date,
                "sibling_entities": "; ".join(sibling_entities),
                "sibling_events": "; ".join(sibling_events),
                "sibling_claims": "; ".join(sibling_claims),
                "context_text": context_text,
                "context_token_count": len(token_set(context_text)),
                "family": infer_structural_family(normalized_label, {normalize_label_basic(source_type)} if source_type else set()),
                "looks_like_acronym": looks_like_acronym(normalized_label),
            }
        )

    return pd.DataFrame(rows)


def build_label_catalog_from_mentions(
    mention_catalog: pd.DataFrame,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    if len(mention_catalog) == 0:
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
                "example_contexts",
                "top_co_mentions",
                "context_text",
                "context_coherence",
                "label_ambiguity",
                "in_alias_dictionary",
            ]
        )

    paragraph_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for _, row in iter_progress(mention_catalog.iterrows(), total=len(mention_catalog), desc="A1 paragraph groups"):
        paragraph_groups[(safe_str(row.get("doc_id", "")), safe_str(row.get("paragraph_id", "")))].append(
            safe_str(row.get("normalized_label", ""))
        )

    co_mentions: Dict[str, Counter] = defaultdict(Counter)
    for labels in paragraph_groups.values():
        unique_labels = sorted(set(label for label in labels if label))
        for left_idx, left in enumerate(unique_labels):
            for right_idx, right in enumerate(unique_labels):
                if left_idx == right_idx:
                    continue
                co_mentions[left][right] += 1

    grouped_rows: List[Dict[str, object]] = []
    grouped_mentions = mention_catalog.groupby("normalized_label")
    for normalized_label, group in iter_progress(grouped_mentions, total=mention_catalog["normalized_label"].nunique(), desc="A1 label catalog"):
        example_mentions = _sample_text_list(group["raw_label"].tolist(), limit=10)
        example_descriptions = _sample_text_list(group["description"].tolist(), limit=8)
        example_contexts = _sample_text_list(group["context_text"].tolist(), limit=3)
        type_counter = Counter(value for value in group["source_type"].astype(str) if value)
        source_types = sorted(type_counter.keys())
        top_co_mentions = [
            label
            for label, _count in co_mentions.get(normalized_label, Counter()).most_common(12)
        ]
        context_token_sets = [token_set(text) for text in example_contexts if token_set(text)]
        context_coherence = mean_pairwise_jaccard(context_token_sets[:12])
        ambiguity = label_ambiguity_score(
            normalized_label=normalized_label,
            source_types=source_types,
            context_token_sets=context_token_sets,
            top_co_mentions=top_co_mentions,
        )
        context_bits = [
            normalized_label,
            " ".join(source_types[:8]),
            " ".join(top_co_mentions[:10]),
            " ".join(example_descriptions[:6]),
            " ".join(example_contexts[:2]),
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
                "example_contexts": " || ".join(example_contexts),
                "top_co_mentions": "; ".join(top_co_mentions),
                "context_text": " || ".join(bit for bit in context_bits if bit).strip(),
                "context_coherence": round(context_coherence, 4),
                "label_ambiguity": round(ambiguity, 4),
                "in_alias_dictionary": normalized_label in alias_map,
            }
        )

    return pd.DataFrame(grouped_rows).sort_values(
        ["mention_count", "unique_docs", "normalized_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def run_stage_a1_context_catalog(
    project_dir: Path,
    entity_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    project_dir = Path(project_dir).expanduser().resolve()
    entity_dir = resolve_entity_directory(project_dir, entity_dir)
    out_dir = entity_resolution_dir(project_dir)

    alias_df = load_seed_alias_table(project_dir, entity_dir)
    alias_map = build_entity_alias_map(alias_df)
    print("Loaded alias seed rows:", len(alias_df))
    entities, events, claims, input_docs = load_stage_a_context_tables(project_dir)

    mention_catalog = build_mention_catalog(entities, events, claims, input_docs, alias_map)
    label_catalog = build_label_catalog_from_mentions(mention_catalog, alias_map)
    entity_inventory = build_entity_inventory(label_catalog)

    out_dir.mkdir(parents=True, exist_ok=True)
    mention_catalog.to_csv(out_dir / MENTION_CATALOG_FILENAME, index=False)
    label_catalog.to_csv(out_dir / "label_catalog.csv", index=False)
    entity_inventory.to_csv(out_dir / ENTITY_INVENTORY_FILENAME, index=False)

    print("A1 mention catalog rows:", len(mention_catalog))
    print("A1 label catalog rows:", len(label_catalog))
    return {
        "alias_df": alias_df,
        "mention_catalog": mention_catalog,
        "label_catalog": label_catalog,
        "entity_inventory": entity_inventory,
    }
