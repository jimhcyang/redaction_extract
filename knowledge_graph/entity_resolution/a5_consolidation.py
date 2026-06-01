"""
Stage A5: final canonical-entity consolidation after A4 mention-level resolution.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import a4_llm_resolution as a4
from . import common as core
from . import scoring


A5_CANONICAL_CATALOG_FILENAME = "a5_canonical_catalog.csv"
A5_CANDIDATE_PAIRS_FILENAME = "a5_candidate_pairs.csv"
A5_AUTO_ACCEPTED_FILENAME = "a5_auto_accepted.csv"
A5_REVIEW_QUEUE_FILENAME = "a5_review_queue.csv"
A5_ADJUDICATIONS_FILENAME = "a5_adjudications.jsonl"
A5_ADJUDICATION_ERRORS_FILENAME = "a5_adjudication_errors.jsonl"
A5_PAIR_DECISIONS_FILENAME = "a5_pair_decisions.csv"
A5_CONSOLIDATION_MAP_FILENAME = "a5_consolidation_map.csv"
A5_ENTITIES_RESOLVED_FILENAME = "a5_entities_resolved.csv"
A5_CANONICAL_MAP_FILENAME = "a5_entity_canonical_map.csv"
A5_GENERATED_ALIASES_FILENAME = "a5_entity_aliases_generated.csv"
A5_LLM_FILENAME = "a5_llm_adjudicated.csv"
A5_PREVIOUS_ENTITIES_SNAPSHOT = "a4_entities_resolved_snapshot.csv"
A5_PREVIOUS_CANONICAL_MAP_SNAPSHOT = "a4_entity_canonical_map_snapshot.csv"


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _strip_sense_suffix(label: str) -> str:
    label = core.safe_str(label)
    label = core.re.sub(r"\s+\[SENSE\s+\d+\]\s*$", "", label)
    return core.norm_space(label)


def _strip_trailing_parenthetical(label: str) -> str:
    label = core.safe_str(label)
    return core.norm_space(core.re.sub(r"\s*\([^)]*\)\s*$", "", label))


def _strip_leading_article(label: str) -> str:
    tokens = core.token_list(label)
    while tokens and tokens[0] in {"A", "AN", "THE"}:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def _flatten_semicolon_values(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value)
        for bit in text.split(";"):
            bit = core.norm_space(bit)
            if bit:
                out.append(bit)
    return out


def _method_set(value: object) -> Set[str]:
    return {core.safe_str(bit) for bit in core.safe_str(value).split(";") if core.safe_str(bit)}


def _placeholder_like(label: str, methods: Sequence[str]) -> bool:
    if "[SENSE " in core.safe_str(label):
        return True
    return any(method in {"a4_keep_separate", "a3_single_cluster_self"} for method in methods)


def _looks_acronymish_canonical(label: str) -> bool:
    base = _strip_trailing_parenthetical(_strip_sense_suffix(label))
    return core.looks_like_acronym(base)


def _canonical_preference_tuple(row: pd.Series, suggested_label: Optional[str] = None) -> Tuple[int, int, int, int, int, int, str]:
    label = core.norm_space(suggested_label or row.get("canonical_label_current", ""))
    methods = sorted(_method_set(row.get("current_methods", "")))
    placeholder = _placeholder_like(label, methods)
    acronym_like = _looks_acronymish_canonical(label)
    token_count = len(core.token_set(label))
    return (
        0 if placeholder else 1,
        0 if acronym_like else 1,
        int(row.get("source_label_count", 0)),
        int(row.get("mention_count", 0)),
        int(row.get("unique_docs", 0)),
        token_count,
        label,
    )


def _choose_best_member_label(member_rows: pd.DataFrame, preferred_labels: Optional[Sequence[str]] = None) -> str:
    preferred_set = {core.norm_space(value) for value in (preferred_labels or []) if core.norm_space(value)}
    ranked: List[Tuple[Tuple[int, int, int, int, int, int, str], str]] = []
    matched_preference = False
    for _, row in member_rows.iterrows():
        label = core.norm_space(row.get("canonical_label_current", ""))
        if not label:
            continue
        bonus = 1 if label in preferred_set else 0
        matched_preference = matched_preference or bool(bonus)
        ranked.append(((bonus,) + _canonical_preference_tuple(row, label), label))
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        if preferred_set and not matched_preference:
            return sorted(
                preferred_set,
                key=lambda value: (
                    0 if _looks_acronymish_canonical(value) else 1,
                    len(core.token_set(value)),
                    len(value),
                    value,
                ),
                reverse=True,
            )[0]
        return ranked[0][1]
    if preferred_set:
        return sorted(preferred_set, key=len, reverse=True)[0]
    return core.norm_space(member_rows.iloc[0].get("canonical_label_current", "")) if len(member_rows) else ""


def load_a5_inputs(project_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = core.entity_resolution_dir(project_dir)
    entities_resolved = pd.read_csv(out_dir / core.ENTITIES_RESOLVED_FILENAME, low_memory=False)
    mention_catalog = pd.read_csv(out_dir / core.MENTION_CATALOG_FILENAME, low_memory=False)
    canonical_map = pd.read_csv(out_dir / core.ENTITY_CANONICAL_MAP_FILENAME, low_memory=False)
    return entities_resolved, mention_catalog, canonical_map


def build_a5_canonical_catalog(
    entities_resolved: pd.DataFrame,
    mention_catalog: pd.DataFrame,
) -> pd.DataFrame:
    if len(entities_resolved) == 0:
        return pd.DataFrame()

    label_col = core.first_existing_col(entities_resolved, ["label", "name", "entity", "text"])
    type_col = core.first_existing_col(entities_resolved, ["type", "entity_type", "broad_type"])
    desc_col = core.first_existing_col(entities_resolved, ["description", "summary", "evidence"])

    mention_view_cols = [
        col
        for col in ["mention_key", "context_text", "sibling_entities", "doc_subject", "doc_date"]
        if col in mention_catalog.columns
    ]
    merged = entities_resolved.copy()
    if mention_view_cols:
        merged = merged.merge(
            mention_catalog.loc[:, mention_view_cols].drop_duplicates(subset=["mention_key"]),
            on="mention_key",
            how="left",
            suffixes=("", "_a1"),
        )

    context_col = "context_text" if "context_text" in merged.columns else None
    sibling_col = "sibling_entities" if "sibling_entities" in merged.columns else None
    if "context_text_a1" in merged.columns:
        context_col = "context_text_a1"
    if "sibling_entities_a1" in merged.columns:
        sibling_col = "sibling_entities_a1"

    rows: List[Dict[str, object]] = []
    grouped = merged.groupby(["canonical_key", "canonical_label"], dropna=False)
    for (canonical_key, canonical_label), group in core.iter_progress(
        grouped,
        total=merged[["canonical_key", "canonical_label"]].drop_duplicates().shape[0],
        desc="A5 canonical catalog",
    ):
        canonical_key = core.safe_str(canonical_key)
        canonical_label = core.norm_space(canonical_label)
        if not canonical_key or not canonical_label:
            continue

        source_types = sorted({core.norm_space(value) for value in group[type_col].astype(str)}) if type_col else []
        source_types = [value for value in source_types if value]
        raw_mentions = core._sample_text_list(group[label_col].astype(str).tolist(), limit=10) if label_col else []
        source_labels = core._sample_text_list(group["normalized_label"].astype(str).tolist(), limit=12)
        descriptions = core._sample_text_list(group[desc_col].astype(str).tolist(), limit=6) if desc_col else []
        example_contexts = (
            core._sample_text_list(group[context_col].astype(str).tolist(), limit=4)
            if context_col is not None
            else []
        )
        top_co_mentions = core._sample_text_list(
            _flatten_semicolon_values(group[sibling_col].tolist()) if sibling_col is not None else [],
            limit=12,
        )
        context_token_sets = [core.token_set(text) for text in example_contexts if core.safe_str(text)]
        current_methods = sorted({core.safe_str(value) for value in group["canonical_method"].astype(str) if core.safe_str(value)})
        placeholder_like = _placeholder_like(canonical_label, current_methods)
        looks_like_acronym = _looks_acronymish_canonical(canonical_label)
        needs_attention = placeholder_like or looks_like_acronym or ("a4_new_inferred_name" in current_methods)
        label_ambiguity = core.label_ambiguity_score(
            core.normalize_label_basic(canonical_label),
            source_types,
            context_token_sets,
            top_co_mentions,
        )
        context_text = " || ".join(
            part
            for part in [
                canonical_label,
                " ".join(source_types[:6]),
                " ".join(descriptions[:4]),
                " ".join(example_contexts[:2]),
                " ".join(top_co_mentions[:10]),
            ]
            if part
        )

        rows.append(
            {
                "canonical_key_current": canonical_key,
                "canonical_label_current": canonical_label,
                "normalized_label": canonical_label,
                "mention_count": int(len(group)),
                "unique_docs": int(group["doc_id"].astype(str).nunique()),
                "unique_paragraphs": int(group[["doc_id", "paragraph_id"]].astype(str).drop_duplicates().shape[0]),
                "source_types": "; ".join(source_types[:8]),
                "primary_type": source_types[0] if source_types else "",
                "example_mentions": "; ".join(raw_mentions),
                "source_normalized_labels": "; ".join(sorted({core.safe_str(value) for value in group["normalized_label"].astype(str) if core.safe_str(value)})[:12]),
                "source_label_count": int(group["normalized_label"].astype(str).nunique()),
                "current_methods": "; ".join(current_methods),
                "example_descriptions": "; ".join(descriptions),
                "example_contexts": " || ".join(example_contexts),
                "top_co_mentions": "; ".join(top_co_mentions),
                "context_text": context_text,
                "looks_like_acronym": looks_like_acronym,
                "placeholder_like": placeholder_like,
                "needs_consolidation_attention": needs_attention,
                "label_ambiguity": round(float(label_ambiguity), 4),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["needs_consolidation_attention", "mention_count", "unique_docs", "canonical_label_current"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    return out


def _a5_rule_reason(alias_row: pd.Series, candidate_row: pd.Series) -> str:
    alias_label = core.safe_str(alias_row.get("canonical_label_current", alias_row.get("normalized_label", "")))
    candidate_label = core.safe_str(candidate_row.get("canonical_label_current", candidate_row.get("normalized_label", "")))
    alias_types = core.type_set_from_text(alias_row.get("source_types", ""))
    candidate_types = core.type_set_from_text(candidate_row.get("source_types", ""))
    if core.structural_conflict_reason(alias_label, candidate_label, alias_types, candidate_types):
        return ""

    alias_base = _strip_sense_suffix(alias_label)
    candidate_base = _strip_sense_suffix(candidate_label)
    alias_no_paren = _strip_trailing_parenthetical(alias_base)
    candidate_no_paren = _strip_trailing_parenthetical(candidate_base)

    if alias_no_paren and candidate_no_paren and alias_no_paren == candidate_no_paren and alias_label != candidate_label:
        if (
            alias_base != alias_no_paren
            and candidate_base != candidate_no_paren
            and core.looks_like_acronym(alias_no_paren)
        ):
            return ""
        if _strip_leading_article(alias_base) == _strip_leading_article(candidate_base) and (
            alias_base != _strip_leading_article(alias_base) or candidate_base != _strip_leading_article(candidate_base)
        ):
            return "leading_article"
        if alias_base != alias_no_paren or candidate_base != candidate_no_paren:
            return "parenthetical_base_match"

    alias_article = _strip_leading_article(alias_base)
    candidate_article = _strip_leading_article(candidate_base)
    if alias_article and candidate_article and alias_article == candidate_article and alias_label != candidate_label:
        return "leading_article"

    alias_core = alias_no_paren or alias_base
    candidate_core = candidate_no_paren or candidate_base
    if alias_core and candidate_core:
        if core.acronym_from_label(alias_core) == candidate_core or core.acronym_from_label(candidate_core) == alias_core:
            return "acronym_match"

    alias_person_core = core.person_core_key(alias_base, alias_types)
    candidate_person_core = core.person_core_key(candidate_base, candidate_types)
    if alias_person_core and candidate_person_core and alias_person_core == candidate_person_core:
        return "person_title_variant"
    return ""


def _recommend_a5_action(
    score: Dict[str, object],
    rule_reason: str,
    alias_row: pd.Series,
    candidate_row: pd.Series,
    review_min_score: float,
    attention_review_min_score: float,
) -> str:
    hard_conflict = core.safe_str(score.get("hard_conflict_reason", ""))
    if hard_conflict:
        return "reject_structural_mismatch"

    final_score = float(score.get("final_score", 0.0) or 0.0)
    context_cosine = float(score.get("context_cosine", 0.0) or 0.0)
    weighted_jaccard = float(score.get("weighted_token_jaccard", 0.0) or 0.0)
    type_score = float(score.get("type_compatibility_score", 0.0) or 0.0)
    token_containment = max(
        float(score.get("alias_token_containment", 0.0) or 0.0),
        float(score.get("candidate_token_containment", 0.0) or 0.0),
    )
    row_attention = _safe_bool(alias_row.get("needs_consolidation_attention", False))
    candidate_attention = _safe_bool(candidate_row.get("needs_consolidation_attention", False))

    if rule_reason in {"leading_article", "parenthetical_base_match", "person_title_variant"} and type_score >= 0.45:
        return "accept_local"
    if final_score >= 0.93 and weighted_jaccard >= 0.88 and context_cosine >= 0.60 and type_score >= 0.82:
        return "accept_local"
    if final_score >= 0.88 and token_containment >= 0.90 and context_cosine >= 0.56 and type_score >= 0.82 and (row_attention or candidate_attention):
        return "accept_local"
    if final_score >= review_min_score:
        return "needs_openai_review"
    if (row_attention or candidate_attention or rule_reason) and final_score >= attention_review_min_score:
        return "needs_openai_review"
    return "reject_low_score"


def build_a5_candidate_pairs(
    canonical_catalog: pd.DataFrame,
    top_k_neighbors: int,
    review_min_score: float,
    attention_review_min_score: float,
) -> pd.DataFrame:
    if len(canonical_catalog) == 0:
        return pd.DataFrame()

    token_idf = core.build_token_idf_map(canonical_catalog)
    (
        label_vectorizer,
        label_matrix,
        label_neighbors,
        context_vectorizer,
        context_matrix,
        context_neighbors,
    ) = scoring.build_similarity_models(canonical_catalog, top_k_neighbors=max(4, top_k_neighbors))
    label_neighbor_map = scoring.neighbor_index_map(label_matrix, label_neighbors)
    context_neighbor_map = scoring.neighbor_index_map(context_matrix, context_neighbors)

    acronym_expansion_map: Dict[str, Set[int]] = defaultdict(set)
    for idx, row in canonical_catalog.iterrows():
        label = _strip_trailing_parenthetical(_strip_sense_suffix(core.safe_str(row["canonical_label_current"])))
        acronym = core.acronym_from_label(label)
        if acronym:
            acronym_expansion_map[acronym].add(int(idx))

    rows: List[Dict[str, object]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    iterator = canonical_catalog.iterrows()
    if core.tqdm is not None:
        iterator = core.tqdm(iterator, total=len(canonical_catalog), desc="A5 candidate pairs", unit="entity")

    for idx, row in iterator:
        row_key = core.safe_str(row["canonical_key_current"])
        neighbor_indices = set(label_neighbor_map.get(int(idx), [])) | set(context_neighbor_map.get(int(idx), []))
        row_label_base = _strip_trailing_parenthetical(_strip_sense_suffix(core.safe_str(row["canonical_label_current"])))
        if core.looks_like_acronym(row_label_base):
            neighbor_indices |= acronym_expansion_map.get(row_label_base, set())

        for candidate_idx in sorted(neighbor_indices):
            if int(candidate_idx) <= int(idx):
                continue
            candidate_row = canonical_catalog.iloc[int(candidate_idx)]
            candidate_key = core.safe_str(candidate_row["canonical_key_current"])
            if not row_key or not candidate_key or row_key == candidate_key:
                continue
            pair_key = tuple(sorted((row_key, candidate_key)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            row_attention = _safe_bool(row.get("needs_consolidation_attention", False))
            candidate_attention = _safe_bool(candidate_row.get("needs_consolidation_attention", False))
            rule_reason = _a5_rule_reason(row, candidate_row)
            score_rule_reason = rule_reason if rule_reason != "acronym_match" else None
            score = scoring.score_candidate_pair(
                row,
                candidate_row,
                label_matrix,
                context_matrix,
                int(idx),
                int(candidate_idx),
                rule_reason=score_rule_reason,
                token_idf=token_idf,
            )
            if not (
                rule_reason
                or float(score["final_score"]) >= review_min_score
                or ((row_attention or candidate_attention) and float(score["final_score"]) >= attention_review_min_score)
            ):
                continue

            recommended_action = _recommend_a5_action(
                score=score,
                rule_reason=rule_reason,
                alias_row=row,
                candidate_row=candidate_row,
                review_min_score=review_min_score,
                attention_review_min_score=attention_review_min_score,
            )

            rows.append(
                {
                    "left_canonical_key": row_key,
                    "left_canonical_label": core.safe_str(row["canonical_label_current"]),
                    "left_source_types": core.safe_str(row.get("source_types", "")),
                    "left_methods": core.safe_str(row.get("current_methods", "")),
                    "left_example_mentions": core.safe_str(row.get("example_mentions", "")),
                    "left_source_labels": core.safe_str(row.get("source_normalized_labels", "")),
                    "left_example_contexts": core.safe_str(row.get("example_contexts", "")),
                    "left_top_co_mentions": core.safe_str(row.get("top_co_mentions", "")),
                    "left_mention_count": int(row.get("mention_count", 0)),
                    "left_unique_docs": int(row.get("unique_docs", 0)),
                    "left_needs_attention": row_attention,
                    "right_canonical_key": candidate_key,
                    "right_canonical_label": core.safe_str(candidate_row["canonical_label_current"]),
                    "right_source_types": core.safe_str(candidate_row.get("source_types", "")),
                    "right_methods": core.safe_str(candidate_row.get("current_methods", "")),
                    "right_example_mentions": core.safe_str(candidate_row.get("example_mentions", "")),
                    "right_source_labels": core.safe_str(candidate_row.get("source_normalized_labels", "")),
                    "right_example_contexts": core.safe_str(candidate_row.get("example_contexts", "")),
                    "right_top_co_mentions": core.safe_str(candidate_row.get("top_co_mentions", "")),
                    "right_mention_count": int(candidate_row.get("mention_count", 0)),
                    "right_unique_docs": int(candidate_row.get("unique_docs", 0)),
                    "right_needs_attention": candidate_attention,
                    "rule_reason": rule_reason,
                    "recommended_action": recommended_action,
                    **score,
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["recommended_action", "final_score", "left_mention_count", "right_mention_count"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    return out


def _a5_adjudication_key(row: pd.Series) -> str:
    raw = "||".join(
        [
            "stage_a5_consolidation_v1",
            core.safe_str(row.get("left_canonical_key", "")),
            core.safe_str(row.get("left_canonical_label", "")),
            core.safe_str(row.get("right_canonical_key", "")),
            core.safe_str(row.get("right_canonical_label", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _a5_openai_prompt(row: pd.Series) -> str:
    return f"""
You are performing a final consolidation pass over canonical entity clusters extracted from historical intelligence documents.

Decide whether the two canonical entities below should be merged into the same real-world entity.
Be conservative. Prefer keep_separate when the evidence is mixed.

Return only valid JSON:
{{
  "decision": "merge|keep_separate",
  "canonical_label": "best merged canonical label if decision is merge, else empty",
  "confidence": 0.0,
  "reason": "short explanation"
}}

Entity A:
- canonical_label: {core.safe_str(row.get("left_canonical_label", ""))}
- source_types: {core.safe_str(row.get("left_source_types", ""))}
- methods: {core.safe_str(row.get("left_methods", ""))}
- mention_count: {core.safe_str(row.get("left_mention_count", ""))}
- unique_docs: {core.safe_str(row.get("left_unique_docs", ""))}
- source_labels: {core.safe_str(row.get("left_source_labels", ""))}
- example_mentions: {core.safe_str(row.get("left_example_mentions", ""))}
- example_contexts: {core.safe_str(row.get("left_example_contexts", ""))}
- top_co_mentions: {core.safe_str(row.get("left_top_co_mentions", ""))}

Entity B:
- canonical_label: {core.safe_str(row.get("right_canonical_label", ""))}
- source_types: {core.safe_str(row.get("right_source_types", ""))}
- methods: {core.safe_str(row.get("right_methods", ""))}
- mention_count: {core.safe_str(row.get("right_mention_count", ""))}
- unique_docs: {core.safe_str(row.get("right_unique_docs", ""))}
- source_labels: {core.safe_str(row.get("right_source_labels", ""))}
- example_mentions: {core.safe_str(row.get("right_example_mentions", ""))}
- example_contexts: {core.safe_str(row.get("right_example_contexts", ""))}
- top_co_mentions: {core.safe_str(row.get("right_top_co_mentions", ""))}

Similarity signals:
- rule_reason: {core.safe_str(row.get("rule_reason", ""))}
- final_score: {core.safe_str(row.get("final_score", ""))}
- lexical_cosine: {core.safe_str(row.get("lexical_cosine", ""))}
- context_cosine: {core.safe_str(row.get("context_cosine", ""))}
- weighted_token_jaccard: {core.safe_str(row.get("weighted_token_jaccard", ""))}
- token_containment: {max(float(row.get("alias_token_containment", 0.0) or 0.0), float(row.get("candidate_token_containment", 0.0) or 0.0))}
- type_compatibility_note: {core.safe_str(row.get("type_compatibility_note", ""))}
""".strip()


def run_a5_openai_adjudication(
    queue: pd.DataFrame,
    out_dir: Path,
    model: str,
    max_candidates: Optional[int],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(queue) == 0:
        return pd.DataFrame(
            columns=[
                "adjudication_key",
                "left_canonical_key",
                "right_canonical_key",
                "decision",
                "canonical_label",
                "confidence",
                "reason",
            ]
        )

    if core.OpenAI is None:
        raise ImportError("OpenAI package is not installed. Run: pip install openai")
    if not core.os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    path = out_dir / A5_ADJUDICATIONS_FILENAME
    error_path = out_dir / A5_ADJUDICATION_ERRORS_FILENAME
    existing_rows = core.load_jsonl(path)
    existing_map = {
        core.safe_str(row.get("adjudication_key", "")): row
        for row in existing_rows
        if core.safe_str(row.get("adjudication_key", ""))
    }

    queue = queue.copy()
    queue["adjudication_key"] = queue.apply(_a5_adjudication_key, axis=1)
    if max_candidates is not None:
        queue = queue.head(max_candidates)
    pending_queue = queue[~queue["adjudication_key"].isin(existing_map)].copy()
    cached_count = int(len(queue) - len(pending_queue))
    print(
        "A5 OpenAI consolidation queue:",
        {
            "total": int(len(queue)),
            "cached": cached_count,
            "pending": int(len(pending_queue)),
        },
    )
    if len(pending_queue) == 0:
        return pd.DataFrame(existing_map.values())

    client = core.OpenAI(
        api_key=core.os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        max_retries=2,
    )

    iterator = pending_queue.iterrows()
    if core.tqdm is not None:
        iterator = core.tqdm(
            iterator,
            total=len(queue),
            initial=cached_count,
            desc="A5 consolidation adjudications",
            unit="pair",
        )

    for _, row in iterator:
        key = core.safe_str(row["adjudication_key"])
        prompt = _a5_openai_prompt(row)
        try:
            response = core.call_openai_json(client, prompt=prompt, model=model)
        except Exception as exc:
            error_record = {
                "adjudication_key": key,
                "left_canonical_key": core.safe_str(row.get("left_canonical_key", "")),
                "right_canonical_key": core.safe_str(row.get("right_canonical_key", "")),
                "model": model,
                "error_type": type(exc).__name__,
                "error": core.safe_str(exc),
            }
            core.append_jsonl(error_path, error_record)
            print(f"Skipping A5 adjudication for {row.get('left_canonical_label', '')} <> {row.get('right_canonical_label', '')}: {type(exc).__name__}: {exc}")
            continue

        record = {
            "adjudication_key": key,
            "left_canonical_key": core.safe_str(row.get("left_canonical_key", "")),
            "right_canonical_key": core.safe_str(row.get("right_canonical_key", "")),
            "model": model,
            "decision": core.safe_str(response.get("decision", "")),
            "canonical_label": core.safe_str(response.get("canonical_label", "")),
            "confidence": float(response.get("confidence", 0.0) or 0.0),
            "reason": core.safe_str(response.get("reason", "")),
        }
        core.append_jsonl(path, record)
        existing_map[key] = record

    return pd.DataFrame(existing_map.values())


def _build_consolidation_label_from_pair(row: pd.Series) -> str:
    suggestion = core.norm_space(row.get("openai_canonical_label", ""))
    if suggestion:
        return suggestion

    pair = pd.DataFrame(
        [
            {
                "canonical_label_current": row.get("left_canonical_label", ""),
                "current_methods": row.get("left_methods", ""),
                "source_label_count": row.get("left_mention_count", 0),
                "mention_count": row.get("left_mention_count", 0),
                "unique_docs": row.get("left_unique_docs", 0),
            },
            {
                "canonical_label_current": row.get("right_canonical_label", ""),
                "current_methods": row.get("right_methods", ""),
                "source_label_count": row.get("right_mention_count", 0),
                "mention_count": row.get("right_mention_count", 0),
                "unique_docs": row.get("right_unique_docs", 0),
            },
        ]
    )
    return _choose_best_member_label(pair)


def finalize_a5_pair_decisions(
    candidate_pairs: pd.DataFrame,
    adjudications: pd.DataFrame,
) -> pd.DataFrame:
    if len(candidate_pairs) == 0:
        return pd.DataFrame()

    out = candidate_pairs.copy()
    out["adjudication_key"] = out.apply(_a5_adjudication_key, axis=1)

    if len(adjudications):
        adjudications = adjudications.rename(
            columns={
                "decision": "openai_decision",
                "canonical_label": "openai_canonical_label",
                "confidence": "openai_confidence",
                "reason": "openai_reason",
            }
        )
        out = out.merge(
            adjudications[
                [
                    "adjudication_key",
                    "openai_decision",
                    "openai_canonical_label",
                    "openai_confidence",
                    "openai_reason",
                ]
            ],
            on="adjudication_key",
            how="left",
        )
    else:
        out["openai_decision"] = ""
        out["openai_canonical_label"] = ""
        out["openai_confidence"] = np.nan
        out["openai_reason"] = ""

    final_decision: List[str] = []
    final_method: List[str] = []
    final_confidence: List[float] = []
    final_label: List[str] = []
    for _, row in out.iterrows():
        action = core.safe_str(row.get("recommended_action", ""))
        openai_decision = core.safe_str(row.get("openai_decision", ""))
        openai_confidence = float(row.get("openai_confidence", 0.0) or 0.0)
        if action == "accept_local":
            final_decision.append("merge")
            final_method.append("a5_auto_merge")
            final_confidence.append(float(row.get("final_score", 1.0) or 1.0))
            final_label.append(_build_consolidation_label_from_pair(row))
        elif openai_decision == "merge" and openai_confidence >= core.OPENAI_ACCEPT_CONFIDENCE_THRESHOLD:
            final_decision.append("merge")
            final_method.append("a5_openai_merge")
            final_confidence.append(openai_confidence)
            final_label.append(_build_consolidation_label_from_pair(row))
        else:
            final_decision.append("keep_separate")
            final_method.append("a5_keep_separate")
            final_confidence.append(openai_confidence if openai_confidence > 0 else float(row.get("final_score", 0.0) or 0.0))
            final_label.append("")

    out["final_decision"] = final_decision
    out["final_method"] = final_method
    out["final_confidence"] = final_confidence
    out["final_canonical_label"] = final_label
    return out


def build_a5_consolidation_map(
    canonical_catalog: pd.DataFrame,
    pair_decisions: pd.DataFrame,
) -> pd.DataFrame:
    if len(canonical_catalog) == 0:
        return pd.DataFrame()

    parent: Dict[str, str] = {
        core.safe_str(row["canonical_key_current"]): core.safe_str(row["canonical_key_current"])
        for _, row in canonical_catalog.iterrows()
        if core.safe_str(row["canonical_key_current"])
    }

    def find(value: str) -> str:
        root = parent[value]
        while root != parent[root]:
            parent[root] = parent[parent[root]]
            root = parent[root]
        while value != root:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    accepted = pair_decisions[pair_decisions["final_decision"].astype(str).eq("merge")].copy() if len(pair_decisions) else pd.DataFrame()
    for _, row in accepted.iterrows():
        left = core.safe_str(row.get("left_canonical_key", ""))
        right = core.safe_str(row.get("right_canonical_key", ""))
        if left and right and left in parent and right in parent:
            union(left, right)

    group_members: Dict[str, List[str]] = defaultdict(list)
    for key in parent:
        group_members[find(key)].append(key)

    pair_suggestions: Dict[str, List[str]] = defaultdict(list)
    pair_confidences: Dict[str, List[float]] = defaultdict(list)
    if len(accepted):
        for _, row in accepted.iterrows():
            left = core.safe_str(row.get("left_canonical_key", ""))
            if not left or left not in parent:
                continue
            root = find(left)
            label = core.norm_space(row.get("final_canonical_label", ""))
            if label:
                pair_suggestions[root].append(label)
            pair_confidences[root].append(float(row.get("final_confidence", 0.0) or 0.0))

    catalog_lookup = {
        core.safe_str(row["canonical_key_current"]): row
        for _, row in canonical_catalog.iterrows()
    }

    rows: List[Dict[str, object]] = []
    for root, member_keys in group_members.items():
        member_rows = canonical_catalog[
            canonical_catalog["canonical_key_current"].astype(str).isin(member_keys)
        ].copy()
        suggestion_counts = Counter(pair_suggestions.get(root, []))
        suggested_labels = [
            label
            for label, _count in suggestion_counts.most_common()
            if label
        ]
        if len(member_keys) == 1:
            final_label = core.norm_space(member_rows.iloc[0]["canonical_label_current"])
            final_key = core.safe_str(member_rows.iloc[0]["canonical_key_current"])
            group_confidence = 1.0
            group_method = "unchanged"
        else:
            final_label = _choose_best_member_label(member_rows, preferred_labels=suggested_labels)
            if not final_label and len(member_rows):
                final_label = core.norm_space(member_rows.iloc[0]["canonical_label_current"])
            matching_member = member_rows[
                member_rows["canonical_label_current"].astype(str).eq(final_label)
            ]
            if len(matching_member):
                final_key = core.safe_str(matching_member.iloc[0]["canonical_key_current"])
            else:
                final_key = f"CANONICAL_ENTITY:{core.keyify(final_label)}"
            group_confidence = max(pair_confidences.get(root, [1.0]))
            group_method = "a5_consolidated"

        for member_key in member_keys:
            member_row = catalog_lookup[member_key]
            current_label = core.safe_str(member_row.get("canonical_label_current", ""))
            rows.append(
                {
                    "canonical_key_current": member_key,
                    "canonical_label_current": current_label,
                    "final_canonical_key": final_key,
                    "final_canonical_label": final_label,
                    "a5_group_root": root,
                    "a5_group_size": int(len(member_keys)),
                    "a5_group_method": group_method,
                    "a5_group_confidence": round(float(group_confidence), 4),
                    "a5_changed": bool(len(member_keys) > 1 and (member_key != final_key or current_label != final_label)),
                    "a5_suggested_labels": "; ".join(suggested_labels[:8]),
                    "a5_member_labels": "; ".join(sorted(member_rows["canonical_label_current"].astype(str).tolist())[:12]),
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["a5_group_size", "a5_changed", "canonical_label_current"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return out


def apply_a5_consolidation(
    entities_resolved: pd.DataFrame,
    consolidation_map: pd.DataFrame,
) -> pd.DataFrame:
    if len(entities_resolved) == 0 or len(consolidation_map) == 0:
        return entities_resolved.copy()

    out = entities_resolved.copy()
    prior_a5_cols = [
        "a5_changed",
        "a5_group_size",
        "a5_group_confidence",
        "a5_group_method",
        "final_canonical_key",
        "final_canonical_label",
        "canonical_key_current",
        "a5_group_root",
    ]
    out = out.drop(columns=[col for col in prior_a5_cols if col in out.columns], errors="ignore")
    map_view = consolidation_map.loc[
        :,
        [
            "canonical_key_current",
            "final_canonical_key",
            "final_canonical_label",
            "a5_group_size",
            "a5_group_method",
            "a5_group_confidence",
            "a5_changed",
        ],
    ].drop_duplicates(subset=["canonical_key_current"])
    out = out.merge(map_view, left_on="canonical_key", right_on="canonical_key_current", how="left")

    out["pre_a5_canonical_label"] = out.get("canonical_label", "")
    out["pre_a5_canonical_key"] = out.get("canonical_key", "")
    out["pre_a5_canonical_method"] = out.get("canonical_method", "")
    out["pre_a5_canonical_confidence"] = pd.to_numeric(out.get("canonical_confidence", np.nan), errors="coerce")

    changed = out["a5_changed"].fillna(False).map(_safe_bool)
    out["canonical_label"] = np.where(changed, out["final_canonical_label"], out["canonical_label"])
    out["canonical_key"] = np.where(changed, out["final_canonical_key"], out["canonical_key"])
    out["canonical_method"] = np.where(changed, "a5_consolidated", out["canonical_method"])
    out["canonical_confidence"] = np.where(
        changed,
        pd.to_numeric(out["a5_group_confidence"], errors="coerce").fillna(1.0),
        pd.to_numeric(out["canonical_confidence"], errors="coerce").fillna(1.0),
    )
    out["a5_changed"] = changed
    out["a5_group_size"] = pd.to_numeric(out["a5_group_size"], errors="coerce").fillna(1).astype(int)
    out["a5_group_confidence"] = pd.to_numeric(out["a5_group_confidence"], errors="coerce").fillna(1.0)
    out = out.drop(columns=["canonical_key_current"], errors="ignore")
    return out


def run_stage_a5_consolidation(
    project_dir: Path,
    run_openai: bool = True,
    model: str = "gpt-4o-mini",
    max_candidates: Optional[int] = None,
    top_k_neighbors: int = 10,
    review_min_score: float = 0.72,
    attention_review_min_score: float = 0.58,
) -> Dict[str, pd.DataFrame]:
    project_dir = Path(project_dir).expanduser().resolve()
    out_dir = core.entity_resolution_dir(project_dir)
    entities_resolved, mention_catalog, canonical_map_before = load_a5_inputs(project_dir)

    canonical_catalog = build_a5_canonical_catalog(entities_resolved, mention_catalog)
    candidate_pairs = build_a5_candidate_pairs(
        canonical_catalog,
        top_k_neighbors=top_k_neighbors,
        review_min_score=review_min_score,
        attention_review_min_score=attention_review_min_score,
    )
    auto_accepted = candidate_pairs[candidate_pairs["recommended_action"].astype(str).eq("accept_local")].copy() if len(candidate_pairs) else pd.DataFrame()
    review_queue = candidate_pairs[candidate_pairs["recommended_action"].astype(str).eq("needs_openai_review")].copy() if len(candidate_pairs) else pd.DataFrame()

    print("A5 canonical catalog rows:", len(canonical_catalog))
    print("A5 candidate pair rows:", len(candidate_pairs))
    print("A5 auto-accepted rows:", len(auto_accepted))
    print("A5 review queue rows:", len(review_queue))

    adjudications = pd.DataFrame()
    if run_openai and len(review_queue):
        adjudications = run_a5_openai_adjudication(
            queue=review_queue,
            out_dir=out_dir,
            model=model,
            max_candidates=max_candidates,
        )

    pair_decisions = finalize_a5_pair_decisions(candidate_pairs, adjudications)
    consolidation_map = build_a5_consolidation_map(canonical_catalog, pair_decisions)
    entities_resolved_a5 = apply_a5_consolidation(entities_resolved, consolidation_map)
    canonical_map_a5 = a4.build_canonical_map_from_resolved_entities(entities_resolved_a5)
    generated_aliases_a5 = a4.build_generated_alias_table_from_canonical_map(canonical_map_a5)
    alias_for_b = canonical_map_a5.loc[
        :,
        ["original_normalized_label", "canonical_label", "canonical_key", "method", "confidence"],
    ].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    entities_resolved.to_csv(out_dir / A5_PREVIOUS_ENTITIES_SNAPSHOT, index=False)
    canonical_map_before.to_csv(out_dir / A5_PREVIOUS_CANONICAL_MAP_SNAPSHOT, index=False)
    canonical_catalog.to_csv(out_dir / A5_CANONICAL_CATALOG_FILENAME, index=False)
    candidate_pairs.to_csv(out_dir / A5_CANDIDATE_PAIRS_FILENAME, index=False)
    auto_accepted.to_csv(out_dir / A5_AUTO_ACCEPTED_FILENAME, index=False)
    review_queue.to_csv(out_dir / A5_REVIEW_QUEUE_FILENAME, index=False)
    pair_decisions.to_csv(out_dir / A5_PAIR_DECISIONS_FILENAME, index=False)
    consolidation_map.to_csv(out_dir / A5_CONSOLIDATION_MAP_FILENAME, index=False)
    entities_resolved_a5.to_csv(out_dir / A5_ENTITIES_RESOLVED_FILENAME, index=False)
    canonical_map_a5.to_csv(out_dir / A5_CANONICAL_MAP_FILENAME, index=False)
    generated_aliases_a5.to_csv(out_dir / A5_GENERATED_ALIASES_FILENAME, index=False)
    if len(review_queue):
        llm_view = pair_decisions[pair_decisions["recommended_action"].astype(str).eq("needs_openai_review")].copy()
    else:
        llm_view = pd.DataFrame()
    llm_view.to_csv(out_dir / A5_LLM_FILENAME, index=False)

    entities_resolved_a5.to_csv(out_dir / core.ENTITIES_RESOLVED_FILENAME, index=False)
    canonical_map_a5.to_csv(out_dir / core.ENTITY_CANONICAL_MAP_FILENAME, index=False)
    alias_for_b.to_csv(out_dir / core.ENTITY_ALIAS_MAP_FOR_B_FILENAME, index=False)
    generated_aliases_a5.to_csv(out_dir / core.GENERATED_ALIAS_FILENAME, index=False)
    generated_aliases_a5.to_csv(out_dir / "entity_aliases_proposed.csv", index=False)

    summary_path = out_dir / "summary.json"
    existing_summary: Dict[str, object] = {}
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing_summary = {}

    summary = {
        **existing_summary,
        "a5_canonical_catalog_rows": int(len(canonical_catalog)),
        "a5_candidate_pair_rows": int(len(candidate_pairs)),
        "a5_auto_accepted_rows": int(len(auto_accepted)),
        "a5_review_queue_rows": int(len(review_queue)),
        "a5_adjudication_rows": int(len(adjudications)),
        "a5_merge_rows": int((pair_decisions["final_decision"] == "merge").sum()) if len(pair_decisions) else 0,
        "a5_openai_merge_rows": int((pair_decisions["final_method"] == "a5_openai_merge").sum()) if len(pair_decisions) else 0,
        "a5_auto_merge_rows": int((pair_decisions["final_method"] == "a5_auto_merge").sum()) if len(pair_decisions) else 0,
        "a5_keep_separate_rows": int((pair_decisions["final_method"] == "a5_keep_separate").sum()) if len(pair_decisions) else 0,
        "a5_consolidated_components": int(consolidation_map["a5_group_root"].astype(str).nunique()) if len(consolidation_map) else 0,
        "a5_changed_canonical_rows": int(consolidation_map["a5_changed"].fillna(False).map(_safe_bool).sum()) if len(consolidation_map) else 0,
        "a5_changed_entity_mentions": int(entities_resolved_a5["a5_changed"].fillna(False).map(_safe_bool).sum()) if len(entities_resolved_a5) else 0,
        "resolved_entity_rows": int(len(entities_resolved_a5)),
        "canonical_map_rows": int(len(canonical_map_a5)),
        "generated_alias_rows": int(len(generated_aliases_a5)),
    }
    core.write_stage_summary(project_dir, summary)
    print(json.dumps(summary, indent=2))

    return {
        "canonical_catalog": canonical_catalog,
        "candidate_pairs": candidate_pairs,
        "review_queue": review_queue,
        "adjudications": adjudications,
        "pair_decisions": pair_decisions,
        "consolidation_map": consolidation_map,
        "entities_resolved": entities_resolved_a5,
        "canonical_map": canonical_map_a5,
        "summary": summary,
    }
