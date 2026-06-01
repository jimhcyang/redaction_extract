"""
Stage A3: split ambiguous labels into mention-level sense clusters.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import (
    AMBIGUOUS_LABELS_FILENAME,
    CLUSTER_CANDIDATES_FILENAME,
    CLUSTER_REVIEW_QUEUE_FILENAME,
    ENTITY_INVENTORY_FILENAME,
    GENERIC_ENTITY_TOKENS,
    MENTION_CATALOG_FILENAME,
    OPENAI_ACCEPT_CONFIDENCE_THRESHOLD,
    RULE_RESOLVED_FILENAME,
    SENSE_ASSIGNMENTS_FILENAME,
    SENSE_CLUSTERS_FILENAME,
    acronym_from_label,
    build_token_idf_map,
    cooccurrence_jaccard,
    entity_resolution_dir,
    first_existing_col,
    infer_structural_family,
    iter_progress,
    label_ambiguity_score,
    looks_like_acronym,
    mean_pairwise_jaccard,
    normalize_label_basic,
    require_stage_csv,
    safe_str,
    structural_conflict_reason,
    token_set,
    type_compatibility,
    type_set_from_text,
    weighted_token_scores,
    _flatten_semicolon_values,
    _sample_text_list,
)
from .scoring import build_similarity_models


def build_cluster_descriptor(
    normalized_label: str,
    source_types: Sequence[str],
    context_texts: Sequence[str],
    top_co_mentions: Sequence[str],
    token_idf: Dict[str, float],
) -> str:
    label_tokens = token_set(normalized_label) | {normalized_label}
    score_counter: Dict[str, float] = defaultdict(float)
    for text in context_texts:
        for token in token_set(text):
            if token in label_tokens or token in GENERIC_ENTITY_TOKENS:
                continue
            score_counter[token] += token_idf.get(token, 1.0)
    for token in top_co_mentions:
        normalized = normalize_label_basic(token)
        if not normalized or normalized in label_tokens:
            continue
        for piece in token_set(normalized):
            if piece not in GENERIC_ENTITY_TOKENS and piece not in label_tokens:
                score_counter[piece] += token_idf.get(piece, 1.0) * 0.75
    if source_types:
        type_hint = normalize_label_basic(source_types[0])
        if type_hint and type_hint not in {"PERSON", "OTHER"}:
            score_counter[type_hint] += 0.25
    ranked = sorted(score_counter.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_tokens = [token for token, _score in ranked[:3]]
    return " ".join(top_tokens).strip()


def _connected_components_from_similarity(similarity: np.ndarray, threshold: float) -> List[int]:
    parent = list(range(similarity.shape[0]))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(similarity.shape[0]):
        for right in range(left + 1, similarity.shape[0]):
            if similarity[left, right] >= threshold:
                union(left, right)

    root_to_cluster: Dict[int, int] = {}
    labels: List[int] = []
    next_cluster = 0
    for idx in range(similarity.shape[0]):
        root = find(idx)
        if root not in root_to_cluster:
            root_to_cluster[root] = next_cluster
            next_cluster += 1
        labels.append(root_to_cluster[root])
    return labels


def cluster_mentions_for_label(
    label_mentions: pd.DataFrame,
    ambiguity_score: float,
) -> List[int]:
    if len(label_mentions) <= 1:
        return [0] * len(label_mentions)

    texts = label_mentions["context_text"].fillna("").astype(str).tolist()
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
        matrix = vectorizer.fit_transform(texts)
        similarity = (matrix * matrix.T).toarray()
    except Exception:
        return [0] * len(label_mentions)

    threshold = 0.55 if ambiguity_score >= 0.45 else 0.50 if ambiguity_score >= 0.30 else 0.45
    labels = _connected_components_from_similarity(similarity, threshold=threshold)
    if len(set(labels)) == 1 and ambiguity_score >= 0.55 and len(label_mentions) >= 4:
        labels = _connected_components_from_similarity(similarity, threshold=min(0.70, threshold + 0.10))
    return labels


def _cluster_query_neighbors(
    text: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    neighbors: NearestNeighbors,
    catalog: pd.DataFrame,
    max_neighbors: int = 12,
) -> List[str]:
    if len(catalog) == 0:
        return []
    query = vectorizer.transform([text or "UNKNOWN"])
    distances, indices = neighbors.kneighbors(query, n_neighbors=min(max_neighbors, matrix.shape[0]))
    out: List[str] = []
    for idx in indices[0]:
        label = safe_str(catalog.iloc[int(idx)]["normalized_label"])
        if label:
            out.append(label)
    return out


def score_cluster_candidate_label(
    cluster_row: pd.Series,
    candidate_row: pd.Series,
    token_idf: Dict[str, float],
) -> float:
    cluster_label = safe_str(cluster_row.get("normalized_label", ""))
    candidate_label = safe_str(candidate_row.get("normalized_label", ""))
    cluster_types = type_set_from_text(cluster_row.get("source_types", ""))
    candidate_types = type_set_from_text(candidate_row.get("source_types", ""))
    conflict = structural_conflict_reason(cluster_label, candidate_label, cluster_types, candidate_types)
    if conflict:
        return 0.0

    type_score, _ = type_compatibility(cluster_types, candidate_types)
    cluster_tokens = token_set(safe_str(cluster_row.get("context_text", "")))
    candidate_tokens = token_set(safe_str(candidate_row.get("context_text", "")))
    weighted_jaccard, left_containment, right_containment = weighted_token_scores(cluster_tokens, candidate_tokens, token_idf)
    co_score = cooccurrence_jaccard(
        safe_str(cluster_row.get("top_co_mentions", "")),
        safe_str(candidate_row.get("top_co_mentions", "")),
    )
    acronym_bonus = 0.40 if acronym_from_label(candidate_label) == cluster_label else 0.0
    mention_bonus = min(0.10, math.log1p(float(candidate_row.get("mention_count", 0))) / 20.0)
    return max(
        0.0,
        0.35 * weighted_jaccard
        + 0.20 * max(left_containment, right_containment)
        + 0.15 * co_score
        + 0.15 * type_score
        + acronym_bonus
        + mention_bonus,
    )


def plan_single_cluster_resolution(
    normalized_label: str,
    cluster_total: int,
    ambiguity_score: float,
    looks_acronym: bool,
    scored_candidates: List[Tuple[float, pd.Series]],
) -> Dict[str, object]:
    top_candidate_label = ""
    top_candidate_score = 0.0
    second_candidate_score = 0.0
    top_candidate_is_expansion = False

    if scored_candidates:
        top_candidate_score, top_candidate_row = scored_candidates[0]
        top_candidate_label = safe_str(top_candidate_row.get("normalized_label", ""))
        second_candidate_score = float(scored_candidates[1][0]) if len(scored_candidates) > 1 else 0.0
        top_candidate_is_expansion = acronym_from_label(top_candidate_label) == normalized_label

    score_gap = max(0.0, float(top_candidate_score) - float(second_candidate_score))
    auto_label = ""
    auto_method = ""
    auto_confidence = np.nan

    if (
        looks_acronym
        and top_candidate_is_expansion
        and top_candidate_score >= 0.78
        and score_gap >= 0.12
    ):
        auto_label = top_candidate_label
        auto_method = "a3_cluster_auto_merge"
        auto_confidence = round(min(0.99, max(0.78, float(top_candidate_score))), 4)

    needs_llm = False if auto_method else bool(cluster_total > 1)
    if not auto_method:
        if looks_acronym:
            needs_llm = True
        elif ambiguity_score >= 0.45:
            needs_llm = True
        elif cluster_total == 1 and top_candidate_label and top_candidate_score >= 0.76 and score_gap >= 0.12:
            needs_llm = True

    return {
        "needs_llm": needs_llm,
        "top_candidate_label": top_candidate_label,
        "top_candidate_score": round(float(top_candidate_score), 4) if top_candidate_label else 0.0,
        "top_candidate_score_gap": round(float(score_gap), 4) if top_candidate_label else 0.0,
        "top_candidate_is_expansion": bool(top_candidate_is_expansion),
        "auto_resolution_label": auto_label,
        "auto_resolution_method": auto_method,
        "auto_resolution_confidence": auto_confidence,
    }


def build_sense_clusters(
    mention_catalog: pd.DataFrame,
    label_catalog: pd.DataFrame,
    ambiguous_labels: pd.DataFrame,
    max_candidate_labels_per_cluster: int,
    top_k_neighbors: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(mention_catalog) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ambiguous_map = {
        safe_str(row["normalized_label"]): row
        for _, row in ambiguous_labels.iterrows()
    }
    catalog_lookup = {
        safe_str(row["normalized_label"]): row
        for _, row in label_catalog.iterrows()
    }
    token_idf = build_token_idf_map(label_catalog)
    (
        label_vectorizer,
        label_matrix,
        label_neighbors,
        context_vectorizer,
        context_matrix,
        context_neighbors,
    ) = build_similarity_models(label_catalog, top_k_neighbors=max(top_k_neighbors, max_candidate_labels_per_cluster * 3))

    assignments: List[Dict[str, object]] = []
    cluster_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []

    grouped_mentions = mention_catalog.groupby("normalized_label")
    for normalized_label, group in iter_progress(grouped_mentions, total=mention_catalog["normalized_label"].nunique(), desc="A3 sense clustering"):
        ambiguous_row = ambiguous_map.get(normalized_label)
        needs_sense = bool(ambiguous_row["needs_sense_clustering"]) if ambiguous_row is not None else False
        ambiguity_score = float(ambiguous_row["label_ambiguity"]) if ambiguous_row is not None else 0.0

        if needs_sense:
            cluster_labels = cluster_mentions_for_label(group.reset_index(drop=True), ambiguity_score=ambiguity_score)
        else:
            cluster_labels = [0] * len(group)

        cluster_total = len(set(cluster_labels))
        group = group.copy().reset_index(drop=True)
        group["cluster_idx"] = cluster_labels

        for cluster_idx, cluster_group in group.groupby("cluster_idx"):
            cluster_id = f"{normalized_label}__S{int(cluster_idx) + 1}"
            example_mentions = _sample_text_list(cluster_group["raw_label"].tolist(), limit=8)
            example_descriptions = _sample_text_list(cluster_group["description"].tolist(), limit=6)
            example_contexts = _sample_text_list(cluster_group["context_text"].tolist(), limit=3)
            source_types = sorted(set(value for value in cluster_group["source_type"].astype(str) if value))
            top_co_mentions = _sample_text_list(
                _flatten_semicolon_values(cluster_group["sibling_entities"].tolist()),
                limit=10,
            )
            descriptor = build_cluster_descriptor(
                normalized_label=normalized_label,
                source_types=source_types,
                context_texts=example_contexts,
                top_co_mentions=top_co_mentions,
                token_idf=token_idf,
            )
            context_text = " || ".join(
                bit
                for bit in [
                    normalized_label,
                    " ".join(source_types[:6]),
                    " ".join(example_descriptions[:4]),
                    " ".join(example_contexts[:2]),
                    " ".join(top_co_mentions[:8]),
                ]
                if bit
            )

            for _, mention_row in cluster_group.iterrows():
                assignments.append(
                    {
                        "mention_key": safe_str(mention_row["mention_key"]),
                        "normalized_label": normalized_label,
                        "sense_id": cluster_id,
                        "sense_cluster_index": int(cluster_idx),
                        "label_cluster_count": int(cluster_total),
                    }
                )

            cluster_row = {
                "sense_id": cluster_id,
                "normalized_label": normalized_label,
                "sense_cluster_index": int(cluster_idx),
                "label_cluster_count": int(cluster_total),
                "mention_count": int(len(cluster_group)),
                "unique_docs": int(cluster_group["doc_id"].astype(str).nunique()),
                "unique_paragraphs": int(cluster_group[["doc_id", "paragraph_id"]].astype(str).drop_duplicates().shape[0]),
                "source_types": "; ".join(source_types[:8]),
                "primary_type": source_types[0] if source_types else "",
                "example_mentions": "; ".join(example_mentions),
                "example_descriptions": "; ".join(example_descriptions),
                "example_contexts": " || ".join(example_contexts),
                "top_co_mentions": "; ".join(top_co_mentions),
                "descriptor": descriptor,
                "context_text": context_text,
                "label_ambiguity": ambiguity_score,
                "looks_like_acronym": looks_like_acronym(normalized_label),
                "needs_llm": False,
                "top_candidate_label": "",
                "top_candidate_score": 0.0,
                "top_candidate_score_gap": 0.0,
                "top_candidate_is_expansion": False,
                "auto_resolution_label": "",
                "auto_resolution_method": "",
                "auto_resolution_confidence": np.nan,
            }

            candidate_labels: List[str] = []
            for candidate_label in _cluster_query_neighbors(context_text, context_vectorizer, context_matrix, context_neighbors, label_catalog):
                if candidate_label != normalized_label:
                    candidate_labels.append(candidate_label)
            for candidate_label in _cluster_query_neighbors(normalized_label, label_vectorizer, label_matrix, label_neighbors, label_catalog):
                if candidate_label != normalized_label:
                    candidate_labels.append(candidate_label)
            if looks_like_acronym(normalized_label):
                for candidate_label in label_catalog["normalized_label"].astype(str):
                    if candidate_label != normalized_label and acronym_from_label(candidate_label) == normalized_label:
                        candidate_labels.append(candidate_label)

            deduped_candidates: List[str] = []
            seen_candidates: Set[str] = set()
            for candidate_label in candidate_labels:
                if candidate_label in seen_candidates:
                    continue
                seen_candidates.add(candidate_label)
                deduped_candidates.append(candidate_label)

            scored_candidates: List[Tuple[float, pd.Series]] = []
            cluster_series = pd.Series(cluster_row)
            for candidate_label in deduped_candidates:
                candidate_row = catalog_lookup.get(candidate_label)
                if candidate_row is None:
                    continue
                score = score_cluster_candidate_label(cluster_series, candidate_row, token_idf)
                if score <= 0.0:
                    continue
                scored_candidates.append((score, candidate_row))

            scored_candidates = sorted(
                scored_candidates,
                key=lambda item: (
                    item[0],
                    int(item[1].get("mention_count", 0)),
                    int(item[1].get("unique_docs", 0)),
                ),
                reverse=True,
            )[:max_candidate_labels_per_cluster]

            for rank, (score, candidate_row) in enumerate(scored_candidates, start=1):
                candidate_rows.append(
                    {
                        "sense_id": cluster_id,
                        "normalized_label": normalized_label,
                        "candidate_rank": rank,
                        "candidate_label": safe_str(candidate_row["normalized_label"]),
                        "candidate_source_types": safe_str(candidate_row.get("source_types", "")),
                        "candidate_example_mentions": safe_str(candidate_row.get("example_mentions", "")),
                        "candidate_example_contexts": safe_str(candidate_row.get("example_contexts", "")),
                        "candidate_context_text": safe_str(candidate_row.get("context_text", "")),
                        "candidate_score": round(float(score), 4),
                    }
                )

            resolution_plan = plan_single_cluster_resolution(
                normalized_label=normalized_label,
                cluster_total=cluster_total,
                ambiguity_score=ambiguity_score,
                looks_acronym=bool(cluster_row["looks_like_acronym"]),
                scored_candidates=scored_candidates,
            )
            cluster_row.update(resolution_plan)
            cluster_rows.append(cluster_row)

    assignments_df = pd.DataFrame(assignments)
    clusters_df = pd.DataFrame(cluster_rows).sort_values(
        ["needs_llm", "label_ambiguity", "mention_count", "sense_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    candidates_df = pd.DataFrame(candidate_rows) if candidate_rows else pd.DataFrame(
        columns=[
            "sense_id",
            "normalized_label",
            "candidate_rank",
            "candidate_label",
            "candidate_source_types",
            "candidate_example_mentions",
            "candidate_example_contexts",
            "candidate_context_text",
            "candidate_score",
        ]
    )
    if len(candidates_df):
        candidates_df = candidates_df.sort_values(["sense_id", "candidate_rank"]).reset_index(drop=True)
    return assignments_df, clusters_df, candidates_df


def cluster_adjudication_key(row: pd.Series) -> str:
    raw = "||".join(
        [
            "stage_a4_cluster_resolution_v2",
            safe_str(row.get("sense_id", "")),
            safe_str(row.get("normalized_label", "")),
            safe_str(row.get("descriptor", "")),
            safe_str(row.get("candidate_labels", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_cluster_review_queue(
    clusters_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(clusters_df) == 0:
        return pd.DataFrame(
            columns=[
                "sense_id",
                "normalized_label",
                "candidate_labels",
                "sibling_clusters",
                "adjudication_key",
            ]
        )

    candidate_map: Dict[str, List[dict]] = defaultdict(list)
    for _, row in candidates_df.iterrows():
        candidate_map[safe_str(row["sense_id"])].append(row.to_dict())

    label_cluster_map: Dict[str, List[pd.Series]] = defaultdict(list)
    for _, row in clusters_df.iterrows():
        label_cluster_map[safe_str(row["normalized_label"])].append(row)

    rows: List[Dict[str, object]] = []
    for _, row in clusters_df.iterrows():
        if not bool(row.get("needs_llm", False)):
            continue
        sense_id = safe_str(row["sense_id"])
        sibling_bits: List[str] = []
        for sibling in label_cluster_map[safe_str(row["normalized_label"])]:
            if safe_str(sibling["sense_id"]) == sense_id:
                continue
            sibling_bits.append(
                f"{safe_str(sibling['sense_id'])}: {safe_str(sibling.get('descriptor', ''))} | {safe_str(sibling.get('example_contexts', ''))}"
            )
        cluster_candidates = candidate_map.get(sense_id, [])
        candidate_labels = " || ".join(
            f"{cand['candidate_rank']}. {cand['candidate_label']} [{cand['candidate_source_types']}] score={cand['candidate_score']} context={cand['candidate_example_contexts']}"
            for cand in cluster_candidates[:5]
        )
        out_row = row.to_dict()
        out_row["candidate_labels"] = candidate_labels
        out_row["sibling_clusters"] = " || ".join(sibling_bits[:4])
        out_row["adjudication_key"] = cluster_adjudication_key(pd.Series(out_row))
        rows.append(out_row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "sense_id",
                "normalized_label",
                "candidate_labels",
                "sibling_clusters",
                "adjudication_key",
            ]
        )
    return pd.DataFrame(rows)


def run_stage_a3_sense_clustering(
    project_dir: Path,
    top_k_neighbors: int = 15,
    max_candidate_labels_per_cluster: int = 5,
) -> Dict[str, pd.DataFrame]:
    project_dir = Path(project_dir).expanduser().resolve()
    out_dir = entity_resolution_dir(project_dir)

    label_catalog = require_stage_csv(project_dir, "label_catalog.csv")
    rule_resolved_mentions = require_stage_csv(project_dir, RULE_RESOLVED_FILENAME)
    ambiguous_labels = require_stage_csv(project_dir, AMBIGUOUS_LABELS_FILENAME)

    sense_assignments, sense_clusters, cluster_candidates = build_sense_clusters(
        mention_catalog=rule_resolved_mentions,
        label_catalog=label_catalog,
        ambiguous_labels=ambiguous_labels,
        max_candidate_labels_per_cluster=max_candidate_labels_per_cluster,
        top_k_neighbors=top_k_neighbors,
    )
    cluster_review_queue = build_cluster_review_queue(sense_clusters, cluster_candidates)

    out_dir.mkdir(parents=True, exist_ok=True)
    sense_assignments.to_csv(out_dir / SENSE_ASSIGNMENTS_FILENAME, index=False)
    sense_clusters.to_csv(out_dir / SENSE_CLUSTERS_FILENAME, index=False)
    cluster_candidates.to_csv(out_dir / CLUSTER_CANDIDATES_FILENAME, index=False)
    cluster_review_queue.to_csv(out_dir / CLUSTER_REVIEW_QUEUE_FILENAME, index=False)

    print("A3 sense clusters:", len(sense_clusters))
    print("A3 review queue rows:", len(cluster_review_queue))
    return {
        "sense_assignments": sense_assignments,
        "sense_clusters": sense_clusters,
        "cluster_candidates": cluster_candidates,
        "cluster_review_queue": cluster_review_queue,
    }
