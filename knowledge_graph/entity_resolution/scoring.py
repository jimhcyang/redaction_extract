"""
Reusable similarity-model and candidate-scoring utilities.
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .common import (
    GENERIC_ENTITY_TOKENS,
    NEGATIVE_DISTINGUISHER_TOKENS,
    cooccurrence_jaccard,
    infer_structural_family,
    informative_token_gap_penalty,
    jaccard,
    safe_str,
    structural_conflict_reason,
    token_set,
    type_compatibility,
    type_set_from_text,
    weighted_token_scores,
)


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
    token_idf: Dict[str, float],
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
    weighted_jaccard, alias_containment, candidate_containment = weighted_token_scores(
        alias_tokens,
        candidate_tokens,
        token_idf,
    )
    co_mention_score = cooccurrence_jaccard(alias_row.get("top_co_mentions", ""), candidate_row.get("top_co_mentions", ""))
    type_score, type_note = type_compatibility(
        alias_types,
        candidate_types,
    )
    alias_ambiguity = float(alias_row.get("label_ambiguity", 0.0) or 0.0)
    candidate_ambiguity = float(candidate_row.get("label_ambiguity", 0.0) or 0.0)
    ambiguity_penalty = 0.10 * max(alias_ambiguity, candidate_ambiguity)

    subset_bonus = 0.05 if token_set(alias_label) and (
        alias_tokens.issubset(candidate_tokens)
        or candidate_tokens.issubset(alias_tokens)
    ) else 0.0

    negative_diff = (alias_tokens ^ candidate_tokens) & NEGATIVE_DISTINGUISHER_TOKENS
    negative_penalty = 0.18 if negative_diff else 0.0
    informative_gap_penalty = informative_token_gap_penalty(alias_tokens, candidate_tokens, token_idf)

    freq_delta = max(
        -0.05,
        min(
            0.05,
            (math.log1p(float(candidate_row.get("mention_count", 1))) - math.log1p(float(alias_row.get("mention_count", 1)))) / 12.0,
        ),
    )

    final_score = (
        0.26 * lexical_cosine
        + 0.12 * sequence_ratio
        + 0.08 * token_jaccard
        + 0.18 * weighted_jaccard
        + 0.12 * max(alias_containment, candidate_containment)
        + 0.14 * context_cosine
        + 0.06 * co_mention_score
        + 0.04 * type_score
        + subset_bonus
        + freq_delta
        - negative_penalty
        - informative_gap_penalty
        - ambiguity_penalty
    )

    if rule_reason == "leading_article":
        final_score = max(final_score, 0.96)
    elif rule_reason == "parenthetical_base_match":
        final_score = max(final_score, 0.91)
    elif rule_reason == "acronym_match":
        final_score = max(final_score, 0.90)
    elif rule_reason == "person_title_variant":
        final_score = max(final_score, 0.95)

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
        "weighted_token_jaccard": round(weighted_jaccard, 4),
        "alias_token_containment": round(alias_containment, 4),
        "candidate_token_containment": round(candidate_containment, 4),
        "co_mention_jaccard": round(co_mention_score, 4),
        "type_compatibility_score": round(type_score, 4),
        "type_compatibility_note": type_note,
        "alias_label_ambiguity": round(alias_ambiguity, 4),
        "candidate_label_ambiguity": round(candidate_ambiguity, 4),
        "ambiguity_penalty": round(ambiguity_penalty, 4),
        "negative_distinguishers": ";".join(sorted(negative_diff)),
        "informative_gap_penalty": round(informative_gap_penalty, 4),
        "hard_conflict_reason": hard_conflict_reason,
        "final_score": round(final_score, 4),
    }

