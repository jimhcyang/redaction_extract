"""
Stage A2: apply only the highest-precision deterministic entity-resolution rules.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .common import (
    AMBIGUOUS_LABELS_FILENAME,
    ENTITY_RESOLUTION_SUMMARY_FILENAME,
    MENTION_CATALOG_FILENAME,
    RULE_ALIAS_FILENAME,
    RULE_RESOLVED_FILENAME,
    acronym_from_label,
    build_entity_alias_map,
    entity_resolution_dir,
    infer_structural_family,
    iter_progress,
    load_seed_alias_table,
    looks_like_acronym,
    normalize_label_basic,
    person_core_key,
    require_stage_csv,
    resolve_entity_directory,
    safe_str,
    strip_leading_article,
    strip_trailing_parenthetical,
    structural_conflict_reason,
    token_set,
    type_compatibility,
    type_set_from_text,
)


def deterministic_rule_candidates_for_label(
    row: pd.Series,
    catalog_lookup: Dict[str, pd.Series],
    labels: Set[str],
    person_core_lookup: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    normalized_label = safe_str(row["normalized_label"])
    source_types = type_set_from_text(row.get("source_types", ""))
    candidates: Dict[str, List[str]] = defaultdict(list)

    stripped = strip_leading_article(normalized_label)
    if stripped != normalized_label and stripped in labels:
        candidates["leading_article"].append(stripped)

    base = strip_trailing_parenthetical(normalized_label)
    if base != normalized_label and base in labels:
        candidates["parenthetical_base_match"].append(base)

    person_key = person_core_key(normalized_label, source_types)
    if person_key:
        matches = [
            label
            for label in person_core_lookup.get(person_key, [])
            if label != normalized_label
        ]
        if matches:
            candidates["person_title_variant"].extend(matches)

    if looks_like_acronym(normalized_label):
        matches = sorted(
            [
                label
                for label in labels
                if label != normalized_label and acronym_from_label(label) == normalized_label
            ],
            key=lambda label: (
                int(catalog_lookup[label].get("mention_count", 0)),
                int(catalog_lookup[label].get("unique_docs", 0)),
                len(label),
            ),
            reverse=True,
        )
        if matches:
            candidates["acronym_match"].extend(matches)

    return {reason: values for reason, values in candidates.items() if values}


def build_rule_aliases(
    label_catalog: pd.DataFrame,
    seed_alias_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alias_map = build_entity_alias_map(seed_alias_df)
    catalog_lookup = {
        safe_str(row["normalized_label"]): row
        for _, row in label_catalog.iterrows()
    }
    labels = set(catalog_lookup.keys())
    person_core_lookup: Dict[str, List[str]] = defaultdict(list)
    for _, row in iter_progress(label_catalog.iterrows(), total=len(label_catalog), desc="A2 person cores"):
        normalized_label = safe_str(row["normalized_label"])
        source_types = type_set_from_text(row.get("source_types", ""))
        person_key = person_core_key(normalized_label, source_types)
        if person_key:
            person_core_lookup[person_key].append(normalized_label)

    accepted_rows: List[Dict[str, object]] = []
    ambiguous_rows: List[Dict[str, object]] = []
    score_by_reason = {
        "leading_article": 0.96,
        "parenthetical_base_match": 0.91,
        "acronym_match": 0.90,
        "person_title_variant": 0.95,
    }

    for _, row in iter_progress(label_catalog.iterrows(), total=len(label_catalog), desc="A2 rule resolution"):
        normalized_label = safe_str(row["normalized_label"])
        if not normalized_label or normalized_label in alias_map:
            continue

        rule_candidates = deterministic_rule_candidates_for_label(row, catalog_lookup, labels, person_core_lookup)
        if not rule_candidates:
            continue

        for reason, matches in rule_candidates.items():
            unique_matches: List[str] = []
            seen: Set[str] = set()
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    unique_matches.append(match)

            valid_matches: List[str] = []
            for match in unique_matches:
                candidate_row = catalog_lookup.get(match)
                if candidate_row is None:
                    continue
                if should_skip_candidate_direction(normalized_label, match):
                    continue
                if not should_keep_candidate_direction(row, candidate_row):
                    continue
                type_score, _ = type_compatibility(
                    type_set_from_text(row.get("source_types", "")),
                    type_set_from_text(candidate_row.get("source_types", "")),
                )
                conflict = structural_conflict_reason(
                    normalized_label,
                    match,
                    type_set_from_text(row.get("source_types", "")),
                    type_set_from_text(candidate_row.get("source_types", "")),
                )
                if conflict or type_score <= 0.0:
                    continue
                valid_matches.append(match)

            if reason == "acronym_match" and len(valid_matches) != 1:
                if valid_matches:
                    ambiguous_rows.append(
                        {
                            "normalized_label": normalized_label,
                            "rule_reason": reason,
                            "candidate_labels": "; ".join(valid_matches[:8]),
                            "mention_count": int(row.get("mention_count", 0)),
                            "unique_docs": int(row.get("unique_docs", 0)),
                            "label_ambiguity": float(row.get("label_ambiguity", 0.0) or 0.0),
                            "needs_sense_clustering": True,
                            "why": "multiple_acronym_expansions",
                        }
                    )
                continue

            if not valid_matches:
                continue

            best = max(valid_matches, key=lambda label: canonical_preference_key(catalog_lookup[label]))
            accepted_rows.append(
                {
                    "alias": normalized_label,
                    "canonical_label": best,
                    "review_status": "accept_rule",
                    "rule_reason": reason,
                    "final_score": score_by_reason.get(reason, 0.90),
                    "mention_count": int(row.get("mention_count", 0)),
                    "unique_docs": int(row.get("unique_docs", 0)),
                }
            )

    accepted = pd.DataFrame(accepted_rows)
    if len(accepted):
        accepted["alias_key"] = accepted["alias"].map(normalize_label_basic)
        accepted = accepted.sort_values(
            ["alias", "final_score", "mention_count", "unique_docs"],
            ascending=[True, False, False, False],
        ).drop_duplicates(subset=["alias_key"], keep="first").drop(columns=["alias_key"]).reset_index(drop=True)
    else:
        accepted = pd.DataFrame(
            columns=["alias", "canonical_label", "review_status", "rule_reason", "final_score", "mention_count", "unique_docs"]
        )

    ambiguous = pd.DataFrame(ambiguous_rows)
    if len(ambiguous) == 0:
        ambiguous = pd.DataFrame(
            columns=[
                "normalized_label",
                "rule_reason",
                "candidate_labels",
                "mention_count",
                "unique_docs",
                "label_ambiguity",
                "needs_sense_clustering",
                "why",
            ]
        )
    return accepted, ambiguous


def apply_rule_resolution_to_mentions(
    mention_catalog: pd.DataFrame,
    seed_alias_df: pd.DataFrame,
    rule_aliases: pd.DataFrame,
) -> pd.DataFrame:
    seed_map = build_entity_alias_map(seed_alias_df)
    rule_map = build_entity_alias_map(rule_aliases) if len(rule_aliases) else {}
    merged_map = {**seed_map, **rule_map}
    method_map = {
        normalize_label_basic(row["alias"]): safe_str(row.get("review_status", "accept_rule"))
        for _, row in rule_aliases.iterrows()
    } if len(rule_aliases) else {}

    out = mention_catalog.copy()
    out["rule_canonical_label"] = out["normalized_label"].map(merged_map).fillna(out["normalized_label"])
    out["rule_resolution_method"] = out["normalized_label"].map(method_map).fillna("self")
    return out


def identify_ambiguous_labels(
    label_catalog: pd.DataFrame,
    rule_aliases: pd.DataFrame,
    rule_ambiguities: pd.DataFrame,
    ambiguity_threshold: float,
) -> pd.DataFrame:
    resolved_aliases = set(rule_aliases["alias"].astype(str)) if len(rule_aliases) else set()
    ambiguous_rules = (
        rule_ambiguities.groupby("normalized_label")["candidate_labels"].first().to_dict()
        if len(rule_ambiguities)
        else {}
    )
    rows: List[Dict[str, object]] = []
    for _, row in iter_progress(label_catalog.iterrows(), total=len(label_catalog), desc="A2 ambiguity scan"):
        normalized_label = safe_str(row["normalized_label"])
        if normalized_label in resolved_aliases:
            continue
        source_types = type_set_from_text(row.get("source_types", ""))
        family = infer_structural_family(normalized_label, source_types)
        ambiguity = float(row.get("label_ambiguity", 0.0) or 0.0)
        mention_count = int(row.get("mention_count", 0))
        looks_acronym = looks_like_acronym(normalized_label)
        generic_family = family in {"government", "organization", "role", "other", "political_group", "military"}
        needs_sense = (
            normalized_label in ambiguous_rules
            or looks_acronym
            or ambiguity >= ambiguity_threshold
            or (generic_family and mention_count >= 2)
        )
        rows.append(
            {
                "normalized_label": normalized_label,
                "mention_count": mention_count,
                "unique_docs": int(row.get("unique_docs", 0)),
                "source_types": safe_str(row.get("source_types", "")),
                "primary_type": safe_str(row.get("primary_type", "")),
                "label_ambiguity": ambiguity,
                "looks_like_acronym": looks_acronym,
                "family": family,
                "rule_candidate_labels": ambiguous_rules.get(normalized_label, ""),
                "needs_sense_clustering": bool(needs_sense),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["needs_sense_clustering", "label_ambiguity", "mention_count", "normalized_label"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def should_skip_candidate_direction(alias_label: str, candidate_label: str) -> bool:
    if not alias_label or not candidate_label or alias_label == candidate_label:
        return False

    if strip_leading_article(candidate_label) == alias_label and candidate_label != alias_label:
        return True

    if strip_trailing_parenthetical(candidate_label) == alias_label and candidate_label != alias_label:
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


def run_stage_a2_rule_resolution(
    project_dir: Path,
    entity_dir: Optional[Path] = None,
    ambiguity_threshold: float = 0.45,
) -> Dict[str, pd.DataFrame]:
    project_dir = Path(project_dir).expanduser().resolve()
    entity_dir = resolve_entity_directory(project_dir, entity_dir)
    out_dir = entity_resolution_dir(project_dir)

    alias_df = load_seed_alias_table(project_dir, entity_dir)
    mention_catalog = require_stage_csv(project_dir, MENTION_CATALOG_FILENAME)
    label_catalog = require_stage_csv(project_dir, "label_catalog.csv")

    rule_aliases, rule_ambiguities = build_rule_aliases(label_catalog, alias_df)
    rule_resolved_mentions = apply_rule_resolution_to_mentions(mention_catalog, alias_df, rule_aliases)
    ambiguous_labels = identify_ambiguous_labels(
        label_catalog=label_catalog,
        rule_aliases=rule_aliases,
        rule_ambiguities=rule_ambiguities,
        ambiguity_threshold=ambiguity_threshold,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rule_aliases.to_csv(out_dir / RULE_ALIAS_FILENAME, index=False)
    rule_resolved_mentions.to_csv(out_dir / RULE_RESOLVED_FILENAME, index=False)
    ambiguous_labels.to_csv(out_dir / AMBIGUOUS_LABELS_FILENAME, index=False)
    rule_aliases.to_csv(out_dir / "alias_auto_accepted.csv", index=False)

    print("A2 rule aliases:", len(rule_aliases))
    print("A2 ambiguous labels:", int(ambiguous_labels["needs_sense_clustering"].sum()) if len(ambiguous_labels) else 0)
    return {
        "alias_df": alias_df,
        "rule_aliases": rule_aliases,
        "rule_resolved_mentions": rule_resolved_mentions,
        "ambiguous_labels": ambiguous_labels,
    }
