"""
Stage A4: resolve sense clusters into canonical entities with optional LLM adjudication.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import (
    AMBIGUOUS_LABELS_FILENAME,
    CLUSTER_ADJUDICATIONS_FILENAME,
    CLUSTER_CANDIDATES_FILENAME,
    CLUSTER_DECISIONS_FILENAME,
    CLUSTER_REVIEW_QUEUE_FILENAME,
    ENTITY_ALIAS_MAP_FOR_B_FILENAME,
    ENTITY_CANONICAL_MAP_FILENAME,
    ENTITY_INVENTORY_FILENAME,
    ENTITY_RESOLUTION_LLM_FILENAME,
    ENTITIES_RESOLVED_FILENAME,
    GENERATED_ALIAS_FILENAME,
    MENTION_CATALOG_FILENAME,
    OPENAI_ACCEPT_CONFIDENCE_THRESHOLD,
    RULE_ALIAS_FILENAME,
    RULE_RESOLVED_FILENAME,
    SENSE_ASSIGNMENTS_FILENAME,
    SENSE_CLUSTERS_FILENAME,
    OpenAI,
    append_jsonl,
    call_openai_json,
    entity_resolution_dir,
    first_existing_col,
    keyify,
    load_entities,
    load_jsonl,
    norm_space,
    normalize_label_basic,
    require_stage_csv,
    safe_str,
    tqdm,
    write_stage_summary,
    _sample_text_list,
)
from .a3_sense_clustering import build_cluster_review_queue, cluster_adjudication_key


def cluster_openai_prompt(row: pd.Series) -> str:
    return f"""
You are resolving one cluster of entity mentions extracted from historical intelligence documents.

Decide whether this cluster should:
1. merge to one existing candidate entity,
2. become a new inferred canonical entity name, or
3. remain separate because the evidence is insufficient.

Be conservative. Do not force a merge when the evidence is mixed.

Return only valid JSON:
{{
  "decision": "merge_existing|new_inferred_name|keep_separate",
  "canonical_label": "canonical name if merge_existing or new_inferred_name, else empty",
  "selected_candidate_label": "existing candidate label if merge_existing, else empty",
  "confidence": 0.0,
  "reason": "short explanation"
}}

Cluster to resolve:
- sense_id: {safe_str(row.get("sense_id", ""))}
- raw label: {safe_str(row.get("normalized_label", ""))}
- descriptor: {safe_str(row.get("descriptor", ""))}
- mention_count: {safe_str(row.get("mention_count", ""))}
- unique_docs: {safe_str(row.get("unique_docs", ""))}
- source_types: {safe_str(row.get("source_types", ""))}
- example_mentions: {safe_str(row.get("example_mentions", ""))}
- example_descriptions: {safe_str(row.get("example_descriptions", ""))}
- example_contexts: {safe_str(row.get("example_contexts", ""))}
- top_co_mentions: {safe_str(row.get("top_co_mentions", ""))}
- sibling_clusters_for_same_label: {safe_str(row.get("sibling_clusters", ""))}

Existing candidate entities:
{safe_str(row.get("candidate_labels", ""))}
""".strip()


def run_cluster_openai_adjudication(
    queue: pd.DataFrame,
    out_dir: Path,
    model: str,
    max_clusters: Optional[int],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(queue) == 0:
        return pd.DataFrame(
            columns=[
                "adjudication_key",
                "sense_id",
                "normalized_label",
                "decision",
                "canonical_label",
                "selected_candidate_label",
                "confidence",
                "reason",
            ]
        )

    if OpenAI is None:
        raise ImportError("OpenAI package is not installed. Run: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    path = out_dir / CLUSTER_ADJUDICATIONS_FILENAME
    error_path = out_dir / "a4_cluster_adjudication_errors.jsonl"
    existing_rows = load_jsonl(path)
    existing_map = {
        safe_str(row.get("adjudication_key", "")): row
        for row in existing_rows
        if safe_str(row.get("adjudication_key", ""))
    }

    queue = queue.copy()
    if max_clusters is not None:
        queue = queue.head(max_clusters)
    pending_queue = queue[~queue["adjudication_key"].isin(existing_map)].copy()
    cached_count = int(len(queue) - len(pending_queue))
    print(
        "A4 OpenAI cluster adjudication queue:",
        {
            "total": int(len(queue)),
            "cached": cached_count,
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
            total=len(queue),
            initial=cached_count,
            desc="A4 cluster adjudications",
            unit="cluster",
        )

    for _, row in iterator:
        key = safe_str(row["adjudication_key"])
        prompt = cluster_openai_prompt(row)
        try:
            response = call_openai_json(client, prompt=prompt, model=model)
        except Exception as exc:
            error_record = {
                "adjudication_key": key,
                "sense_id": safe_str(row.get("sense_id", "")),
                "normalized_label": safe_str(row.get("normalized_label", "")),
                "model": model,
                "error_type": type(exc).__name__,
                "error": safe_str(exc),
            }
            append_jsonl(error_path, error_record)
            print(f"Skipping A4 adjudication for {row.get('sense_id', '')}: {type(exc).__name__}: {exc}")
            continue

        record = {
            "adjudication_key": key,
            "sense_id": safe_str(row.get("sense_id", "")),
            "normalized_label": safe_str(row.get("normalized_label", "")),
            "model": model,
            "decision": safe_str(response.get("decision", "")),
            "canonical_label": safe_str(response.get("canonical_label", "")),
            "selected_candidate_label": safe_str(response.get("selected_candidate_label", "")),
            "confidence": float(response.get("confidence", 0.0) or 0.0),
            "reason": safe_str(response.get("reason", "")),
        }
        append_jsonl(path, record)
        existing_map[key] = record

    return pd.DataFrame(existing_map.values())


def _fallback_canonical_label_for_cluster(row: pd.Series) -> str:
    normalized_label = safe_str(row.get("normalized_label", ""))
    descriptor = safe_str(row.get("descriptor", ""))
    cluster_index = int(row.get("sense_cluster_index", 0)) + 1
    label_cluster_count = int(row.get("label_cluster_count", 1))
    if descriptor:
        return f"{normalized_label} ({descriptor})"
    if label_cluster_count > 1:
        return f"{normalized_label} [SENSE {cluster_index}]"
    return normalized_label


def finalize_cluster_decisions(
    clusters_df: pd.DataFrame,
    adjudications: pd.DataFrame,
) -> pd.DataFrame:
    if len(clusters_df) == 0:
        return clusters_df.copy()

    out = clusters_df.copy()
    if "adjudication_key" not in out.columns:
        out["adjudication_key"] = out.apply(cluster_adjudication_key, axis=1)
    if len(adjudications):
        adjudications = adjudications.rename(
            columns={
                "decision": "openai_decision",
                "canonical_label": "openai_canonical_label",
                "selected_candidate_label": "openai_selected_candidate_label",
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
                    "openai_selected_candidate_label",
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
        out["openai_selected_candidate_label"] = ""
        out["openai_confidence"] = np.nan
        out["openai_reason"] = ""

    final_labels: List[str] = []
    final_methods: List[str] = []
    final_confidences: List[float] = []
    for _, row in out.iterrows():
        decision = safe_str(row.get("openai_decision", ""))
        confidence = float(row.get("openai_confidence", 0.0) or 0.0)
        canonical_label = norm_space(row.get("openai_canonical_label", ""))
        selected_candidate = norm_space(row.get("openai_selected_candidate_label", ""))
        auto_method = safe_str(row.get("auto_resolution_method", ""))
        auto_label = norm_space(row.get("auto_resolution_label", ""))
        auto_confidence = pd.to_numeric(row.get("auto_resolution_confidence", np.nan), errors="coerce")

        if auto_method and auto_label:
            final_labels.append(auto_label)
            final_methods.append(auto_method)
            final_confidences.append(float(auto_confidence) if not pd.isna(auto_confidence) else 1.0)
        elif decision == "merge_existing" and confidence >= OPENAI_ACCEPT_CONFIDENCE_THRESHOLD:
            final_labels.append(canonical_label or selected_candidate or _fallback_canonical_label_for_cluster(row))
            final_methods.append("a4_merge_existing")
            final_confidences.append(confidence)
        elif decision == "new_inferred_name" and confidence >= 0.78 and canonical_label:
            final_labels.append(canonical_label)
            final_methods.append("a4_new_inferred_name")
            final_confidences.append(confidence)
        elif bool(row.get("needs_llm", False)):
            final_labels.append(_fallback_canonical_label_for_cluster(row))
            final_methods.append("a4_keep_separate")
            final_confidences.append(confidence if confidence > 0 else 1.0)
        else:
            final_labels.append(_fallback_canonical_label_for_cluster(row))
            final_methods.append("a3_single_cluster_self")
            final_confidences.append(1.0)

    out["final_canonical_label"] = final_labels
    out["final_method"] = final_methods
    out["final_confidence"] = final_confidences
    out["final_canonical_key"] = out["final_canonical_label"].map(lambda value: f"CANONICAL_ENTITY:{keyify(value)}")
    return out


def build_entities_resolved_from_clusters(
    entities: pd.DataFrame,
    mention_catalog: pd.DataFrame,
    rule_resolved_mentions: pd.DataFrame,
    sense_assignments: pd.DataFrame,
    cluster_decisions: pd.DataFrame,
) -> pd.DataFrame:
    label_col = first_existing_col(entities, ["label", "name", "entity", "text"])
    if label_col is None:
        raise KeyError(f"Could not find entity label column in entities. Columns: {entities.columns.tolist()}")

    entities = entities.copy()
    if "local_id" not in entities.columns:
        if "id" in entities.columns:
            entities["local_id"] = entities["id"]
        else:
            entities["local_id"] = [f"e{idx}" for idx in range(len(entities))]
    entities["mention_key"] = (
        entities["doc_id"].astype(str)
        + "::"
        + entities["paragraph_id"].astype(str)
        + "::"
        + entities["local_id"].astype(str)
    )
    entities["normalized_label"] = entities[label_col].map(normalize_label_basic)

    rule_view = rule_resolved_mentions.loc[
        :,
        ["mention_key", "rule_canonical_label", "rule_resolution_method"],
    ].drop_duplicates(subset=["mention_key"])
    out = entities.merge(rule_view, on="mention_key", how="left")

    if len(sense_assignments):
        cluster_view = sense_assignments.merge(
            cluster_decisions.loc[
                :,
                ["sense_id", "final_canonical_label", "final_canonical_key", "final_method", "final_confidence"],
            ],
            on="sense_id",
            how="left",
        )
        out = out.merge(cluster_view, on=["mention_key", "normalized_label"], how="left")
    else:
        out["sense_id"] = ""
        out["final_canonical_label"] = ""
        out["final_canonical_key"] = ""
        out["final_method"] = ""
        out["final_confidence"] = np.nan

    out["canonical_label"] = np.where(
        out["rule_resolution_method"].fillna("self").ne("self"),
        out["rule_canonical_label"],
        out["final_canonical_label"].fillna(""),
    )
    out["canonical_label"] = out["canonical_label"].fillna("").replace("", np.nan).fillna(out["normalized_label"])
    out["canonical_key"] = np.where(
        out["rule_resolution_method"].fillna("self").ne("self"),
        out["rule_canonical_label"].map(lambda value: f"CANONICAL_ENTITY:{keyify(value)}"),
        out["final_canonical_key"].fillna(""),
    )
    out["canonical_key"] = out["canonical_key"].fillna("").replace("", np.nan).fillna(
        out["canonical_label"].map(lambda value: f"CANONICAL_ENTITY:{keyify(value)}")
    )
    out["canonical_method"] = np.where(
        out["rule_resolution_method"].fillna("self").ne("self"),
        out["rule_resolution_method"],
        out["final_method"].fillna(""),
    )
    out["canonical_method"] = out["canonical_method"].replace("", np.nan).fillna("self")
    out["canonical_confidence"] = np.where(
        out["rule_resolution_method"].fillna("self").ne("self"),
        1.0,
        pd.to_numeric(out["final_confidence"], errors="coerce"),
    )
    out["canonical_confidence"] = pd.to_numeric(out["canonical_confidence"], errors="coerce").fillna(1.0)
    return out


def build_canonical_map_from_resolved_entities(
    entities_resolved: pd.DataFrame,
) -> pd.DataFrame:
    if len(entities_resolved) == 0:
        return pd.DataFrame(
            columns=[
                "original_normalized_label",
                "canonical_label",
                "canonical_key",
                "method",
                "confidence",
                "mention_count",
                "doc_count",
                "entity_type",
                "surface_forms",
                "sample_context",
            ]
        )

    label_col = first_existing_col(entities_resolved, ["label", "name", "entity", "text"])
    type_col = first_existing_col(entities_resolved, ["type", "entity_type", "broad_type"])
    desc_col = first_existing_col(entities_resolved, ["description", "summary", "evidence"])

    rows: List[Dict[str, object]] = []
    for normalized_label, group in entities_resolved.groupby("normalized_label"):
        canonical_counts = group["canonical_label"].astype(str).value_counts()
        if len(canonical_counts) == 1:
            canonical_label = canonical_counts.index[0]
            method = _sample_text_list(group["canonical_method"].tolist(), limit=1)[0] if len(group) else "self"
            confidence = float(pd.to_numeric(group["canonical_confidence"], errors="coerce").fillna(1.0).max())
        else:
            canonical_label = normalized_label
            method = "multi_sense_label"
            confidence = 1.0

        surface_forms = "; ".join(_sample_text_list(group[label_col].tolist(), limit=10)) if label_col else ""
        sample_context = " || ".join(_sample_text_list(group[desc_col].tolist(), limit=3)) if desc_col else ""
        rows.append(
            {
                "original_normalized_label": normalized_label,
                "canonical_label": canonical_label,
                "canonical_key": f"CANONICAL_ENTITY:{keyify(canonical_label)}",
                "method": method,
                "confidence": confidence,
                "mention_count": int(len(group)),
                "doc_count": int(group["doc_id"].astype(str).nunique()),
                "entity_type": _sample_text_list(group[type_col].tolist(), limit=1)[0] if type_col else "",
                "surface_forms": surface_forms,
                "sample_context": sample_context,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["mention_count", "doc_count", "original_normalized_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_generated_alias_table_from_canonical_map(
    canonical_map: pd.DataFrame,
) -> pd.DataFrame:
    if len(canonical_map) == 0:
        return pd.DataFrame(
            columns=["alias", "canonical_label", "notes", "source", "review_status", "final_score"]
        )

    out = canonical_map.copy()
    out = out[
        out["original_normalized_label"].astype(str) != out["canonical_label"].astype(str)
    ].copy()
    out = out[~out["method"].astype(str).eq("multi_sense_label")].copy()
    if len(out) == 0:
        return pd.DataFrame(
            columns=["alias", "canonical_label", "notes", "source", "review_status", "final_score"]
        )

    out = out.rename(
        columns={
            "original_normalized_label": "alias",
            "confidence": "final_score",
        }
    )
    out["notes"] = ""
    out["source"] = "a4_pipeline"
    out["review_status"] = out["method"]
    keep_cols = ["alias", "canonical_label", "notes", "source", "review_status", "final_score"]
    return out.loc[:, keep_cols].reset_index(drop=True)


def build_cluster_adjudicated_table(
    adjudications: pd.DataFrame,
    cluster_review_queue: pd.DataFrame,
    cluster_decisions: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "sense_id",
        "normalized_label",
        "decision",
        "confidence",
        "canonical_label",
        "selected_candidate_label",
        "reason",
        "final_canonical_label",
        "final_method",
    ]
    if len(cluster_review_queue) == 0 and len(adjudications) == 0:
        return pd.DataFrame(columns=columns)

    out = cluster_review_queue.copy()
    if len(adjudications):
        out = out.merge(adjudications, on=["adjudication_key", "sense_id", "normalized_label"], how="left")
    if len(cluster_decisions):
        out = out.merge(
            cluster_decisions.loc[:, ["sense_id", "final_canonical_label", "final_method"]],
            on="sense_id",
            how="left",
        )

    rename_map = {
        "openai_confidence": "confidence",
        "openai_reason": "reason",
    }
    out = out.rename(columns=rename_map)
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan if col == "confidence" else ""
    return out.loc[:, columns].reset_index(drop=True)


def run_stage_a4_llm_resolution(
    project_dir: Path,
    run_openai: bool = True,
    model: str = "gpt-4o-mini",
    max_clusters: Optional[int] = None,
    openai_min_mentions: int = 1,
) -> Dict[str, pd.DataFrame]:
    project_dir = Path(project_dir).expanduser().resolve()
    out_dir = entity_resolution_dir(project_dir)

    entities = load_entities(project_dir)
    mention_catalog = require_stage_csv(project_dir, MENTION_CATALOG_FILENAME)
    label_catalog = require_stage_csv(project_dir, "label_catalog.csv")
    entity_inventory = require_stage_csv(project_dir, ENTITY_INVENTORY_FILENAME)
    rule_aliases = require_stage_csv(project_dir, RULE_ALIAS_FILENAME)
    rule_resolved_mentions = require_stage_csv(project_dir, RULE_RESOLVED_FILENAME)
    ambiguous_labels = require_stage_csv(project_dir, AMBIGUOUS_LABELS_FILENAME)
    sense_assignments = require_stage_csv(project_dir, SENSE_ASSIGNMENTS_FILENAME)
    sense_clusters = require_stage_csv(project_dir, SENSE_CLUSTERS_FILENAME)
    cluster_candidates = require_stage_csv(project_dir, CLUSTER_CANDIDATES_FILENAME)

    cluster_review_queue = build_cluster_review_queue(sense_clusters, cluster_candidates)
    if len(cluster_review_queue) and openai_min_mentions > 1:
        cluster_review_queue = cluster_review_queue[
            (cluster_review_queue["mention_count"] >= openai_min_mentions)
            | (cluster_review_queue["label_cluster_count"] > 1)
        ].reset_index(drop=True)
    cluster_review_queue.to_csv(out_dir / CLUSTER_REVIEW_QUEUE_FILENAME, index=False)
    print("A4 review queue rows:", len(cluster_review_queue))

    adjudications = pd.DataFrame()
    if run_openai and len(cluster_review_queue):
        adjudications = run_cluster_openai_adjudication(
            queue=cluster_review_queue,
            out_dir=out_dir,
            model=model,
            max_clusters=max_clusters,
        )

    clusters_for_decision = sense_clusters.merge(
        cluster_review_queue.loc[:, [col for col in ["sense_id", "candidate_labels", "sibling_clusters", "adjudication_key"] if col in cluster_review_queue.columns]],
        on="sense_id",
        how="left",
    )
    cluster_decisions = finalize_cluster_decisions(clusters_for_decision, adjudications)
    entities_resolved = build_entities_resolved_from_clusters(
        entities=entities,
        mention_catalog=mention_catalog,
        rule_resolved_mentions=rule_resolved_mentions,
        sense_assignments=sense_assignments,
        cluster_decisions=cluster_decisions,
    )
    canonical_map = build_canonical_map_from_resolved_entities(entities_resolved)
    generated_aliases = build_generated_alias_table_from_canonical_map(canonical_map)
    alias_for_b = canonical_map.loc[
        :,
        ["original_normalized_label", "canonical_label", "canonical_key", "method", "confidence"],
    ].copy()
    llm_adjudicated = build_cluster_adjudicated_table(adjudications, cluster_review_queue, cluster_decisions)

    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_decisions.to_csv(out_dir / CLUSTER_DECISIONS_FILENAME, index=False)
    entities_resolved.to_csv(out_dir / ENTITIES_RESOLVED_FILENAME, index=False)
    canonical_map.to_csv(out_dir / ENTITY_CANONICAL_MAP_FILENAME, index=False)
    alias_for_b.to_csv(out_dir / ENTITY_ALIAS_MAP_FOR_B_FILENAME, index=False)
    generated_aliases.to_csv(out_dir / GENERATED_ALIAS_FILENAME, index=False)
    generated_aliases.to_csv(out_dir / "entity_aliases_proposed.csv", index=False)
    llm_adjudicated.to_csv(out_dir / ENTITY_RESOLUTION_LLM_FILENAME, index=False)

    summary = {
        "mention_catalog_rows": int(len(mention_catalog)),
        "label_catalog_rows": int(len(label_catalog)),
        "entity_inventory_rows": int(len(entity_inventory)),
        "rule_alias_rows": int(len(rule_aliases)),
        "ambiguous_label_rows": int(int(ambiguous_labels["needs_sense_clustering"].sum()) if len(ambiguous_labels) else 0),
        "sense_assignment_rows": int(len(sense_assignments)),
        "sense_cluster_rows": int(len(sense_clusters)),
        "cluster_candidate_rows": int(len(cluster_candidates)),
        "cluster_review_queue_rows": int(len(cluster_review_queue)),
        "cluster_adjudication_rows": int(len(adjudications)),
        "resolved_entity_rows": int(len(entities_resolved)),
        "canonical_map_rows": int(len(canonical_map)),
        "generated_alias_rows": int(len(generated_aliases)),
        "a3_cluster_auto_merge_rows": int((cluster_decisions["final_method"] == "a3_cluster_auto_merge").sum()) if len(cluster_decisions) else 0,
        "a4_merge_existing_rows": int((cluster_decisions["final_method"] == "a4_merge_existing").sum()) if len(cluster_decisions) else 0,
        "a4_new_inferred_name_rows": int((cluster_decisions["final_method"] == "a4_new_inferred_name").sum()) if len(cluster_decisions) else 0,
        "a4_keep_separate_rows": int((cluster_decisions["final_method"] == "a4_keep_separate").sum()) if len(cluster_decisions) else 0,
        "a3_single_cluster_self_rows": int((cluster_decisions["final_method"] == "a3_single_cluster_self").sum()) if len(cluster_decisions) else 0,
    }
    write_stage_summary(project_dir, summary)
    print(json.dumps(summary, indent=2))
    return {
        "cluster_review_queue": cluster_review_queue,
        "adjudications": adjudications,
        "cluster_decisions": cluster_decisions,
        "entities_resolved": entities_resolved,
        "canonical_map": canonical_map,
        "generated_aliases": generated_aliases,
        "summary": summary,
    }
