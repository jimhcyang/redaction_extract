#!/usr/bin/env python3
"""
Local CLI version of Matryoshka Notebook C for the CIA corpus.

Consumes Stage B graph outputs and writes the analyst-facing two-level hierarchy:
    hierarchy/community_nodes.csv
    hierarchy/community_edges.csv
    hierarchy/community_to_narrative.csv
    hierarchy/narrative_nodes.csv
    hierarchy/narrative_edges.csv
    hierarchy/hierarchy_index.csv

Stage C exposes a finer hierarchy than Stage B's default recursive summaries:
- communities: subcommunities induced within each Stage B level-1 group
- narratives: the Stage B level-1 groups themselves

This keeps the exposed narrative layer broad and interpretable while giving the
community layer enough granularity to be internally coherent.

Also writes backward-compatible files used by the app:
    graphs/community_summary.csv
    graphs/node_to_community.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from graph_labeling import (
        clean_label_text,
        make_theme_family,
        make_topic_label,
        split_label_list,
    )
except ImportError:
    from knowledge_graph.graph_labeling import (
        clean_label_text,
        make_theme_family,
        make_topic_label,
        split_label_list,
    )
try:
    from entity_resolution.common import OpenAI, append_jsonl, call_openai_json, iter_progress, load_jsonl
except ImportError:
    from knowledge_graph.entity_resolution.common import OpenAI, append_jsonl, call_openai_json, iter_progress, load_jsonl
try:
    from stage_b_graph import META_EDGE_MIN_WEIGHT_BY_LEVEL, build_igraph, build_meta_graph_from_partition, run_leiden, summarize_communities
except ImportError:
    from knowledge_graph.stage_b_graph import META_EDGE_MIN_WEIGHT_BY_LEVEL, build_igraph, build_meta_graph_from_partition, run_leiden, summarize_communities


DEFAULT_HIERARCHY_LABEL_MODEL = "gpt-4o-mini"
COMMUNITY_LABEL_CACHE_FILENAME = "community_label_adjudications.jsonl"
NARRATIVE_LABEL_CACHE_FILENAME = "narrative_label_adjudications.jsonl"
HIERARCHY_LABEL_PROMPT_VERSION = "stage_c_label_v2"
COMMUNITY_DUPLICATE_LABEL_CACHE_FILENAME = "community_duplicate_label_repairs.jsonl"
NARRATIVE_DUPLICATE_LABEL_CACHE_FILENAME = "narrative_duplicate_label_repairs.jsonl"
SUBCOMMUNITY_LEIDEN_RESOLUTION = 3.2
SUBCOMMUNITY_MIN_META_EDGE_WEIGHT = 0.08


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_semicolon_list(value: object) -> List[str]:
    return split_label_list(value)


def make_analyst_summary(label: str, size_text: str, top_labels: object) -> str:
    parts = split_semicolon_list(top_labels)
    if parts:
        return f"This group connects {size_text} around: {', '.join(parts[:5])}."
    return f"This group connects {size_text}."


def _safe_openai_client() -> Optional[OpenAI]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _label_cache_key(kind: str, payload: dict, version: str = HIERARCHY_LABEL_PROMPT_VERSION) -> str:
    raw = json.dumps({"kind": kind, "version": version, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _truncate_labels(values: List[str], limit: int) -> List[str]:
    cleaned = [clean_label_text(value) for value in values if clean_label_text(value)]
    return cleaned[:limit]


def _community_label_payload(row: pd.Series) -> dict:
    return {
        "community_id": int(row.get("community_id", -1)),
        "size": int(row.get("size", 0) or 0),
        "top_labels": _truncate_labels(split_semicolon_list(row.get("top_labels", "")), 8),
        "top_types": clean_label_text(row.get("top_types", "")),
        "heuristic_label": clean_label_text(row.get("label", "")),
        "heuristic_family": clean_label_text(row.get("analyst_label", "")),
    }


def _narrative_label_payload(
    row: pd.Series,
    community_to_narrative: pd.DataFrame,
    community_nodes: pd.DataFrame,
) -> dict:
    narrative_id = int(row.get("narrative_id", -1))
    child_ids = (
        community_to_narrative.loc[community_to_narrative["narrative_id"] == narrative_id, "community_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    child_labels = (
        community_nodes.loc[community_nodes["community_id"].isin(child_ids), "label"]
        .dropna()
        .astype(str)
        .tolist()
    )
    return {
        "narrative_id": narrative_id,
        "num_communities": int(row.get("num_communities", 0) or 0),
        "size": int(row.get("size", 0) or 0),
        "child_community_labels": _truncate_labels(child_labels, 10),
        "top_labels": _truncate_labels(split_semicolon_list(row.get("top_labels", "")), 10),
        "heuristic_label": clean_label_text(row.get("label", "")),
        "heuristic_family": clean_label_text(row.get("analyst_label", "")),
    }


def _hierarchy_label_prompt(kind: str, payload: dict) -> str:
    if kind == "community":
        return f"""
You are naming one community in a historical geopolitical knowledge graph.

Goal:
- produce a short, readable, analyst-facing topic label
- avoid slash-separated lists of entities
- do not simply restate the top labels verbatim
- keep the label specific enough to be meaningful but broad enough to cover the evidence

JSON schema:
{{
  "label": "4-10 word community label",
  "analyst_label": "slightly broader thematic family, 4-10 words",
  "summary": "one sentence, under 25 words"
}}

Evidence:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    return f"""
You are naming one top-level narrative in a historical geopolitical knowledge graph.

Goal:
- produce an umbrella label broader than any single child community
- avoid over-specific labels if the narrative spans several related communities
- avoid slash-separated lists of entities or events
- keep the label readable and general enough to serve as a top-level theme

JSON schema:
{{
  "label": "4-10 word narrative label",
  "analyst_label": "broader thematic family, 4-10 words",
  "summary": "one sentence, under 25 words"
}}

Evidence:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def _sanitize_generated_label(value: object, fallback: str) -> str:
    text = clean_label_text(value)
    if not text:
        return fallback
    text = re.sub(r"\s*/\s*", " ", text)
    text = re.sub(r"[;|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:90] if text else fallback


def apply_openai_hierarchy_labels(
    nodes_df: pd.DataFrame,
    kind: str,
    hier_dir: Path,
    model: str,
    community_to_narrative: Optional[pd.DataFrame] = None,
    community_nodes: Optional[pd.DataFrame] = None,
    run_openai: bool = True,
) -> pd.DataFrame:
    if nodes_df.empty:
        return nodes_df

    if kind not in {"community", "narrative"}:
        raise ValueError(f"Unknown hierarchy label kind: {kind}")

    if kind == "community":
        payload_builder = _community_label_payload
        cache_path = hier_dir / COMMUNITY_LABEL_CACHE_FILENAME
    else:
        if community_to_narrative is None or community_nodes is None:
            raise ValueError("Narrative labeling requires community_to_narrative and community_nodes.")
        payload_builder = lambda row: _narrative_label_payload(row, community_to_narrative, community_nodes)
        cache_path = hier_dir / NARRATIVE_LABEL_CACHE_FILENAME

    out = nodes_df.copy()
    payloads = [payload_builder(row) for _, row in out.iterrows()]
    cache_keys = [_label_cache_key(kind, payload, version=HIERARCHY_LABEL_PROMPT_VERSION) for payload in payloads]
    out["label_cache_key"] = cache_keys

    cached_rows = {str(row.get("cache_key", "")): row for row in load_jsonl(cache_path)}
    pending: List[tuple[int, dict]] = []

    final_labels: List[str] = []
    final_families: List[str] = []
    final_summaries: List[str] = []

    for idx, payload in enumerate(payloads):
        cached = cached_rows.get(cache_keys[idx])
        fallback_label = clean_label_text(out.iloc[idx].get("label", ""))
        fallback_family = clean_label_text(out.iloc[idx].get("analyst_label", fallback_label))
        fallback_summary = clean_label_text(out.iloc[idx].get("analyst_summary", ""))
        if cached:
            final_labels.append(_sanitize_generated_label(cached.get("label", ""), fallback_label))
            final_families.append(_sanitize_generated_label(cached.get("analyst_label", ""), fallback_family))
            final_summaries.append(clean_label_text(cached.get("summary", "")) or fallback_summary)
        else:
            final_labels.append(fallback_label)
            final_families.append(fallback_family)
            final_summaries.append(fallback_summary)
            pending.append((idx, payload))

    client = _safe_openai_client() if run_openai else None
    if pending and client is not None:
        cached_count = len(out) - len(pending)
        total_count = len(out)
        print(f"{kind.title()} label queue:", {"total": total_count, "cached": cached_count, "pending": len(pending)})
        for idx, payload in iter_progress(
            pending,
            total=total_count,
            initial=cached_count,
            desc=f"Stage C {kind} labels",
        ):
            prompt = _hierarchy_label_prompt(kind, payload)
            response = call_openai_json(client, prompt=prompt, model=model)
            record = {
                "cache_key": cache_keys[idx],
                "kind": kind,
                "node_key": clean_label_text(out.iloc[idx].get("node_key", "")),
                "label": clean_label_text(response.get("label", "")),
                "analyst_label": clean_label_text(response.get("analyst_label", "")),
                "summary": clean_label_text(response.get("summary", "")),
            }
            append_jsonl(cache_path, record)
            fallback_label = clean_label_text(out.iloc[idx].get("label", ""))
            fallback_family = clean_label_text(out.iloc[idx].get("analyst_label", fallback_label))
            fallback_summary = clean_label_text(out.iloc[idx].get("analyst_summary", ""))
            final_labels[idx] = _sanitize_generated_label(record["label"], fallback_label)
            final_families[idx] = _sanitize_generated_label(record["analyst_label"], fallback_family)
            final_summaries[idx] = record["summary"] or fallback_summary
    elif pending and run_openai and client is None:
        print(f"Stage C {kind} labeling: no OpenAI client or API key; keeping heuristic labels.")

    out["label"] = final_labels
    out["analyst_label"] = final_families
    out["analyst_summary"] = final_summaries
    out = out.drop(columns=["label_cache_key"], errors="ignore")
    return out


def repair_duplicate_labels(
    nodes_df: pd.DataFrame,
    kind: str,
    hier_dir: Path,
    model: str,
    run_openai: bool = True,
) -> pd.DataFrame:
    if nodes_df.empty:
        return nodes_df

    out = nodes_df.copy()
    cache_name = COMMUNITY_DUPLICATE_LABEL_CACHE_FILENAME if kind == "community" else NARRATIVE_DUPLICATE_LABEL_CACHE_FILENAME
    cache_path = hier_dir / cache_name
    cached_rows = {str(row.get("cache_key", "")): row for row in load_jsonl(cache_path)}
    client = _safe_openai_client() if run_openai else None

    normalized = out["label"].fillna("").astype(str).str.strip().str.lower()
    duplicate_groups = [group.index.tolist() for key, group in out.groupby(normalized) if key and len(group) > 1]
    if not duplicate_groups:
        return out

    for group_indices in duplicate_groups:
        payload = {
            "kind": kind,
            "items": [
                {
                    "row_index": int(idx),
                    "node_key": clean_label_text(out.loc[idx, "node_key"]),
                    "current_label": clean_label_text(out.loc[idx, "label"]),
                    "current_analyst_label": clean_label_text(out.loc[idx, "analyst_label"]),
                    "top_labels": _truncate_labels(split_semicolon_list(out.loc[idx, "top_labels"]), 8),
                    "size": int(out.loc[idx, "size"] if "size" in out.columns else out.loc[idx, "num_communities"]),
                }
                for idx in group_indices
            ],
        }
        cache_key = _label_cache_key(f"{kind}_duplicate_repair", payload, version="stage_c_duplicate_v1")
        cached = cached_rows.get(cache_key)
        if cached is None and client is not None:
            prompt = f"""
You are disambiguating duplicate analyst-facing labels in a geopolitical knowledge graph.

Goal:
- assign distinct short labels to each item
- preserve fidelity to the evidence
- avoid generic repeated names
- keep labels readable and under 10 words

Return JSON:
{{
  "items": [
    {{
      "row_index": 0,
      "label": "distinct label",
      "analyst_label": "slightly broader family label"
    }}
  ]
}}

Evidence:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()
            response = call_openai_json(client, prompt=prompt, model=model)
            cached = {
                "cache_key": cache_key,
                "items": response.get("items", []),
            }
            append_jsonl(cache_path, cached)
        if cached:
            for item in cached.get("items", []):
                try:
                    idx = int(item.get("row_index"))
                except Exception:
                    continue
                if idx not in group_indices:
                    continue
                out.loc[idx, "label"] = _sanitize_generated_label(item.get("label", ""), clean_label_text(out.loc[idx, "label"]))
                out.loc[idx, "analyst_label"] = _sanitize_generated_label(item.get("analyst_label", ""), clean_label_text(out.loc[idx, "analyst_label"]))
        else:
            # Heuristic fallback if OpenAI is unavailable.
            for idx in group_indices:
                top_parts = _truncate_labels(split_semicolon_list(out.loc[idx, "top_labels"]), 2)
                suffix = " / ".join(top_parts) if top_parts else clean_label_text(out.loc[idx, "node_key"])
                out.loc[idx, "label"] = f"{clean_label_text(out.loc[idx, 'label'])} [{suffix}]"[:90]
    return out


def infer_graph_dir(project_dir: Path, graph_dir: Optional[Path] = None) -> Path:
    candidates: List[Path] = []
    if graph_dir is not None:
        candidates.append(Path(graph_dir))
    project_dir = Path(project_dir)
    candidates.append(project_dir / "graphs")
    candidates.append(project_dir)

    required_hint_files = {"canonical_nodes_cluster.csv", "analysis_edges_cluster.csv"}
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            names = {path.name for path in candidate.glob("*.csv")}
            if required_hint_files.issubset(names):
                return candidate

    msg = ["Could not find Notebook B graph folder."]
    msg.append(f"Checked project_dir={project_dir.resolve()}")
    if graph_dir is not None:
        msg.append(f"Checked graph_dir={Path(graph_dir).resolve()}")
    raise FileNotFoundError("\n".join(msg))


def read_csv_first_available(graph_dir: Path, candidates: List[str], required: bool = True) -> pd.DataFrame:
    for name in candidates:
        path = graph_dir / name
        if path.exists():
            print(f"Loaded {name}: {path}")
            return pd.read_csv(path)

    if required:
        available = [path.name for path in sorted(graph_dir.glob("*.csv"))]
        raise FileNotFoundError(
            "Missing required file. Tried:\n  - "
            + "\n  - ".join(candidates)
            + f"\n\nIn graph_dir: {graph_dir.resolve()}\nAvailable CSVs:\n  - "
            + "\n  - ".join(available[:120])
        )
    return pd.DataFrame()


def detect_id_col(df: pd.DataFrame, level: int) -> str:
    expected = f"level{level}_id"
    if expected in df.columns:
        return expected
    candidates = [col for col in df.columns if f"level{level}" in col.lower() and "id" in col.lower()]
    if candidates:
        return candidates[0]
    generic = [col for col in df.columns if col.lower().endswith("_id") or col.lower() == "id"]
    if generic:
        return generic[0]
    raise KeyError(f"Could not detect level{level} id column. Columns: {df.columns.tolist()}")


def load_bv2_outputs(project_dir: Path, graph_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    graph_dir = infer_graph_dir(project_dir, graph_dir)
    data = {
        "graph_dir": graph_dir,
        "nodes_cluster": read_csv_first_available(graph_dir, ["canonical_nodes_cluster.csv"], required=True),
        "edges_cluster": read_csv_first_available(graph_dir, ["analysis_edges_cluster.csv"], required=True),
        "node_to_level1": read_csv_first_available(graph_dir, ["node_to_level1.csv", "node_to_level1_id.csv"], required=True),
        "level1_summary": read_csv_first_available(graph_dir, ["level1_summary.csv", "level1_id_summary.csv"], required=True),
        "level1_meta_edges": read_csv_first_available(graph_dir, ["level1_meta_edges.csv"], required=False),
        "node_to_level2": read_csv_first_available(graph_dir, ["node_to_level2.csv", "node_to_level2_id.csv"], required=False),
        "level2_summary": read_csv_first_available(graph_dir, ["level2_summary.csv", "level2_id_summary.csv"], required=False),
        "level2_meta_edges": read_csv_first_available(graph_dir, ["level2_meta_edges.csv"], required=False),
    }
    print("\nUsing graph_dir:", graph_dir.resolve())
    return data


def _build_summary_nodes(
    summary_df: pd.DataFrame,
    id_col: str,
    node_prefix: str,
    size_summary_text: str,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[id_col, "node_key", "label", "analyst_label", "size", "top_labels", "top_types", "analyst_summary"]
        )

    rows = []
    for _, row in summary_df.iterrows():
        group_id = int(row[id_col])
        top_labels = row.get("top_labels", "")
        fallback = row.get("display_label", f"{node_prefix.title()} {group_id}")
        label = make_topic_label(top_labels, fallback=str(fallback))
        analyst_label = make_theme_family(top_labels, fallback=label)
        size = int(row.get("size", 0) or 0)
        rows.append(
            {
                id_col: group_id,
                "node_key": f"{node_prefix.upper()}:{group_id}",
                "label": label,
                "analyst_label": analyst_label,
                "size": size,
                "top_labels": clean_label_text(top_labels),
                "top_types": row.get("top_types", ""),
                "analyst_summary": make_analyst_summary(label, size_summary_text.format(size=size), top_labels),
            }
        )
    return pd.DataFrame(rows).sort_values(["size", id_col], ascending=[False, True]).reset_index(drop=True)


def build_subcommunity_hierarchy(
    nodes_cluster: pd.DataFrame,
    edges_cluster: pd.DataFrame,
    node_to_level1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if nodes_cluster.empty or node_to_level1.empty:
        empty_nodes = pd.DataFrame(
            columns=["community_id", "node_key", "label", "analyst_label", "size", "top_labels", "top_types", "analyst_summary"]
        )
        empty_edges = pd.DataFrame(columns=["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])
        empty_map = pd.DataFrame(columns=["node_key", "community_id"])
        empty_ctn = pd.DataFrame(columns=["community_id", "community_key", "narrative_id", "narrative_key"])
        return empty_nodes, empty_edges, empty_map, empty_ctn

    level1_col = detect_id_col(node_to_level1, 1)
    graph, node_to_idx, idx_to_node = build_igraph(nodes_cluster, edges_cluster)
    node_row_idx = {node_key: idx for idx, node_key in enumerate(nodes_cluster["node_key"].astype(str).tolist())}
    memberships = [-1] * len(nodes_cluster)
    next_community_id = 0

    grouped_level1 = list(node_to_level1.groupby(level1_col))
    for narrative_id, group in iter_progress(grouped_level1, total=len(grouped_level1), desc="Stage C subcommunities"):
        group_node_keys = [node_key for node_key in group["node_key"].astype(str).tolist() if node_key in node_to_idx]
        if not group_node_keys:
            continue
        group_vertex_ids = [node_to_idx[node_key] for node_key in group_node_keys]
        subgraph = graph.subgraph(group_vertex_ids)
        local_membership = run_leiden(subgraph, resolution=SUBCOMMUNITY_LEIDEN_RESOLUTION, seed=42)
        local_to_global: Dict[int, int] = {}
        for local_vertex_idx, local_group_id in enumerate(local_membership):
            if local_group_id not in local_to_global:
                local_to_global[local_group_id] = next_community_id
                next_community_id += 1
            original_node_key = subgraph.vs[local_vertex_idx]["node_key"]
            memberships[node_row_idx[original_node_key]] = local_to_global[local_group_id]

    for idx, community_id in enumerate(memberships):
        if community_id < 0:
            memberships[idx] = next_community_id
            next_community_id += 1

    node_to_community, community_summary = summarize_communities(
        nodes_cluster,
        edges_cluster,
        memberships,
        level_name="community",
    )
    community_nodes = _build_summary_nodes(
        community_summary,
        id_col="community_id",
        node_prefix="COMMUNITY",
        size_summary_text="{size} lower-level nodes",
    )

    node_to_parent = node_to_level1.rename(columns={level1_col: "narrative_id"}).copy()
    community_to_narrative = node_to_community.merge(node_to_parent, on="node_key", how="left")
    community_to_narrative = (
        community_to_narrative.groupby("community_id", as_index=False)
        .agg(narrative_id=("narrative_id", "first"))
        .sort_values(["narrative_id", "community_id"])
        .reset_index(drop=True)
    )
    community_to_narrative["community_key"] = community_to_narrative["community_id"].apply(lambda value: f"COMMUNITY:{int(value)}")
    community_to_narrative["narrative_key"] = community_to_narrative["narrative_id"].apply(lambda value: f"NARRATIVE:{int(value)}")
    community_to_narrative = community_to_narrative[
        ["community_id", "community_key", "narrative_id", "narrative_key"]
    ]

    _, meta_edges_all = build_meta_graph_from_partition(
        nodes_cluster,
        edges_cluster,
        node_to_community,
        group_col="community_id",
        next_node_prefix="COMMUNITY",
        min_meta_edge_weight=0.0,
    )
    meta_edges_filtered = meta_edges_all[meta_edges_all["weight"] >= SUBCOMMUNITY_MIN_META_EDGE_WEIGHT].copy()
    community_edges = attach_isolated_group_edges(
        meta_edges_all,
        meta_edges_filtered,
        community_nodes["node_key"].astype(str).tolist(),
    )

    return community_nodes, community_edges, node_to_community, community_to_narrative


def build_narrative_layer_from_level1(
    nodes_cluster: pd.DataFrame,
    edges_cluster: pd.DataFrame,
    node_to_level1: pd.DataFrame,
    level1_summary: pd.DataFrame,
    community_to_narrative: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if level1_summary.empty or node_to_level1.empty:
        empty_nodes = pd.DataFrame(
            columns=["narrative_id", "node_key", "label", "analyst_label", "num_communities", "size", "top_labels", "top_types", "analyst_summary"]
        )
        empty_edges = pd.DataFrame(columns=["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])
        return empty_nodes, empty_edges

    level1_col = detect_id_col(level1_summary, 1)
    narrative_nodes = _build_summary_nodes(
        level1_summary,
        id_col=level1_col,
        node_prefix="NARRATIVE",
        size_summary_text="{size} lower-level nodes",
    ).rename(columns={level1_col: "narrative_id"})
    community_counts = community_to_narrative.groupby("narrative_id")["community_id"].nunique().to_dict()
    narrative_nodes["num_communities"] = narrative_nodes["narrative_id"].map(community_counts).fillna(0).astype(int)
    narrative_nodes["analyst_summary"] = narrative_nodes.apply(
        lambda row: make_analyst_summary(
            clean_label_text(row.get("label", "")),
            f"{int(row.get('num_communities', 0) or 0)} communities",
            row.get("top_labels", ""),
        ),
        axis=1,
    )
    narrative_nodes = narrative_nodes[
        ["narrative_id", "node_key", "label", "analyst_label", "num_communities", "size", "top_labels", "top_types", "analyst_summary"]
    ].sort_values(["num_communities", "narrative_id"], ascending=[False, True]).reset_index(drop=True)

    level1_partition = node_to_level1.rename(columns={detect_id_col(node_to_level1, 1): "narrative_id"}).copy()
    _, narrative_edges_all = build_meta_graph_from_partition(
        nodes_cluster,
        edges_cluster,
        level1_partition,
        group_col="narrative_id",
        next_node_prefix="NARRATIVE",
        min_meta_edge_weight=0.0,
    )
    narrative_edges_filtered = narrative_edges_all[
        narrative_edges_all["weight"] >= float(META_EDGE_MIN_WEIGHT_BY_LEVEL.get(1, 0.0))
    ].copy()
    narrative_edges = attach_isolated_group_edges(
        narrative_edges_all,
        narrative_edges_filtered,
        narrative_nodes["node_key"].astype(str).tolist(),
    )
    return narrative_nodes, narrative_edges


def standardize_meta_edges(meta_edges: pd.DataFrame) -> pd.DataFrame:
    if meta_edges.empty:
        return pd.DataFrame(columns=["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])
    out = meta_edges.copy()
    out["source"] = out["source_global"].astype(str)
    out["target"] = out["target_global"].astype(str)
    if "rel" not in out.columns:
        out["rel"] = "META_LINK"
    if "weight" not in out.columns:
        out["weight"] = 1.0
    if "support_count" not in out.columns:
        out["support_count"] = 1
    if "doc_ids" not in out.columns:
        out["doc_ids"] = ""
    return out[["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"]].copy()


def attach_isolated_group_edges(
    meta_edges_all: pd.DataFrame,
    meta_edges_filtered: pd.DataFrame,
    node_keys: List[str],
) -> pd.DataFrame:
    if meta_edges_all.empty:
        return standardize_meta_edges(meta_edges_filtered)

    out = meta_edges_filtered.copy()
    existing_pairs = {
        tuple(sorted((str(row["source_global"]), str(row["target_global"]))))
        for _, row in out.iterrows()
    }
    connected = set(out["source_global"].astype(str)).union(set(out["target_global"].astype(str))) if len(out) else set()
    remaining_isolated = [node for node in node_keys if node not in connected]
    added_rows: List[dict] = []

    while remaining_isolated:
        node = remaining_isolated.pop(0)
        candidates = meta_edges_all[
            (meta_edges_all["source_global"].astype(str) == node) | (meta_edges_all["target_global"].astype(str) == node)
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(["weight", "support_count"], ascending=[False, False])
        chosen = None
        for _, row in candidates.iterrows():
            pair = tuple(sorted((str(row["source_global"]), str(row["target_global"]))))
            if pair in existing_pairs:
                continue
            chosen = row.to_dict()
            existing_pairs.add(pair)
            connected.update(pair)
            break
        if chosen is not None:
            added_rows.append(chosen)
            remaining_isolated = [value for value in remaining_isolated if value not in connected]

    if added_rows:
        out = pd.concat([out, pd.DataFrame(added_rows)], ignore_index=True)
    return standardize_meta_edges(out)


def build_hierarchy_index(
    community_nodes: pd.DataFrame,
    community_to_narrative: pd.DataFrame,
    narrative_nodes: pd.DataFrame,
) -> pd.DataFrame:
    tmp = community_to_narrative.merge(
        community_nodes[["community_id", "label", "size", "top_labels"]],
        on="community_id",
        how="left",
    ).rename(
        columns={
            "label": "community_label",
            "size": "community_size",
            "top_labels": "community_top_labels",
        }
    )

    tmp = tmp.merge(
        narrative_nodes[["narrative_id", "label", "num_communities", "analyst_summary"]],
        on="narrative_id",
        how="left",
    ).rename(columns={"label": "narrative_label", "analyst_summary": "narrative_summary"})

    return tmp.sort_values(["narrative_id", "community_id"]).reset_index(drop=True)


def write_backward_compatibility_files(graph_dir: Path, community_nodes: pd.DataFrame, node_to_community: pd.DataFrame) -> None:
    ensure_dir(graph_dir)

    compat_summary = community_nodes.copy()
    compat_summary["community"] = compat_summary["community_id"]
    compat_summary["display_label"] = compat_summary["label"]
    compat_summary.to_csv(graph_dir / "community_summary.csv", index=False)

    node_table = node_to_community.copy()
    node_table["community"] = node_table["community_id"]
    node_table.to_csv(graph_dir / "node_to_community.csv", index=False)


def write_labeled_summary_files(
    graph_dir: Path,
    level1_summary: pd.DataFrame,
    narrative_nodes: pd.DataFrame,
) -> None:
    if not level1_summary.empty:
        labeled_level1 = level1_summary.copy()
        level1_id_col = detect_id_col(labeled_level1, 1)
        narrative_lookup = narrative_nodes[["narrative_id", "label", "analyst_label"]].rename(
            columns={"narrative_id": level1_id_col, "label": "display_label_labeled", "analyst_label": "analyst_label_labeled"}
        )
        labeled_level1 = labeled_level1.merge(narrative_lookup, on=level1_id_col, how="left")
        if "display_label" in labeled_level1.columns:
            labeled_level1["display_label_raw"] = labeled_level1["display_label"]
        labeled_level1["display_label"] = labeled_level1["display_label_labeled"].fillna(labeled_level1.get("display_label", ""))
        labeled_level1["analyst_label"] = labeled_level1["analyst_label_labeled"].fillna(labeled_level1.get("analyst_label", ""))
        labeled_level1 = labeled_level1.drop(columns=["display_label_labeled", "analyst_label_labeled"], errors="ignore")
        labeled_level1.to_csv(graph_dir / "level1_summary.csv", index=False)


def write_pyvis_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, out_path: Path, max_nodes: Optional[int] = None) -> None:
    try:
        from pyvis.network import Network
    except Exception:
        print("PyVis not installed; skipping:", out_path)
        return

    ensure_dir(out_path.parent)
    if max_nodes is not None:
        nodes_df = nodes_df.head(max_nodes).copy()
    else:
        nodes_df = nodes_df.copy()
    allowed = set(nodes_df["node_key"].astype(str))
    # Use inline assets so each HTML file is self-contained and the pipeline
    # does not create or rely on a repo-root lib/ directory.
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        notebook=False,
        cdn_resources="in_line",
    )

    for _, row in nodes_df.iterrows():
        node_key = str(row["node_key"])
        label = clean_label_text(row.get("label", node_key))
        size = int(min(50, max(10, math.log1p(float(row.get("size", row.get("num_communities", 1)))) * 8)))
        title = (
            f"<b>{label}</b><br>"
            f"Node key: {node_key}<br>"
            f"Size: {row.get('size', '')}<br>"
            f"Summary: {clean_label_text(row.get('analyst_summary', ''))}"
        )
        net.add_node(node_key, label=label[:70], title=title, size=size)

    for _, row in edges_df.iterrows():
        source = str(row.get("source", row.get("source_global", "")))
        target = str(row.get("target", row.get("target_global", "")))
        if source in allowed and target in allowed:
            net.add_edge(source, target, value=float(row.get("weight", 1.0)), title=f"weight={row.get('weight', '')}")

    html = net.generate_html(name=str(out_path), local=True, notebook=False)
    out_path.write_text(html, encoding="utf-8")
    print("Wrote visualization:", out_path)


def print_diagnostics(
    community_nodes: pd.DataFrame,
    community_edges: pd.DataFrame,
    community_to_narrative: pd.DataFrame,
    narrative_nodes: pd.DataFrame,
    narrative_edges: pd.DataFrame,
) -> None:
    print("\n=== Notebook C diagnostics ===")
    print("community_nodes:", community_nodes.shape)
    print("community_edges:", community_edges.shape)
    print("community_to_narrative:", community_to_narrative.shape)
    print("narrative_nodes:", narrative_nodes.shape)
    print("narrative_edges:", narrative_edges.shape)


def run_notebook_c_v2(
    project_dir: Path,
    graph_dir: Optional[Path] = None,
    run_visualization: bool = True,
    run_openai_labels: bool = True,
    label_model: str = DEFAULT_HIERARCHY_LABEL_MODEL,
) -> Dict[str, pd.DataFrame]:
    project_dir = Path(project_dir)
    resolved_graph_dir = infer_graph_dir(project_dir, graph_dir)
    out_project_dir = Path(graph_dir).parent if graph_dir is not None else project_dir

    hier_dir = out_project_dir / "hierarchy"
    vis_dir = hier_dir / "visualizations"
    ensure_dir(hier_dir)

    data = load_bv2_outputs(out_project_dir, resolved_graph_dir)
    community_nodes, community_edges, node_to_community, community_to_narrative = build_subcommunity_hierarchy(
        data["nodes_cluster"],
        data["edges_cluster"],
        data["node_to_level1"],
    )
    community_nodes = apply_openai_hierarchy_labels(
        community_nodes,
        kind="community",
        hier_dir=hier_dir,
        model=label_model,
        run_openai=run_openai_labels,
    )
    community_nodes = repair_duplicate_labels(
        community_nodes,
        kind="community",
        hier_dir=hier_dir,
        model=label_model,
        run_openai=run_openai_labels,
    )
    narrative_nodes, narrative_edges = build_narrative_layer_from_level1(
        data["nodes_cluster"],
        data["edges_cluster"],
        data["node_to_level1"],
        data["level1_summary"],
        community_to_narrative,
    )
    narrative_nodes = apply_openai_hierarchy_labels(
        narrative_nodes,
        kind="narrative",
        hier_dir=hier_dir,
        model=label_model,
        community_to_narrative=community_to_narrative,
        community_nodes=community_nodes,
        run_openai=run_openai_labels,
    )
    narrative_nodes = repair_duplicate_labels(
        narrative_nodes,
        kind="narrative",
        hier_dir=hier_dir,
        model=label_model,
        run_openai=run_openai_labels,
    )
    hierarchy_index = build_hierarchy_index(community_nodes, community_to_narrative, narrative_nodes)

    community_nodes.to_csv(hier_dir / "community_nodes.csv", index=False)
    community_edges.to_csv(hier_dir / "community_edges.csv", index=False)
    community_to_narrative.to_csv(hier_dir / "community_to_narrative.csv", index=False)
    narrative_nodes.to_csv(hier_dir / "narrative_nodes.csv", index=False)
    narrative_edges.to_csv(hier_dir / "narrative_edges.csv", index=False)
    hierarchy_index.to_csv(hier_dir / "hierarchy_index.csv", index=False)

    write_backward_compatibility_files(resolved_graph_dir, community_nodes, node_to_community)
    write_labeled_summary_files(
        resolved_graph_dir,
        data["level1_summary"],
        narrative_nodes,
    )
    print_diagnostics(community_nodes, community_edges, community_to_narrative, narrative_nodes, narrative_edges)

    if run_visualization:
        # The community layer is intended for analyst exploration, so export
        # the full graph rather than truncating to the largest communities.
        write_pyvis_graph(community_nodes, community_edges, vis_dir / "community_graph_pyvis.html", max_nodes=None)
        write_pyvis_graph(narrative_nodes, narrative_edges, vis_dir / "narrative_graph_pyvis.html", max_nodes=250)

    return {
        "community_nodes": community_nodes,
        "community_edges": community_edges,
        "community_to_narrative": community_to_narrative,
        "narrative_nodes": narrative_nodes,
        "narrative_edges": narrative_edges,
        "hierarchy_index": hierarchy_index,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Matryoshka hierarchy outputs from Notebook B graph files.")
    parser.add_argument("PROJECT_DIR", help="Project folder containing graphs/ outputs from Notebook B.")
    parser.add_argument("--graph-dir", default=None, help="Optional direct path to graphs/ directory.")
    parser.add_argument("--no-visualization", action="store_true", help="Skip optional PyVis HTML generation.")
    parser.set_defaults(run_openai_labels=True)
    parser.add_argument("--run-openai-labels", dest="run_openai_labels", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-openai-labels", dest="run_openai_labels", action="store_false", help="Skip Stage C OpenAI hierarchy labeling and keep heuristic labels.")
    parser.add_argument("--openai-label-model", default=DEFAULT_HIERARCHY_LABEL_MODEL, help="OpenAI model for hierarchy node labels when enabled.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_notebook_c_v2(
        project_dir=Path(args.PROJECT_DIR),
        graph_dir=Path(args.graph_dir) if args.graph_dir else None,
        run_visualization=not args.no_visualization,
        run_openai_labels=args.run_openai_labels,
        label_model=args.openai_label_model,
    )


if __name__ == "__main__":
    main()
