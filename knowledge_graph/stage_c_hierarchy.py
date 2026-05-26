#!/usr/bin/env python3
"""
Local CLI version of Matryoshka Notebook C for the CIA corpus.

Consumes Notebook B graph outputs and writes the analyst-facing hierarchy tables:
    hierarchy/community_nodes.csv
    hierarchy/community_edges.csv
    hierarchy/community_to_narrative.csv
    hierarchy/narrative_nodes.csv
    hierarchy/narrative_edges.csv
    hierarchy/hierarchy_index.csv

Also writes backward-compatible files used by the app:
    graphs/community_summary.csv
    graphs/node_to_community.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_label_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)


def split_semicolon_list(value: object) -> List[str]:
    if pd.isna(value):
        return []
    return [clean_label_text(part) for part in str(value).split(";") if str(part).strip()]


def short_label_from_top_labels(top_labels: object, fallback: object, max_items: int = 4) -> str:
    parts = split_semicolon_list(top_labels)
    if parts:
        return "; ".join(parts[:max_items])
    return clean_label_text(fallback)


def make_analyst_summary(label: str, size_text: str, top_labels: object) -> str:
    parts = split_semicolon_list(top_labels)
    if parts:
        return f"This group connects {size_text} around: {'; '.join(parts[:8])}."
    return f"This group connects {size_text}."


def make_analyst_label(top_labels: object, fallback: str = "") -> str:
    text = str(top_labels).upper()
    fallback = str(fallback).strip()

    if any(token in text for token in ["SOVIET", "USSR", "MOSCOW", "KREMLIN"]):
        return "Soviet policy, leadership, and bloc maneuvering"
    if any(token in text for token in ["CHINA", "PEIPING", "PEKING", "COMMUNIST CHINA"]):
        return "Chinese policy and regional positioning"
    if any(token in text for token in ["LAOS", "VIETNAM", "CONGO", "BERLIN", "CUBA"]):
        return "Cold War crisis monitoring and escalation"
    if any(token in text for token in ["UNITED STATES", "STATE DEPARTMENT", "EMBASSY", "CIA"]):
        return "U.S. reporting, diplomacy, and intelligence assessment"
    if any(token in text for token in ["STRIKE", "LABOR", "UNION", "WORKERS"]):
        return "Labor unrest and internal political pressure"
    if any(token in text for token in ["ARMY", "MILITARY", "SECURITY", "DEFENSE", "GENDARMERIE"]):
        return "Military posture, security forces, and regime stability"
    if any(token in text for token in ["UNITED NATIONS", "UNGA", "NATO", "CENTO", "SEATO"]):
        return "International diplomatic and security coordination"
    if any(token in text for token in ["ELECTION", "PARLIAMENT", "CABINET", "PRESIDENT", "PREMIER"]):
        return "Government leadership, cabinet politics, and state decision-making"
    if any(token in text for token in ["OIL", "PETROLEUM", "REFINERY", "PIPELINE"]):
        return "Oil, energy infrastructure, and economic leverage"

    if fallback:
        return fallback[:90]
    return str(top_labels).split(";")[0][:90]


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


def parse_level_key(node_key: object, level: int) -> Optional[int]:
    if pd.isna(node_key):
        return None
    match = re.search(rf"LEVEL{level}_COMMUNITY:(-?\d+)", str(node_key))
    return int(match.group(1)) if match else None


def convert_level_key(node_key: object, level: int, prefix: str) -> str:
    group_id = parse_level_key(node_key, level)
    if group_id is None:
        return str(node_key)
    return f"{prefix}:{group_id}"


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


def build_community_nodes(level1_summary: pd.DataFrame) -> pd.DataFrame:
    if level1_summary.empty:
        return pd.DataFrame(columns=["community_id", "node_key", "label", "analyst_label", "size", "top_labels", "top_types", "analyst_summary"])

    id_col = detect_id_col(level1_summary, 1)
    rows = []
    for _, row in level1_summary.iterrows():
        community_id = int(row[id_col])
        top_labels = row.get("top_labels", "")
        fallback = row.get("display_label", f"Community {community_id}")
        label = short_label_from_top_labels(top_labels, fallback, max_items=4)
        analyst_label = make_analyst_label(top_labels, fallback=label)
        size = int(row.get("size", 0))
        rows.append(
            {
                "community_id": community_id,
                "node_key": f"COMMUNITY:{community_id}",
                "label": label,
                "analyst_label": analyst_label,
                "size": size,
                "top_labels": clean_label_text(top_labels),
                "top_types": row.get("top_types", ""),
                "analyst_summary": make_analyst_summary(label, f"{size} lower-level nodes", top_labels),
            }
        )
    return pd.DataFrame(rows).sort_values(["size", "community_id"], ascending=[False, True]).reset_index(drop=True)


def build_community_edges(level1_meta_edges: pd.DataFrame) -> pd.DataFrame:
    if level1_meta_edges.empty:
        return pd.DataFrame(columns=["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    out = level1_meta_edges.copy()
    out["source"] = out["source_global"].apply(lambda value: convert_level_key(value, 1, "COMMUNITY"))
    out["target"] = out["target_global"].apply(lambda value: convert_level_key(value, 1, "COMMUNITY"))
    out["source_global"] = out["source"]
    out["target_global"] = out["target"]
    if "rel" not in out.columns:
        out["rel"] = "COMMUNITY_LINK"
    if "weight" not in out.columns:
        out["weight"] = 1.0
    if "support_count" not in out.columns:
        out["support_count"] = 1
    if "doc_ids" not in out.columns:
        out["doc_ids"] = ""
    return out[["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"]].copy()


def build_community_to_narrative(node_to_level2: pd.DataFrame, level1_summary: pd.DataFrame) -> pd.DataFrame:
    level1_id_col = detect_id_col(level1_summary, 1)
    level1_ids = sorted(level1_summary[level1_id_col].dropna().astype(int).unique().tolist())

    if node_to_level2.empty:
        return pd.DataFrame(
            [
                {
                    "community_id": community_id,
                    "community_key": f"COMMUNITY:{community_id}",
                    "narrative_id": community_id,
                    "narrative_key": f"NARRATIVE:{community_id}",
                }
                for community_id in level1_ids
            ]
        )

    level2_id_col = detect_id_col(node_to_level2, 2)
    rows = []
    for _, row in node_to_level2.iterrows():
        node_key = row.get("node_key", "")
        community_id = parse_level_key(node_key, 1)
        if community_id is None:
            try:
                community_id = int(node_key)
            except Exception:
                continue
        narrative_id = int(row[level2_id_col])
        rows.append(
            {
                "community_id": community_id,
                "community_key": f"COMMUNITY:{community_id}",
                "narrative_id": narrative_id,
                "narrative_key": f"NARRATIVE:{narrative_id}",
            }
        )

    out = pd.DataFrame(rows).drop_duplicates()
    mapped = set(out["community_id"].astype(int)) if len(out) else set()
    missing = [community_id for community_id in level1_ids if community_id not in mapped]
    if missing:
        extra = pd.DataFrame(
            [
                {
                    "community_id": community_id,
                    "community_key": f"COMMUNITY:{community_id}",
                    "narrative_id": community_id,
                    "narrative_key": f"NARRATIVE:{community_id}",
                }
                for community_id in missing
            ]
        )
        out = pd.concat([out, extra], ignore_index=True)

    return out.sort_values(["narrative_id", "community_id"]).reset_index(drop=True)


def _build_narrative_nodes_from_mapping(community_to_narrative: pd.DataFrame, community_nodes: pd.DataFrame) -> pd.DataFrame:
    tmp = community_to_narrative.merge(
        community_nodes[["community_id", "label", "top_labels", "size"]],
        on="community_id",
        how="left",
    )

    rows = []
    for narrative_id, group in tmp.groupby("narrative_id"):
        labels = [clean_label_text(value) for value in group["label"].dropna().tolist()]
        label = "; ".join(labels[:5]) if labels else f"Narrative {narrative_id}"
        analyst_label = make_analyst_label("; ".join(labels), fallback=label)
        top_labels = "; ".join(labels[:15])
        num_communities = int(group["community_id"].nunique())
        size = int(group["size"].fillna(0).sum())
        rows.append(
            {
                "narrative_id": int(narrative_id),
                "node_key": f"NARRATIVE:{int(narrative_id)}",
                "label": label,
                "analyst_label": analyst_label,
                "num_communities": num_communities,
                "size": size,
                "top_labels": top_labels,
                "top_types": "",
                "analyst_summary": make_analyst_summary(analyst_label, f"{num_communities} communities", top_labels),
            }
        )

    return pd.DataFrame(rows).sort_values(["num_communities", "narrative_id"], ascending=[False, True]).reset_index(drop=True)


def build_narrative_nodes(level2_summary: pd.DataFrame, community_to_narrative: pd.DataFrame, community_nodes: pd.DataFrame) -> pd.DataFrame:
    if not level2_summary.empty:
        id_col = detect_id_col(level2_summary, 2)
        count_by_narrative = community_to_narrative.groupby("narrative_id")["community_id"].nunique().to_dict()

        rows = []
        for _, row in level2_summary.iterrows():
            narrative_id = int(row[id_col])
            top_labels = row.get("top_labels", "")
            fallback = row.get("display_label", f"Narrative {narrative_id}")
            label = short_label_from_top_labels(top_labels, fallback, max_items=5)
            analyst_label = make_analyst_label(top_labels, fallback=label)
            num_communities = int(count_by_narrative.get(narrative_id, 0))
            size = int(row.get("size", num_communities))
            rows.append(
                {
                    "narrative_id": narrative_id,
                    "node_key": f"NARRATIVE:{narrative_id}",
                    "label": label,
                    "analyst_label": analyst_label,
                    "num_communities": num_communities,
                    "size": size,
                    "top_labels": clean_label_text(top_labels),
                    "top_types": row.get("top_types", ""),
                    "analyst_summary": make_analyst_summary(label, f"{num_communities} communities", top_labels),
                }
            )

        out = pd.DataFrame(rows)
        existing = set(out["narrative_id"].astype(int)) if len(out) else set()
        missing = sorted(set(community_to_narrative["narrative_id"].astype(int)) - existing)
        if missing:
            extra = _build_narrative_nodes_from_mapping(
                community_to_narrative[community_to_narrative["narrative_id"].isin(missing)],
                community_nodes,
            )
            out = pd.concat([out, extra], ignore_index=True)

        return out.sort_values(["num_communities", "narrative_id"], ascending=[False, True]).reset_index(drop=True)

    return _build_narrative_nodes_from_mapping(community_to_narrative, community_nodes)


def build_narrative_edges(
    level2_meta_edges: pd.DataFrame,
    community_edges: pd.DataFrame,
    community_to_narrative: pd.DataFrame,
) -> pd.DataFrame:
    if not level2_meta_edges.empty:
        out = level2_meta_edges.copy()
        out["source"] = out["source_global"].apply(lambda value: convert_level_key(value, 2, "NARRATIVE"))
        out["target"] = out["target_global"].apply(lambda value: convert_level_key(value, 2, "NARRATIVE"))
        out["source_global"] = out["source"]
        out["target_global"] = out["target"]
        if "rel" not in out.columns:
            out["rel"] = "NARRATIVE_LINK"
        if "weight" not in out.columns:
            out["weight"] = 1.0
        if "support_count" not in out.columns:
            out["support_count"] = 1
        if "doc_ids" not in out.columns:
            out["doc_ids"] = ""
        return out[["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"]].copy()

    if community_edges.empty or community_to_narrative.empty:
        return pd.DataFrame(columns=["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    community_to_narrative_key = dict(zip(community_to_narrative["community_key"], community_to_narrative["narrative_key"]))
    rows = []
    for _, row in community_edges.iterrows():
        source = community_to_narrative_key.get(str(row["source"]))
        target = community_to_narrative_key.get(str(row["target"]))
        if source is None or target is None or source == target:
            continue
        left, right = sorted([source, target])
        rows.append(
            {
                "source": left,
                "target": right,
                "source_global": left,
                "target_global": right,
                "rel": "NARRATIVE_LINK",
                "weight": float(row.get("weight", 1.0)),
                "support_count": int(row.get("support_count", 1)),
                "doc_ids": str(row.get("doc_ids", "")),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["source", "target", "source_global", "target_global", "rel", "weight", "support_count", "doc_ids"])

    return (
        pd.DataFrame(rows)
        .groupby(["source", "target", "source_global", "target_global", "rel"], as_index=False)
        .agg(
            weight=("weight", "sum"),
            support_count=("support_count", "sum"),
            doc_ids=("doc_ids", lambda values: ";".join(sorted(set(";".join(map(str, values)).split(";")))[:30])),
        )
    )


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


def write_backward_compatibility_files(graph_dir: Path, community_nodes: pd.DataFrame, node_to_level1: pd.DataFrame) -> None:
    ensure_dir(graph_dir)

    compat_summary = community_nodes.copy()
    compat_summary["community"] = compat_summary["community_id"]
    compat_summary["display_label"] = compat_summary["label"]
    compat_summary.to_csv(graph_dir / "community_summary.csv", index=False)

    node_table = node_to_level1.copy()
    level1_col = detect_id_col(node_table, 1)
    node_table["community"] = node_table[level1_col]
    node_table.to_csv(graph_dir / "node_to_community.csv", index=False)


def write_pyvis_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, out_path: Path, max_nodes: int = 250) -> None:
    try:
        from pyvis.network import Network
    except Exception:
        print("PyVis not installed; skipping:", out_path)
        return

    ensure_dir(out_path.parent)
    nodes_df = nodes_df.head(max_nodes).copy()
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

    net.write_html(str(out_path))
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


def run_notebook_c_v2(project_dir: Path, graph_dir: Optional[Path] = None, run_visualization: bool = True) -> Dict[str, pd.DataFrame]:
    project_dir = Path(project_dir)
    resolved_graph_dir = infer_graph_dir(project_dir, graph_dir)
    out_project_dir = Path(graph_dir).parent if graph_dir is not None else project_dir

    hier_dir = out_project_dir / "hierarchy"
    vis_dir = hier_dir / "visualizations"
    ensure_dir(hier_dir)

    data = load_bv2_outputs(out_project_dir, resolved_graph_dir)
    community_nodes = build_community_nodes(data["level1_summary"])
    community_edges = build_community_edges(data["level1_meta_edges"])
    community_to_narrative = build_community_to_narrative(data["node_to_level2"], data["level1_summary"])
    narrative_nodes = build_narrative_nodes(data["level2_summary"], community_to_narrative, community_nodes)
    narrative_edges = build_narrative_edges(data["level2_meta_edges"], community_edges, community_to_narrative)
    hierarchy_index = build_hierarchy_index(community_nodes, community_to_narrative, narrative_nodes)

    community_nodes.to_csv(hier_dir / "community_nodes.csv", index=False)
    community_edges.to_csv(hier_dir / "community_edges.csv", index=False)
    community_to_narrative.to_csv(hier_dir / "community_to_narrative.csv", index=False)
    narrative_nodes.to_csv(hier_dir / "narrative_nodes.csv", index=False)
    narrative_edges.to_csv(hier_dir / "narrative_edges.csv", index=False)
    hierarchy_index.to_csv(hier_dir / "hierarchy_index.csv", index=False)

    write_backward_compatibility_files(resolved_graph_dir, community_nodes, data["node_to_level1"])
    print_diagnostics(community_nodes, community_edges, community_to_narrative, narrative_nodes, narrative_edges)

    if run_visualization:
        write_pyvis_graph(community_nodes, community_edges, vis_dir / "community_graph_pyvis.html", max_nodes=250)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_notebook_c_v2(
        project_dir=Path(args.PROJECT_DIR),
        graph_dir=Path(args.graph_dir) if args.graph_dir else None,
        run_visualization=not args.no_visualization,
    )


if __name__ == "__main__":
    main()
