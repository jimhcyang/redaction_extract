#!/usr/bin/env python3
"""
Run the full knowledge-graph pipeline end to end.

Stages:
1. _prepare_project.py
2. stage_a_extract.py
3. stage_a1_context_catalog.py
4. stage_a2_rule_resolution.py
5. stage_a3_sense_clustering.py
6. stage_a4_llm_resolution.py
7. stage_a5_consolidation.py
8. stage_b_graph.py
9. stage_c_hierarchy.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd


KNOWLEDGE_GRAPH_ROOT = Path(__file__).resolve().parent

MIN_CHARS_PER_CHUNK = 80
MAX_CHARS_PER_CHUNK = 3000
PARAGRAPH_SPLIT_OVERLAP = 150


def run_step(cmd: List[str]) -> None:
    print("\n== Running ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _is_effectively_empty(text: object) -> bool:
    if text is None or pd.isna(text):
        return True
    text = str(text).strip()
    if not text:
        return True
    core = re.sub(r"[\W_]+", "", text)
    return len(core) < 10


def _split_long_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK, overlap: int = PARAGRAPH_SPLIT_OVERLAP) -> List[str]:
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHARS_PER_CHUNK:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _split_into_paragraph_chunks(body: object) -> List[str]:
    if _is_effectively_empty(body):
        return []
    body = str(body).replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    if len(raw_paragraphs) <= 1:
        raw_paragraphs = [p.strip() for p in body.split("\n") if p.strip()]

    chunks: List[str] = []
    buffer = ""
    for para in raw_paragraphs:
        if len(para) > MAX_CHARS_PER_CHUNK:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split_long_text(para))
            continue
        if not buffer:
            buffer = para
        elif len(buffer) + 2 + len(para) <= MAX_CHARS_PER_CHUNK:
            buffer += "\n\n" + para
        else:
            chunks.append(buffer.strip())
            buffer = para
    if buffer.strip():
        chunks.append(buffer.strip())
    return [c for c in chunks if len(c.strip()) >= MIN_CHARS_PER_CHUNK and not _is_effectively_empty(c)]


def _split_into_document_units(body: object) -> List[str]:
    if _is_effectively_empty(body):
        return []
    body = str(body).replace("\r\n", "\n").replace("\r", "\n").strip()
    if _is_effectively_empty(body):
        return []
    return [body]


def _split_text_units(body: object, chunk_mode: str) -> List[str]:
    if chunk_mode == "document":
        return _split_into_document_units(body)
    return _split_into_paragraph_chunks(body)


def _make_unit_id(index: int, chunk_mode: str) -> str:
    return f"doc{index}" if chunk_mode == "document" else f"p{index}"


def _completed_stage_a_keys(project_dir: Path) -> set[Tuple[str, str, str]]:
    run_dir = project_dir / "extraction_run"
    checkpoint = run_dir / "chunk_results.jsonl"
    if not checkpoint.exists():
        return set()
    keys: set[Tuple[str, str, str]] = set()
    for row in _load_jsonl(checkpoint):
        provenance = row.get("provenance", {}) if isinstance(row, dict) else {}
        keys.add(
            (
                str(provenance.get("doc_id", "")),
                str(provenance.get("source_id", "")),
                str(provenance.get("paragraph_id", "")),
            )
        )
    return keys


def _pending_stage_a_units(
    project_dir: Path,
    csv_name: str,
    start_at: int,
    n_docs: int | None,
    chunk_mode: str,
) -> int | None:
    source_csv = project_dir / csv_name
    if not source_csv.exists():
        return None
    try:
        df = pd.read_csv(source_csv)
    except Exception:
        return None
    if "body" not in df.columns:
        return None
    if "doc_id" not in df.columns:
        if "id" in df.columns:
            df["doc_id"] = df["id"].astype(str)
        else:
            df["doc_id"] = [f"DOC_{i:06d}" for i in range(len(df))]
    if "id" not in df.columns:
        df["id"] = df["doc_id"].astype(str)

    if n_docs is None:
        work_df = df.iloc[start_at:].copy()
    else:
        work_df = df.iloc[start_at:start_at + n_docs].copy()

    completed = _completed_stage_a_keys(project_dir)
    pending = 0
    for _, row in work_df.iterrows():
        body = str(row["body"]) if pd.notna(row["body"]) else ""
        if _is_effectively_empty(body):
            continue
        chunks = _split_text_units(body, chunk_mode)
        for idx, _chunk in enumerate(chunks):
            key = (
                str(row["doc_id"]),
                str(row["id"]),
                _make_unit_id(idx, chunk_mode),
            )
            if key not in completed:
                pending += 1
    return pending


def extraction_csvs_exist(project_dir: Path) -> bool:
    extraction_dir = project_dir / "extraction"
    return all(
        (extraction_dir / name).exists()
        for name in ["entities.csv", "events.csv", "claims.csv", "relations.csv"]
    )


def should_run_stage_a(
    project_dir: Path,
    csv_name: str,
    start_at: int,
    n_docs: int | None,
    chunk_mode: str,
) -> bool:
    checkpoint = project_dir / "extraction_run" / "chunk_results.jsonl"
    csv_snapshot_exists = extraction_csvs_exist(project_dir)

    if checkpoint.exists():
        pending = _pending_stage_a_units(project_dir, csv_name, start_at, n_docs, chunk_mode)
        if pending == 0 and csv_snapshot_exists:
            print("Stage A already complete for the selected slice; skipping extraction.")
            return False
        print(f"Stage A pending units for selected slice: {pending if pending is not None else 'unknown'}")
        return True

    if csv_snapshot_exists:
        print("Stage A extraction CSVs already exist and no checkpoint run dir is present; treating extraction as complete and skipping Stage A.")
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIA knowledge-graph pipeline.")
    parser.set_defaults(alias_run_openai=True)
    parser.set_defaults(hierarchy_run_openai_labels=True)
    parser.add_argument(
        "--corpus-root",
        default="postprocessed",
        help="Root folder containing the cleaned CIA corpus. Defaults to postprocessed.",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Output project directory where extraction/, graphs/, and hierarchy/ will be written.",
    )
    parser.add_argument(
        "--csv-name",
        default="input_documents.csv",
        help="Name of the prepared input CSV inside the project directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit when preparing a small project sample.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Row offset for Stage A extraction.",
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=None,
        help="Number of documents to extract in Stage A. Omit to process all prepared rows.",
    )
    parser.add_argument(
        "--max-new-chunks",
        type=int,
        default=None,
        help="Optional cap on Stage A chunk calls. Defaults to all pending chunks.",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=["paragraph", "document"],
        default="paragraph",
        help="Stage A extraction unit. 'paragraph' is safer; 'document' makes one LLM call per document.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run only project preparation and Stage A extraction.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip project preparation and reuse an existing project directory.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip Stage A and reuse an existing extraction/ directory.",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip Stage B and reuse an existing graphs/ directory.",
    )
    parser.add_argument(
        "--skip-alias-resolution",
        action="store_true",
        help="Skip Stage A1 and reuse existing entity_resolution/ canonicalization outputs if present.",
    )
    parser.add_argument(
        "--skip-hierarchy",
        action="store_true",
        help="Skip Stage C and reuse an existing hierarchy/ directory.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable optional PyVis HTML outputs in Stage C.",
    )
    parser.add_argument(
        "--run-openai-labels",
        dest="hierarchy_run_openai_labels",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-openai-labels",
        dest="hierarchy_run_openai_labels",
        action="store_false",
        help="Skip Stage C OpenAI hierarchy labeling and keep heuristic hierarchy labels.",
    )
    parser.add_argument(
        "--openai-label-model",
        default=None,
        help="Optional OpenAI model override for Stage C hierarchy labeling.",
    )
    parser.add_argument(
        "--alias-run-openai",
        dest="alias_run_openai",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-alias-openai",
        dest="alias_run_openai",
        action="store_false",
        help="Skip Stage A4 OpenAI cluster adjudication and keep unresolved A3 clusters separate.",
    )
    parser.add_argument(
        "--alias-max-openai-candidates",
        type=int,
        default=None,
        help="Optional cap on Stage A4 sense clusters to adjudicate with OpenAI. Omit to adjudicate all surviving clusters.",
    )
    parser.add_argument(
        "--a5-max-openai-candidates",
        type=int,
        default=None,
        help="Optional cap on Stage A5 consolidation pairs to adjudicate with OpenAI. Omit to adjudicate all surviving pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extract_only:
        args.skip_graph = True
        args.skip_hierarchy = True

    python = sys.executable
    project_dir = str(Path(args.project_dir).expanduser().resolve())
    csv_name = args.csv_name

    if not args.skip_prepare:
        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "_prepare_project.py"),
            "--corpus-root",
            args.corpus_root,
            "--project-dir",
            project_dir,
            "--csv-name",
            csv_name,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        run_step(cmd)

    run_extract = (not args.skip_extract) and should_run_stage_a(
        Path(project_dir),
        csv_name,
        args.start_at,
        args.n_docs,
        args.chunk_mode,
    )

    if run_extract:
        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_a_extract.py"),
            project_dir,
            csv_name,
            "--start-at",
            str(args.start_at),
        ]
        if args.n_docs is not None:
            cmd.extend(["--n-docs", str(args.n_docs)])
        if args.max_new_chunks is not None:
            cmd.extend(["--max-new-chunks", str(args.max_new_chunks)])
        cmd.extend(["--chunk-mode", args.chunk_mode])
        run_step(cmd)

    if not args.skip_alias_resolution and not args.skip_graph:
        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_a1_context_catalog.py"),
            project_dir,
        ]
        run_step(cmd)

        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_a2_rule_resolution.py"),
            project_dir,
        ]
        run_step(cmd)

        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_a3_sense_clustering.py"),
            project_dir,
        ]
        run_step(cmd)

        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_a4_llm_resolution.py"),
            project_dir,
        ]
        if args.alias_max_openai_candidates is not None:
            cmd.extend(["--max-openai-candidates", str(args.alias_max_openai_candidates)])
        if args.alias_run_openai:
            pass
        else:
            cmd.append("--skip-openai")
        run_step(cmd)

        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_a5_consolidation.py"),
            project_dir,
        ]
        if args.a5_max_openai_candidates is not None:
            cmd.extend(["--max-openai-candidates", str(args.a5_max_openai_candidates)])
        if args.alias_run_openai:
            pass
        else:
            cmd.append("--skip-openai")
        run_step(cmd)

    if not args.skip_graph:
        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_b_graph.py"),
            project_dir,
        ]
        run_step(cmd)

    if not args.skip_hierarchy:
        cmd = [
            python,
            str(KNOWLEDGE_GRAPH_ROOT / "stage_c_hierarchy.py"),
            project_dir,
        ]
        if args.no_visualization:
            cmd.append("--no-visualization")
        if not args.hierarchy_run_openai_labels:
            cmd.append("--skip-openai-labels")
        if args.openai_label_model:
            cmd.extend(["--openai-label-model", str(args.openai_label_model)])
        run_step(cmd)

    print("\nPipeline complete.")
    print(f"Project directory: {project_dir}")
    print(f"Extraction: {project_dir}/extraction")
    print(f"Graphs: {project_dir}/graphs")
    print(f"Hierarchy: {project_dir}/hierarchy")


if __name__ == "__main__":
    main()
