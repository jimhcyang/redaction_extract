#!/usr/bin/env python3
"""
Run the full knowledge-graph pipeline end to end.

Stages:
1. prepare_project.py
2. stage_a_extract.py
3. stage_a1_alias_resolution.py
4. stage_b_graph.py
5. stage_c_hierarchy.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


KNOWLEDGE_GRAPH_ROOT = Path(__file__).resolve().parent


def run_step(cmd: List[str]) -> None:
    print("\n== Running ==")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIA knowledge-graph pipeline.")
    parser.set_defaults(alias_run_openai=True)
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
        help="Skip Stage A1 alias resolution and reuse an existing generated alias dictionary if present.",
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
        "--alias-run-openai",
        dest="alias_run_openai",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-alias-openai",
        dest="alias_run_openai",
        action="store_false",
        help="Skip OpenAI adjudication during Stage A1 and use heuristic alias resolution only.",
    )
    parser.add_argument(
        "--alias-max-openai-candidates",
        type=int,
        default=None,
        help="Optional cap on Stage A1 alias candidates to adjudicate with OpenAI. Omit to adjudicate all surviving candidates.",
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
            str(KNOWLEDGE_GRAPH_ROOT / "prepare_project.py"),
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

    if not args.skip_extract:
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
            str(KNOWLEDGE_GRAPH_ROOT / "stage_a1_alias_resolution.py"),
            project_dir,
        ]
        if args.alias_run_openai:
            if args.alias_max_openai_candidates is not None:
                cmd.extend(["--max-openai-candidates", str(args.alias_max_openai_candidates)])
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
        run_step(cmd)

    print("\nPipeline complete.")
    print(f"Project directory: {project_dir}")
    print(f"Extraction: {project_dir}/extraction")
    print(f"Graphs: {project_dir}/graphs")
    print(f"Hierarchy: {project_dir}/hierarchy")


if __name__ == "__main__":
    main()
