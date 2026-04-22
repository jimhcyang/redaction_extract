#!/usr/bin/env python3
"""Run the OpenAI inverse-candidate generation and evaluation workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI inverse generation followed by forward-model evaluation.")
    parser.add_argument("--postprocessed-root", type=Path, default=Path("postprocessed"))
    parser.add_argument("--output-root", type=Path, default=Path("modeling_outputs"))
    parser.add_argument("--document-id", default=None, help="Run exactly one pair key.")
    parser.add_argument("--document-index", type=int, default=0, help="Zero-based start index when document-id is omitted.")
    parser.add_argument("--max-documents", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--candidates-per-box", type=int, default=10)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="low",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default because GPT-5.4 nano does not support it.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=24000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--forward-model-dir", type=Path, default=Path("modeling_outputs/forward_model/model"))
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=0.5)
    parser.add_argument("--forward-weight", type=float, default=2.0)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--include-ground-truth-candidate", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write OpenAI request previews, then skip evaluation.")
    parser.add_argument("--store", action="store_true", help="Allow OpenAI to store generation responses.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    python = sys.executable

    inverse_openai_dir = args.output_root / "inverse_openai"
    inverse_eval_dir = args.output_root / "inverse_openai_eval"

    generation_cmd = [
        python,
        str(script_dir / "generate_openai_inverse_fills.py"),
        "--postprocessed-root",
        str(args.postprocessed_root),
        "--document-index",
        str(args.document_index),
        "--max-documents",
        str(args.max_documents),
        "--output-root",
        str(inverse_openai_dir),
        "--model",
        args.model,
        "--candidates-per-box",
        str(args.candidates_per_box),
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
        "--retry-sleep-seconds",
        str(args.retry_sleep_seconds),
    ]
    if args.document_id:
        generation_cmd.extend(["--document-id", args.document_id])
    if args.temperature is not None:
        generation_cmd.extend(["--temperature", str(args.temperature)])
    if args.dry_run:
        generation_cmd.append("--dry-run")
    if args.store:
        generation_cmd.append("--store")

    eval_cmd = [
        python,
        str(script_dir / "evaluate_openai_inverse_fills.py"),
        "--input-root",
        str(inverse_openai_dir),
        "--output-dir",
        str(inverse_eval_dir),
        "--forward-model-dir",
        str(args.forward_model_dir),
        "--max-documents",
        str(args.max_documents),
        "--length-weight",
        str(args.length_weight),
        "--context-weight",
        str(args.context_weight),
        "--forward-weight",
        str(args.forward_weight),
        "--posterior-temperature",
        str(args.posterior_temperature),
    ]
    if args.include_ground_truth_candidate:
        eval_cmd.append("--include-ground-truth-candidate")

    commands: list[list[str]] = []
    if not args.skip_generation:
        commands.append(generation_cmd)
    if not args.skip_eval and not args.dry_run:
        commands.append(eval_cmd)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "inverse_pipeline_manifest.json",
        {
            "postprocessed_root": str(args.postprocessed_root),
            "output_root": str(args.output_root),
            "generation_dir": str(inverse_openai_dir),
            "evaluation_dir": None if args.skip_eval or args.dry_run else str(inverse_eval_dir),
            "document_id": args.document_id,
            "document_index": args.document_index,
            "max_documents": args.max_documents,
            "model": args.model,
            "candidates_per_box": args.candidates_per_box,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "dry_run": bool(args.dry_run),
            "commands": commands,
            "canonical_generation_input": "postprocessed/<pair_key>/difference/unredacted_bracketed.filtered.aligned.txt",
        },
    )

    for cmd in commands:
        run(cmd)
    print("\n[inverse pipeline] done", flush=True)


if __name__ == "__main__":
    main()
