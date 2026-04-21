#!/usr/bin/env python3
"""Run the full ModernBERT forward-modeling workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset build, length report, ModernBERT training, plotting, and CV.")
    parser.add_argument("--postprocessed-root", type=Path, default=Path("postprocessed"))
    parser.add_argument("--output-root", type=Path, default=Path("modeling_outputs"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--skip-cv", action="store_true", help="Skip the five-model CV pass.")
    parser.add_argument("--no-save-model", action="store_true", help="Do not write the final fine-tuned model weights.")
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

    dataset_dir = args.output_root / "forward_dataset"
    model_dir = args.output_root / "forward_model"
    cv_dir = args.output_root / "forward_cv"

    commands: list[list[str]] = []

    commands.append(
        [
            python,
            str(script_dir / "build_forward_dataset.py"),
            "--postprocessed-root",
            str(args.postprocessed_root),
            "--output-dir",
            str(dataset_dir),
            "--cv-folds",
            str(args.cv_folds),
        ]
    )
    commands.append(
        [
            python,
            str(script_dir / "summarize_document_lengths.py"),
            "--dataset-jsonl",
            str(dataset_dir / "documents.jsonl"),
            "--output-dir",
            str(dataset_dir),
        ]
    )

    train_cmd = [
        python,
        str(script_dir / "train_forward_model.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(model_dir),
        "--epochs",
        str(args.epochs),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--seed",
        str(args.seed),
    ]
    if not args.no_save_model:
        train_cmd.append("--save-model")
    commands.append(train_cmd)

    commands.append(
        [
            python,
            str(script_dir / "plot_forward_results.py"),
            "--dataset-summary",
            str(dataset_dir / "dataset_summary.json"),
            "--training-summary",
            str(model_dir / "training_summary.json"),
            "--test-predictions",
            str(model_dir / "test_word_predictions.jsonl"),
        ]
    )

    if not args.skip_cv:
        commands.append(
            [
                python,
                str(script_dir / "run_forward_cv.py"),
                "--dataset-dir",
                str(dataset_dir),
                "--output-root",
                str(cv_dir),
                "--folds",
                str(args.cv_folds),
                "--epochs",
                str(args.epochs),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--seed",
                str(args.seed),
            ]
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "forward_pipeline_manifest.json",
        {
            "postprocessed_root": str(args.postprocessed_root),
            "output_root": str(args.output_root),
            "dataset_dir": str(dataset_dir),
            "model_dir": str(model_dir),
            "cv_dir": None if args.skip_cv else str(cv_dir),
            "epochs": args.epochs,
            "cv_folds": args.cv_folds,
            "commands": commands,
        },
    )

    for cmd in commands:
        run(cmd)


if __name__ == "__main__":
    main()
