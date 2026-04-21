#!/usr/bin/env python3
"""Run document-level cross-validation for the forward model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ModernBERT forward-model CV folds.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("modeling_outputs/forward_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("modeling_outputs/forward_cv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260421)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().parent / "train_forward_model.py"
    summaries = []
    for fold_id in range(args.folds):
        out_dir = args.output_root / f"fold_{fold_id:02d}"
        cmd = [
            sys.executable,
            str(script),
            "--dataset-dir",
            str(args.dataset_dir),
            "--output-dir",
            str(out_dir),
            "--fold-id",
            str(fold_id),
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
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        summary_path = out_dir / "training_summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    aggregate = {
        "folds": args.folds,
        "summaries": summaries,
        "test_f1_mean": mean_metric(summaries, "f1"),
        "test_precision_mean": mean_metric(summaries, "precision"),
        "test_recall_mean": mean_metric(summaries, "recall"),
        "test_auroc_mean": mean_metric(summaries, "auroc"),
        "test_auprc_mean": mean_metric(summaries, "auprc"),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cv_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mean_metric(summaries: list[dict], key: str) -> float | None:
    vals = []
    for row in summaries:
        value = row.get("test_metrics", {}).get(key)
        if isinstance(value, (int, float)):
            vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


if __name__ == "__main__":
    main()
