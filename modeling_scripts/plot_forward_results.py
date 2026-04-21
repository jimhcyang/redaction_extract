#!/usr/bin/env python3
"""Create reporting figures for the forward redaction model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dataset and forward-model evaluation figures.")
    parser.add_argument("--dataset-summary", type=Path, default=Path("modeling_outputs/forward_dataset/dataset_summary.json"))
    parser.add_argument("--training-summary", type=Path, default=Path("modeling_outputs/forward_model/training_summary.json"))
    parser.add_argument("--test-predictions", type=Path, default=Path("modeling_outputs/forward_model/test_word_predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flatten_predictions(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_prob: list[float] = []
    for row in rows:
        y_true.extend(int(x) for x in row.get("word_labels", []))
        y_prob.extend(float(x) for x in row.get("word_probabilities", []))
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if "MPLCONFIGDIR" not in os.environ:
        tmp = Path("/tmp/mplconfig")
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(tmp)

    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

    training = read_json(args.training_summary)
    dataset = read_json(args.dataset_summary)
    preds = read_jsonl(args.test_predictions)
    y_true, y_prob = flatten_predictions(preds)
    y_pred = (y_prob >= args.threshold).astype(np.int64)

    out_dir = args.output_dir or (args.training_summary.parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    split_stats = dataset.get("split_stats", {})
    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = ["train", "val", "test"]
    vals = [int(split_stats.get(name, {}).get("documents", 0)) for name in names]
    bars = ax.bar(names, vals, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("Document Split Counts")
    ax.set_ylabel("Documents")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom")
    path = out_dir / "split_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    manifest["split_counts"] = str(path)

    train_counts = training.get("train_label_counts", {})
    c0 = int(train_counts.get("label_0", 0))
    c1 = int(train_counts.get("label_1", 0))
    rate = c1 / (c0 + c1) if (c0 + c1) else 0.0
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(["Unredacted (0)", "Targeted redaction (1)"], [c0, c1], color=["#4c78a8", "#d62728"])
    ax.set_title(f"Training Label Balance (positive rate={rate:.4f})")
    ax.set_ylabel("Supervised subtokens")
    for bar, val in zip(bars, [c0, c1]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom")
    path = out_dir / "train_label_balance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    manifest["train_label_balance"] = str(path)

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Unredacted (0)", "Targeted (1)"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
        ax.set_title(f"Test Confusion Matrix at threshold={args.threshold:.2f}")
        path = out_dir / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        manifest["confusion_matrix"] = str(path)

        if len(set(y_true.tolist())) == 2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name="test")
            ax.set_title("Test ROC Curve")
            path = out_dir / "roc_curve.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            manifest["roc_curve"] = str(path)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name="test")
            ax.set_title("Test Precision-Recall Curve")
            path = out_dir / "precision_recall_curve.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            manifest["precision_recall_curve"] = str(path)

            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
            ax.plot(prob_pred, prob_true, "o-", label="model")
            ax.set_xlabel("Mean predicted redaction probability")
            ax.set_ylabel("Observed redaction rate")
            ax.set_title("Test Calibration")
            ax.legend()
            path = out_dir / "calibration_curve.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            manifest["calibration_curve"] = str(path)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(y_prob[y_true == 0], bins=50, alpha=0.75, label="Ground truth 0", color="#4c78a8", log=True)
        ax.hist(y_prob[y_true == 1], bins=50, alpha=0.75, label="Ground truth 1", color="#d62728", log=True)
        ax.set_title("Test Probability Distribution")
        ax.set_xlabel("Predicted redaction probability")
        ax.set_ylabel("Word tokens, log scale")
        ax.legend()
        path = out_dir / "probability_histogram.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        manifest["probability_histogram"] = str(path)

    notes = {
        "split_counts": "Verifies document-level train/validation/test separation.",
        "train_label_balance": "Shows class imbalance, motivating weighted cross entropy and threshold reporting.",
        "confusion_matrix": "Shows false positives vs false negatives at the selected operating threshold.",
        "roc_curve": "Threshold-free discrimination for targeted vs non-targeted tokens.",
        "precision_recall_curve": "Positive-class performance under class imbalance.",
        "calibration_curve": "Checks whether predicted probabilities correspond to observed event rates.",
        "probability_histogram": "Shows score separation and overlap by ground-truth class.",
    }
    write_json(out_dir / "figure_manifest.json", {"figures": manifest, "notes": notes})
    print(json.dumps({"output_dir": str(out_dir), "figures": manifest}, indent=2))


if __name__ == "__main__":
    main()
