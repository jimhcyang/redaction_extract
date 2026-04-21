#!/usr/bin/env python3
"""Summarize document lengths against the forward-model context window."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

MODEL_NAME = "answerdotai/ModernBERT-base"
MODEL_TOKEN_CUTOFF = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize document token lengths for the forward model.")
    parser.add_argument("--dataset-jsonl", type=Path, default=Path("modeling_outputs/forward_dataset/documents.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("modeling_outputs/forward_dataset"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    xs = sorted(float(v) for v in values)

    def q(p: float) -> float:
        return xs[int((len(xs) - 1) * p)]

    return {
        "min": xs[0],
        "p25": q(0.25),
        "median": statistics.median(xs),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": xs[-1],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "document_id",
        "word_tokens",
        "model_tokens",
        "model_tokens_over_cutoff",
        "word_tokens_over_cutoff",
        "encoded_words_at_cutoff",
        "positive_words_inside_cutoff",
        "positive_words_outside_cutoff",
        "redaction_spans_inside_cutoff",
        "redaction_spans_crossing_cutoff",
        "redaction_spans_outside_cutoff",
        "positive_words",
        "positive_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def model_token_count(tokenizer, words: list[str]) -> int:
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    return int(len(enc["input_ids"]))


def encoded_word_count_at_cutoff(tokenizer, words: list[str], cutoff: int) -> int:
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=cutoff,
        return_attention_mask=False,
    )
    word_ids = [idx for idx in enc.word_ids() if idx is not None]
    return int(max(word_ids) + 1) if word_ids else 0


def span_cutoff_counts(spans: list[dict[str, Any]], encoded_words: int) -> dict[str, int]:
    inside = 0
    crossing = 0
    outside = 0
    for span in spans:
        start = int(span.get("token_start", 0))
        end = int(span.get("token_end", start))
        if end < encoded_words:
            inside += 1
        elif start >= encoded_words:
            outside += 1
        else:
            crossing += 1
    return {"inside": inside, "crossing": crossing, "outside": outside}


def make_figures(out_dir: Path, rows: list[dict[str, Any]], cutoff: int, tokenizer_name: str | None) -> dict[str, str]:
    if "MPLCONFIGDIR" not in os.environ:
        tmp = Path("/tmp/mplconfig")
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(tmp)
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    word_counts = [int(row["word_tokens"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(word_counts, bins=60, color="#4c78a8", edgecolor="white")
    ax.axvline(cutoff, color="#d62728", linestyle="--", linewidth=2, label=f"{cutoff:,} token cutoff")
    ax.set_title("Document Lengths: Dataset Word Tokens")
    ax.set_xlabel("Word tokens per document")
    ax.set_ylabel("Documents")
    ax.legend()
    path = fig_dir / "document_word_token_count_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    manifest["document_word_token_count_hist"] = str(path)

    model_counts = [row.get("model_tokens") for row in rows if row.get("model_tokens") not in (None, "")]
    if model_counts:
        model_counts = [int(v) for v in model_counts]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(model_counts, bins=60, color="#f58518", edgecolor="white")
        ax.axvline(cutoff, color="#d62728", linestyle="--", linewidth=2, label=f"{cutoff:,} token cutoff")
        ax.set_title(f"Document Lengths: {tokenizer_name} Tokens")
        ax.set_xlabel("Model subtokens per document")
        ax.set_ylabel("Documents")
        ax.legend()
        path = fig_dir / "document_model_token_count_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        manifest["document_model_token_count_hist"] = str(path)
    return manifest


def main() -> None:
    args = parse_args()
    docs = load_jsonl(args.dataset_jsonl)
    tokenizer = load_tokenizer()

    rows: list[dict[str, Any]] = []
    for doc in docs:
        words = [str(w) for w in doc.get("words", [])]
        labels = [int(v) for v in doc.get("word_labels", [])]
        word_count = len(words)
        model_count = model_token_count(tokenizer, words) if tokenizer is not None else None
        encoded_words = encoded_word_count_at_cutoff(tokenizer, words, MODEL_TOKEN_CUTOFF)
        positive_inside = int(sum(labels[:encoded_words]))
        positive_outside = int(sum(labels[encoded_words:]))
        span_counts = span_cutoff_counts(doc.get("redacted_token_spans", []), encoded_words)
        rows.append(
            {
                "document_id": doc.get("document_id"),
                "word_tokens": word_count,
                "model_tokens": model_count,
                "word_tokens_over_cutoff": word_count > MODEL_TOKEN_CUTOFF,
                "model_tokens_over_cutoff": model_count > MODEL_TOKEN_CUTOFF,
                "encoded_words_at_cutoff": encoded_words,
                "positive_words_inside_cutoff": positive_inside,
                "positive_words_outside_cutoff": positive_outside,
                "redaction_spans_inside_cutoff": span_counts["inside"],
                "redaction_spans_crossing_cutoff": span_counts["crossing"],
                "redaction_spans_outside_cutoff": span_counts["outside"],
                "positive_words": int(doc.get("num_positive_words", 0)),
                "positive_rate": float(doc.get("word_positive_rate", 0.0)),
            }
        )

    word_counts = [int(row["word_tokens"]) for row in rows]
    model_counts = [int(row["model_tokens"]) for row in rows if row.get("model_tokens") not in (None, "")]
    positive_total = sum(int(row["positive_words"]) for row in rows)
    positive_outside = sum(int(row["positive_words_outside_cutoff"]) for row in rows)
    spans_inside = sum(int(row["redaction_spans_inside_cutoff"]) for row in rows)
    spans_crossing = sum(int(row["redaction_spans_crossing_cutoff"]) for row in rows)
    spans_outside = sum(int(row["redaction_spans_outside_cutoff"]) for row in rows)
    summary = {
        "dataset_jsonl": str(args.dataset_jsonl),
        "document_count": len(rows),
        "tokenizer_name": MODEL_NAME,
        "training_token_cutoff": MODEL_TOKEN_CUTOFF,
        "word_token_count_quantiles": quantiles(word_counts),
        "word_token_docs_over_cutoff": sum(1 for value in word_counts if value > MODEL_TOKEN_CUTOFF),
        "word_token_docs_over_cutoff_rate": (
            sum(1 for value in word_counts if value > MODEL_TOKEN_CUTOFF) / len(word_counts)
            if word_counts
            else 0.0
        ),
        "model_token_count_quantiles": quantiles(model_counts) if model_counts else None,
        "model_token_docs_over_cutoff": sum(1 for value in model_counts if value > MODEL_TOKEN_CUTOFF),
        "model_token_docs_over_cutoff_rate": (
            sum(1 for value in model_counts if value > MODEL_TOKEN_CUTOFF) / len(model_counts)
            if model_counts
            else 0.0
        ),
        "positive_words_total": positive_total,
        "positive_words_inside_cutoff": positive_total - positive_outside,
        "positive_words_outside_cutoff": positive_outside,
        "positive_words_outside_cutoff_rate": positive_outside / positive_total if positive_total else 0.0,
        "documents_with_positive_words_outside_cutoff": sum(
            1 for row in rows if int(row["positive_words_outside_cutoff"]) > 0
        ),
        "redaction_spans_total": spans_inside + spans_crossing + spans_outside,
        "redaction_spans_inside_cutoff": spans_inside,
        "redaction_spans_crossing_cutoff": spans_crossing,
        "redaction_spans_outside_cutoff": spans_outside,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "document_lengths.csv", rows)
    write_json(args.output_dir / "document_length_summary.json", summary)
    figure_manifest = make_figures(args.output_dir, rows, MODEL_TOKEN_CUTOFF, MODEL_NAME)
    write_json(args.output_dir / "document_length_figure_manifest.json", figure_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
