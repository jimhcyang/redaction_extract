#!/usr/bin/env python3
"""Build token-level 0/1 redaction labels from postprocessed cleaned spans."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLEAN_TAG_RE = re.compile(r"\[\[(/?)CLEAN_REDACTION_(\d+)\]\]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[A-Za-z0-9]+", re.UNICODE)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    span_id: int | None = None


@dataclass(frozen=True)
class TokenItem:
    text: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build forward-model 0/1 token labels from postprocessed/.")
    parser.add_argument("--postprocessed-root", type=Path, default=Path("postprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("modeling_outputs/forward_dataset"))
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=20260421)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_clean_bracketed_text(text: str) -> tuple[str, list[Span]]:
    parts: list[str] = []
    spans: list[Span] = []
    stack: list[tuple[int, int]] = []
    src = 0
    dst = 0

    for match in CLEAN_TAG_RE.finditer(text):
        chunk = text[src : match.start()]
        parts.append(chunk)
        dst += len(chunk)

        closing = bool(match.group(1))
        span_id = int(match.group(2))
        if closing:
            for idx in range(len(stack) - 1, -1, -1):
                open_id, start = stack[idx]
                if open_id == span_id:
                    stack.pop(idx)
                    if dst > start:
                        spans.append(Span(start=start, end=dst, span_id=span_id))
                    break
        else:
            stack.append((span_id, dst))
        src = match.end()

    tail = text[src:]
    parts.append(tail)
    dst += len(tail)
    for span_id, start in stack:
        if dst > start:
            spans.append(Span(start=start, end=dst, span_id=span_id))

    plain = "".join(parts)
    return plain, merge_spans(spans)


def merge_spans(spans: list[Span]) -> list[Span]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, s.end, s.span_id or -1))
    merged: list[Span] = []
    for span in ordered:
        if not merged or span.start > merged[-1].end:
            merged.append(span)
            continue
        prev = merged[-1]
        merged[-1] = Span(start=prev.start, end=max(prev.end, span.end), span_id=prev.span_id)
    return merged


def tokenize(text: str) -> list[TokenItem]:
    return [TokenItem(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def intersects(a_start: int, a_end: int, span: Span) -> bool:
    return a_start < span.end and a_end > span.start


def labels_for_tokens(tokens: list[TokenItem], spans: list[Span]) -> list[int]:
    labels: list[int] = []
    cursor = 0
    for tok in tokens:
        while cursor < len(spans) and spans[cursor].end <= tok.start:
            cursor += 1
        label = 0
        j = cursor
        while j < len(spans) and spans[j].start < tok.end:
            if intersects(tok.start, tok.end, spans[j]):
                label = 1
                break
            j += 1
        labels.append(label)
    return labels


def token_span_groups(tokens: list[TokenItem], labels: list[int]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    i = 0
    while i < len(labels):
        if labels[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(labels) and labels[j + 1] == 1:
            j += 1
        groups.append(
            {
                "token_start": i,
                "token_end": j,
                "token_count": j - i + 1,
                "text": " ".join(tok.text for tok in tokens[i : j + 1]),
                "char_start": tokens[i].start,
                "char_end": tokens[j].end,
            }
        )
        i = j + 1
    return groups


def collect_bracketed_files(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for pair_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        path = pair_dir / "difference" / "unredacted_bracketed.filtered.aligned.txt"
        if path.exists():
            rows.append((pair_dir.name, path))
    return rows


def build_document(pair_key: str, path: Path) -> dict[str, Any] | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    plain, char_spans = parse_clean_bracketed_text(raw)
    tokens = tokenize(plain)
    if not tokens:
        return None
    labels = labels_for_tokens(tokens, char_spans)
    positive = int(sum(labels))
    groups = token_span_groups(tokens, labels)
    return {
        "document_id": pair_key,
        "pair_key": pair_key,
        "source_file": str(path),
        "text": plain,
        "words": [t.text for t in tokens],
        "word_spans": [[t.start, t.end] for t in tokens],
        "word_labels": labels,
        "redacted_char_spans": [[s.start, s.end] for s in char_spans],
        "redacted_token_spans": groups,
        "num_words": len(tokens),
        "num_positive_words": positive,
        "word_positive_rate": positive / len(tokens),
        "num_redaction_spans": len(groups),
    }


def positive_rate_bin(rate: float) -> int:
    if rate <= 0:
        return 0
    if rate < 0.01:
        return 1
    if rate < 0.03:
        return 2
    if rate < 0.08:
        return 3
    if rate < 0.15:
        return 4
    return 5


def stratified_split(docs: list[dict[str, Any]], train_fraction: float, val_fraction: float, seed: int) -> dict[str, list[str]]:
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must satisfy train > 0, val >= 0, train + val < 1.")

    bins: dict[int, list[str]] = {i: [] for i in range(6)}
    for row in docs:
        bins[positive_rate_bin(float(row["word_positive_rate"]))].append(str(row["document_id"]))

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(bins):
        ids = bins[key][:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        else:
            n_train = max(1, min(n, n_train))
            n_val = max(0, min(n - n_train, n_val))
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])

    return {k: sorted(v) for k, v in splits.items()}


def stratified_cv_folds(docs: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, list[str]]]:
    if k < 2:
        return []
    bins: dict[int, list[str]] = {i: [] for i in range(6)}
    for row in docs:
        bins[positive_rate_bin(float(row["word_positive_rate"]))].append(str(row["document_id"]))
    fold_tests = [[] for _ in range(k)]
    rng = random.Random(seed)
    for key in sorted(bins):
        ids = bins[key][:]
        rng.shuffle(ids)
        for idx, doc_id in enumerate(ids):
            fold_tests[idx % k].append(doc_id)
    all_ids = {str(row["document_id"]) for row in docs}
    folds: list[dict[str, list[str]]] = []
    for i in range(k):
        test = set(fold_tests[i])
        val = set(fold_tests[(i + 1) % k])
        train = all_ids - test - val
        folds.append({"fold": i, "train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return folds


def subset_rows(docs: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    id_set = set(ids)
    return [row for row in docs if str(row["document_id"]) in id_set]


def quantiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0}
    xs = sorted(float(v) for v in values)
    return {
        "min": xs[0],
        "p25": xs[int((len(xs) - 1) * 0.25)],
        "median": statistics.median(xs),
        "p75": xs[int((len(xs) - 1) * 0.75)],
        "max": xs[-1],
    }


def dataset_summary(docs: list[dict[str, Any]], splits: dict[str, list[str]], folds: list[dict[str, list[str]]]) -> dict[str, Any]:
    total_words = sum(int(row["num_words"]) for row in docs)
    total_pos = sum(int(row["num_positive_words"]) for row in docs)
    span_lengths = [
        int(span["token_count"])
        for row in docs
        for span in row.get("redacted_token_spans", [])
    ]
    split_stats: dict[str, Any] = {}
    for name, ids in splits.items():
        rows = subset_rows(docs, ids)
        words = sum(int(row["num_words"]) for row in rows)
        pos = sum(int(row["num_positive_words"]) for row in rows)
        split_stats[name] = {
            "documents": len(rows),
            "words": words,
            "positive_words": pos,
            "positive_rate": pos / words if words else 0.0,
        }
    return {
        "document_count": len(docs),
        "total_words": total_words,
        "positive_words": total_pos,
        "negative_words": total_words - total_pos,
        "positive_rate": total_pos / total_words if total_words else 0.0,
        "redaction_span_count": sum(int(row["num_redaction_spans"]) for row in docs),
        "document_word_count_quantiles": quantiles([int(row["num_words"]) for row in docs]),
        "document_positive_rate_quantiles": quantiles([float(row["word_positive_rate"]) for row in docs]),
        "redaction_span_token_count_quantiles": quantiles(span_lengths),
        "split_stats": split_stats,
        "cv_fold_count": len(folds),
    }


def make_figures(out_dir: Path, docs: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    if "MPLCONFIGDIR" not in os.environ:
        tmp = Path("/tmp/mplconfig")
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(tmp)
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    label_counts = [summary["negative_words"], summary["positive_words"]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(["No redaction target (0)", "Redaction target (1)"], label_counts, color=["#4c78a8", "#d62728"])
    ax.set_title(f"Forward Labels: positive rate={summary['positive_rate']:.4f}")
    ax.set_ylabel("Word tokens")
    for bar, val in zip(bars, label_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom")
    path = fig_dir / "label_balance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest["label_balance"] = str(path)

    rates = [float(row["word_positive_rate"]) for row in docs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rates, bins=40, color="#59a14f", edgecolor="white")
    ax.set_title("Document-Level Redaction Positive Rate")
    ax.set_xlabel("Positive word-token rate")
    ax.set_ylabel("Documents")
    path = fig_dir / "document_positive_rate_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest["document_positive_rate_hist"] = str(path)

    span_lengths = [
        int(span["token_count"])
        for row in docs
        for span in row.get("redacted_token_spans", [])
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([min(x, 200) for x in span_lengths], bins=50, color="#f28e2b", edgecolor="white")
    ax.set_title("Clean Redaction Span Lengths")
    ax.set_xlabel("Word tokens per span, clipped at 200")
    ax.set_ylabel("Spans")
    path = fig_dir / "span_length_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest["span_length_hist"] = str(path)

    split_stats = summary["split_stats"]
    names = ["train", "val", "test"]
    vals = [int(split_stats[name]["documents"]) for name in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("Document Split Counts")
    ax.set_ylabel("Documents")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom")
    path = fig_dir / "split_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    manifest["split_counts"] = str(path)
    return manifest


def main() -> None:
    args = parse_args()
    if not args.postprocessed_root.exists():
        raise FileNotFoundError(f"postprocessed root not found: {args.postprocessed_root}")
    files = collect_bracketed_files(args.postprocessed_root)
    if args.max_docs is not None:
        files = files[: max(0, args.max_docs)]
    if not files:
        raise RuntimeError("No unredacted_bracketed.filtered.aligned.txt files found.")

    docs: list[dict[str, Any]] = []
    skipped = 0
    for pair_key, path in files:
        row = build_document(pair_key, path)
        if row is None or int(row["num_words"]) < args.min_words:
            skipped += 1
            continue
        docs.append(row)

    splits = stratified_split(docs, args.train_fraction, args.val_fraction, args.split_seed)
    folds = stratified_cv_folds(docs, args.cv_folds, args.split_seed)
    summary = dataset_summary(docs, splits, folds)
    summary["postprocessed_root"] = str(args.postprocessed_root)
    summary["output_dir"] = str(args.output_dir)
    summary["skipped_documents"] = skipped
    summary["min_words"] = args.min_words

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "documents.jsonl", docs)
    for split_name, ids in splits.items():
        write_jsonl(args.output_dir / f"documents.{split_name}.jsonl", subset_rows(docs, ids))
    write_json(args.output_dir / "splits.json", splits)
    write_json(args.output_dir / "cv_folds.json", folds)
    write_json(args.output_dir / "dataset_summary.json", summary)
    if not args.no_figures:
        figure_manifest = make_figures(args.output_dir, docs, summary)
        write_json(args.output_dir / "figure_manifest.json", figure_manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
