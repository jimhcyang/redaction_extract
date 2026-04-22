#!/usr/bin/env python3
"""Build and score oracle-length inverse redaction fill candidates.

This is the first inverse-model experiment:

1. Select known redacted spans from the held-out forward-model dataset.
2. "Cheat" by using each span's true character count.
3. Generate N candidate fills by retrieving same-length redaction spans from
   other documents, while optionally including the true span as an oracle item.
4. Score candidates with length fit, local-context similarity, and optionally
   the trained forward model p_theta(m_i = 1 | x[y_k]).
5. Report posterior entropy, ground-truth rank, and probability-weighted
   semantic similarity to the true fill.

The retrieval generator is intentionally simple. It gives a deterministic
baseline and a clean evaluation harness before adding LLM/OpenAI generation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[A-Za-z0-9]+", re.UNICODE)
MAX_LENGTH = 8192


@dataclass(frozen=True)
class PoolSpan:
    document_id: str
    text: str
    char_count: int
    token_count: int


@dataclass(frozen=True)
class TargetSpan:
    target_id: str
    document_id: str
    span_index: int
    text: str
    char_count: int
    token_count: int
    token_start: int
    token_end: int
    left_context: str
    right_context: str
    context_text: str
    document: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oracle-length inverse candidate evaluation.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("modeling_outputs/forward_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("modeling_outputs/inverse_model"))
    parser.add_argument("--target-file", default="documents.test.jsonl")
    parser.add_argument("--pool-file", default="documents.train.jsonl")
    parser.add_argument("--forward-model-dir", type=Path, default=Path("modeling_outputs/forward_model/model"))
    parser.add_argument("--candidates-per-span", type=int, default=10)
    parser.add_argument("--max-spans", type=int, default=25)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--context-window", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--score-with-forward-model", action="store_true")
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=0.5)
    parser.add_argument("--forward-weight", type=float, default=2.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_words(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(str(text))]


def collect_pool_spans(docs: list[dict[str, Any]]) -> list[PoolSpan]:
    spans: list[PoolSpan] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        doc_id = str(doc["document_id"])
        for span in doc.get("redacted_token_spans", []):
            text = normalize_text(span.get("text", ""))
            if not text:
                continue
            key = (doc_id, text)
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                PoolSpan(
                    document_id=doc_id,
                    text=text,
                    char_count=len(text),
                    token_count=int(span.get("token_count", len(tokenize_words(text)))),
                )
            )
    return spans


def select_targets(docs: list[dict[str, Any]], args: argparse.Namespace) -> list[TargetSpan]:
    targets: list[TargetSpan] = []
    for doc in docs:
        doc_id = str(doc["document_id"])
        words = [str(w) for w in doc.get("words", [])]
        for span_index, span in enumerate(doc.get("redacted_token_spans", []), start=1):
            text = normalize_text(span.get("text", ""))
            if not text:
                continue
            char_count = len(text)
            if char_count < args.min_chars or char_count > args.max_chars:
                continue
            token_start = int(span["token_start"])
            token_end = int(span["token_end"])
            left = " ".join(words[max(0, token_start - args.context_window) : token_start])
            right = " ".join(words[token_end + 1 : token_end + 1 + args.context_window])
            targets.append(
                TargetSpan(
                    target_id=f"{doc_id}::span_{span_index:04d}",
                    document_id=doc_id,
                    span_index=span_index,
                    text=text,
                    char_count=char_count,
                    token_count=int(span.get("token_count", len(tokenize_words(text)))),
                    token_start=token_start,
                    token_end=token_end,
                    left_context=left,
                    right_context=right,
                    context_text=f"{left} [MISSING_SPAN] {right}".strip(),
                    document=doc,
                )
            )
    rng = random.Random(args.seed)
    rng.shuffle(targets)
    return targets[: max(0, args.max_spans)]


def generate_candidates(target: TargetSpan, pool: list[PoolSpan], n: int, seed: int) -> list[dict[str, Any]]:
    stable_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(target.target_id)) % 1000003
    rng = random.Random(seed + stable_offset)
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": f"{target.target_id}::cand_000",
            "source": "ground_truth",
            "source_document_id": target.document_id,
            "text": target.text,
            "char_count": target.char_count,
            "token_count": target.token_count,
        }
    ]
    target_norm = target.text.lower()
    eligible = [
        span
        for span in pool
        if span.document_id != target.document_id and span.text.lower() != target_norm
    ]
    rng.shuffle(eligible)
    eligible.sort(key=lambda s: (abs(s.char_count - target.char_count), abs(s.token_count - target.token_count)))
    for idx, span in enumerate(eligible[: max(0, n - 1)], start=1):
        rows.append(
            {
                "candidate_id": f"{target.target_id}::cand_{idx:03d}",
                "source": "retrieved_span",
                "source_document_id": span.document_id,
                "text": span.text,
                "char_count": span.char_count,
                "token_count": span.token_count,
            }
        )
    return rows


def length_logprob(candidate_chars: int, target_chars: int) -> float:
    sigma = max(3.0, 0.05 * max(1, target_chars))
    z = (candidate_chars - target_chars) / sigma
    return float(-0.5 * z * z)


def vector_cosines(reference: str, candidates: list[str], *, analyzer: str, ngram_range: tuple[int, int]) -> list[float]:
    if not candidates:
        return []
    try:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=True)
        mat = vectorizer.fit_transform([reference] + candidates)
        sims = cosine_similarity(mat[0], mat[1:]).ravel()
        return [float(x) for x in sims]
    except ValueError:
        return [0.0 for _ in candidates]


class ForwardScorer:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def score_candidate(self, target: TargetSpan, candidate_text: str) -> dict[str, Any]:
        candidate_words = tokenize_words(candidate_text)
        if not candidate_words:
            return {"forward_mean_prob": None, "forward_mean_logprob": None, "forward_seen_words": 0}
        original_words = [str(w) for w in target.document.get("words", [])]
        start = target.token_start
        end_exclusive = target.token_end + 1
        completed_words = original_words[:start] + candidate_words + original_words[end_exclusive:]
        candidate_word_ids = set(range(start, start + len(candidate_words)))
        enc = self.tokenizer(
            completed_words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits.float()
            probs = self.torch.softmax(logits, dim=-1)[0, :, 1].detach().cpu().numpy()
        selected = [float(prob) for prob, word_id in zip(probs, word_ids) if word_id in candidate_word_ids]
        if not selected:
            return {"forward_mean_prob": None, "forward_mean_logprob": None, "forward_seen_words": 0}
        clipped = np.clip(np.asarray(selected, dtype=np.float64), 1e-6, 1.0)
        return {
            "forward_mean_prob": float(np.mean(clipped)),
            "forward_mean_logprob": float(np.mean(np.log(clipped))),
            "forward_seen_words": int(len(set(w for w in word_ids if w in candidate_word_ids))),
        }


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = float(exp.sum())
    if denom <= 0:
        return [1.0 / len(scores) for _ in scores]
    return [float(x / denom) for x in exp]


def entropy(probs: list[float]) -> tuple[float, float]:
    if not probs:
        return 0.0, 0.0
    h = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return float(h), float(h / math.log(len(probs))) if len(probs) > 1 else 0.0


def score_target(
    target: TargetSpan,
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    forward_scorer: ForwardScorer | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_texts = [str(row["text"]) for row in candidates]
    context_sims = vector_cosines(target.context_text, candidate_texts, analyzer="word", ngram_range=(1, 2))
    semantic_sims = vector_cosines(target.text, candidate_texts, analyzer="char_wb", ngram_range=(3, 5))
    scored: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        row = dict(row)
        row["target_id"] = target.target_id
        row["target_document_id"] = target.document_id
        row["target_span_index"] = target.span_index
        row["target_char_count"] = target.char_count
        row["length_logprob"] = length_logprob(int(row["char_count"]), target.char_count)
        row["context_similarity"] = context_sims[idx] if idx < len(context_sims) else 0.0
        row["semantic_similarity_to_truth"] = semantic_sims[idx] if idx < len(semantic_sims) else 0.0
        if row["source"] == "ground_truth":
            row["semantic_similarity_to_truth"] = 1.0
        if forward_scorer is not None:
            row.update(forward_scorer.score_candidate(target, str(row["text"])))
        else:
            row["forward_mean_prob"] = None
            row["forward_mean_logprob"] = 0.0
            row["forward_seen_words"] = None
        forward_log = row["forward_mean_logprob"] if row["forward_mean_logprob"] is not None else 0.0
        row["score"] = (
            args.length_weight * float(row["length_logprob"])
            + args.context_weight * float(row["context_similarity"])
            + args.forward_weight * float(forward_log)
        )
        scored.append(row)
    probs = softmax([float(row["score"]) for row in scored])
    for row, prob in zip(scored, probs):
        row["posterior_probability"] = prob
    ranked = sorted(scored, key=lambda x: float(x["posterior_probability"]), reverse=True)
    gt_rank = next((idx for idx, row in enumerate(ranked, start=1) if row["source"] == "ground_truth"), None)
    gt_row = next(row for row in scored if row["source"] == "ground_truth")
    h, h_norm = entropy(probs)
    summary = {
        "target_id": target.target_id,
        "document_id": target.document_id,
        "span_index": target.span_index,
        "target_char_count": target.char_count,
        "target_token_count": target.token_count,
        "candidate_count": len(scored),
        "posterior_entropy": h,
        "posterior_entropy_normalized": h_norm,
        "ground_truth_rank": gt_rank,
        "ground_truth_probability": gt_row["posterior_probability"],
        "probability_weighted_semantic_similarity": float(
            sum(float(row["posterior_probability"]) * float(row["semantic_similarity_to_truth"]) for row in scored)
        ),
        "best_candidate_source": ranked[0]["source"] if ranked else None,
        "best_candidate_probability": ranked[0]["posterior_probability"] if ranked else None,
        "best_candidate_semantic_similarity": ranked[0]["semantic_similarity_to_truth"] if ranked else None,
    }
    return scored, summary


def write_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Inverse Model Plan

## Goal

The forward model estimates `p_theta(m_i = 1 | x)`: how likely each token is to be selected as a target span in a fully observed document.

The inverse task uses that forward model as a scoring prior. For an observed missing span, the inverse model generates candidate fills, inserts each candidate into context, and scores the completed document.

## Oracle-Length Experiment

The first experiment intentionally cheats by using the true character count of each held-out span. This gives an upper-bound style evaluation before replacing oracle length with box-geometry estimates.

For each target span:

1. Keep the surrounding visible context.
2. Use the true character count as the length constraint.
3. Generate `N` candidates.
4. Include the original ground-truth fill as one candidate.
5. Score every candidate.
6. Normalize scores into a finite posterior distribution.

## Scoring

```text
score(y_k) =
  beta_len * log p_len(length(y_k) | observed length)
+ beta_ctx * sim(y_k, local context)
+ beta_fwd * log p_theta(target span | x[y_k])
```

The current script uses retrieval-based candidates from other known spans. This is not the final generator, but it creates a deterministic baseline and a clean evaluation harness.

## Evaluation

The outputs report:

- posterior entropy and normalized entropy;
- rank and posterior probability of the true fill;
- probability-weighted semantic similarity to the true fill;
- best-candidate source and similarity.

If the true fill receives high posterior mass and low rank, the forward model is useful for reranking. If entropy remains high, the observed evidence does not narrow the candidate set enough. If retrieved candidates beat the truth, the score function is under-specified or the forward prior is not discriminative for that span.
""",
        encoding="utf-8",
    )


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}

    def mean(key: str) -> float:
        vals = [float(row[key]) for row in summaries if row.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    rank_vals = [int(row["ground_truth_rank"]) for row in summaries if row.get("ground_truth_rank") is not None]
    top1 = sum(1 for r in rank_vals if r == 1)
    return {
        "target_span_count": len(summaries),
        "mean_posterior_entropy": mean("posterior_entropy"),
        "mean_posterior_entropy_normalized": mean("posterior_entropy_normalized"),
        "mean_ground_truth_probability": mean("ground_truth_probability"),
        "mean_probability_weighted_semantic_similarity": mean("probability_weighted_semantic_similarity"),
        "mean_ground_truth_rank": float(np.mean(rank_vals)) if rank_vals else None,
        "ground_truth_top1_rate": top1 / len(rank_vals) if rank_vals else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_docs = read_jsonl(args.dataset_dir / args.target_file)
    pool_docs = read_jsonl(args.dataset_dir / args.pool_file)
    pool = collect_pool_spans(pool_docs)
    targets = select_targets(target_docs, args)
    if not targets:
        raise SystemExit("No target spans selected.")
    forward_scorer = ForwardScorer(args.forward_model_dir) if args.score_with_forward_model else None

    all_candidates: list[dict[str, Any]] = []
    span_summaries: list[dict[str, Any]] = []
    for target in tqdm(targets, desc="Scoring inverse candidates", dynamic_ncols=True):
        candidates = generate_candidates(target, pool, args.candidates_per_span, args.seed)
        scored, summary = score_target(target, candidates, args, forward_scorer)
        all_candidates.extend(scored)
        span_summaries.append(summary)

    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "target_file": args.target_file,
        "pool_file": args.pool_file,
        "forward_model_dir": str(args.forward_model_dir),
        "score_with_forward_model": bool(args.score_with_forward_model),
        "candidates_per_span": args.candidates_per_span,
        "max_spans": args.max_spans,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "context_window": args.context_window,
        "weights": {
            "length": args.length_weight,
            "context": args.context_weight,
            "forward": args.forward_weight,
        },
        "pool_span_count": len(pool),
        "summary": aggregate_summaries(span_summaries),
        "outputs": {
            "candidate_scores": str(args.output_dir / "candidate_scores.jsonl"),
            "span_summaries": str(args.output_dir / "span_summaries.jsonl"),
            "plan": str(args.output_dir / "inverse_model_plan.md"),
        },
    }
    write_jsonl(args.output_dir / "candidate_scores.jsonl", all_candidates)
    write_jsonl(args.output_dir / "span_summaries.jsonl", span_summaries)
    write_json(args.output_dir / "inverse_summary.json", manifest)
    write_plan(args.output_dir / "inverse_model_plan.md")
    print(json.dumps(to_jsonable(manifest["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
