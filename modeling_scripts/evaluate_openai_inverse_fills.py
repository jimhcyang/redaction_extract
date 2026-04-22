#!/usr/bin/env python3
"""Score OpenAI inverse redaction candidates with the trained forward model.

Input is the local output of generate_openai_inverse_fills.py. For each
candidate fill, this script inserts the candidate into its target box, fills
the other boxes with the local answer key, asks the forward token classifier
how redaction-like the candidate span is, and normalizes candidate scores into
a per-box posterior-style distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


MAX_LENGTH = 8192
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[A-Za-z0-9]+", re.UNICODE)
BOX_MARKER_RE = re.compile(r"\[\[(BOX_\d{3}):\s*(\d+)\s+chars\]\]")


@dataclass(frozen=True)
class WordSpan:
    text: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI inverse fill candidates.")
    parser.add_argument("--input-root", type=Path, default=Path("modeling_outputs/inverse_openai"))
    parser.add_argument("--output-dir", type=Path, default=Path("modeling_outputs/inverse_openai_eval"))
    parser.add_argument("--forward-model-dir", type=Path, default=Path("modeling_outputs/forward_model/model"))
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--context-window-chars", type=int, default=1200)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=0.5)
    parser.add_argument("--forward-weight", type=float, default=2.0)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument(
        "--include-ground-truth-candidate",
        action="store_true",
        help="Append the local answer-key span as an oracle candidate for calibration.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_with_spans(text: str) -> list[WordSpan]:
    return [WordSpan(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(str(text))]


def intersects(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


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


def softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-6)
    arr = np.asarray(scores, dtype=np.float64) / temp
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


def candidate_dirs(input_root: Path, max_documents: int | None) -> list[Path]:
    dirs = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and (path / "candidate_fills.jsonl").exists()
    )
    if max_documents is not None:
        dirs = dirs[: max(0, max_documents)]
    if not dirs:
        raise SystemExit(f"No candidate_fills.jsonl files found under {input_root}.")
    return dirs


def find_marker(masked_document: str, box_id: str) -> re.Match[str]:
    for match in BOX_MARKER_RE.finditer(masked_document):
        if match.group(1) == box_id:
            return match
    raise ValueError(f"Marker for {box_id} not found in masked_document.txt")


def local_context(masked_document: str, box_id: str, window_chars: int) -> str:
    marker = find_marker(masked_document, box_id)
    start = max(0, marker.start() - window_chars)
    end = min(len(masked_document), marker.end() + window_chars)
    context = masked_document[start:marker.start()] + " [TARGET_BOX] " + masked_document[marker.end() : end]
    return normalize_space(BOX_MARKER_RE.sub(" [OTHER_BOX] ", context))


def completed_text_for_candidate(
    masked_document: str,
    boxes_by_id: dict[str, dict[str, Any]],
    target_box_id: str,
    candidate_text: str,
) -> tuple[str, int, int]:
    parts: list[str] = []
    cursor = 0
    target_start: int | None = None
    target_end: int | None = None
    for match in BOX_MARKER_RE.finditer(masked_document):
        box_id = match.group(1)
        parts.append(masked_document[cursor : match.start()])
        if box_id == target_box_id:
            replacement = candidate_text
            target_start = sum(len(part) for part in parts)
            target_end = target_start + len(replacement)
        else:
            replacement = str(boxes_by_id.get(box_id, {}).get("truth_text_exact", ""))
        parts.append(replacement)
        cursor = match.end()
    parts.append(masked_document[cursor:])
    if target_start is None or target_end is None:
        raise ValueError(f"Target marker {target_box_id} was not found.")
    return "".join(parts), target_start, target_end


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

    def score_text_span(self, completed_text: str, span_start: int, span_end: int) -> dict[str, Any]:
        word_spans = tokenize_with_spans(completed_text)
        candidate_indices = [
            idx for idx, word in enumerate(word_spans) if intersects(word.start, word.end, span_start, span_end)
        ]
        if not candidate_indices:
            return {
                "candidate_word_count": 0,
                "forward_seen_words": 0,
                "forward_mean_prob": None,
                "forward_mean_logprob": None,
                "forward_window_start_word": None,
                "forward_window_end_word": None,
            }

        window_start, window_end = self._window_bounds(len(word_spans), candidate_indices)
        window_words = [word.text for word in word_spans[window_start:window_end]]
        shifted_candidate_indices = set(idx - window_start for idx in candidate_indices)
        enc = self.tokenizer(
            window_words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        inputs = {key: value.to(self.device) for key, value in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits.float()
            probs = self.torch.softmax(logits, dim=-1)[0, :, 1].detach().cpu().numpy()
        selected = [
            float(prob)
            for prob, word_id in zip(probs, word_ids)
            if word_id is not None and int(word_id) in shifted_candidate_indices
        ]
        seen_words = {int(word_id) for word_id in word_ids if word_id is not None and int(word_id) in shifted_candidate_indices}
        if not selected:
            return {
                "candidate_word_count": len(candidate_indices),
                "forward_seen_words": 0,
                "forward_mean_prob": None,
                "forward_mean_logprob": None,
                "forward_window_start_word": window_start,
                "forward_window_end_word": window_end,
            }
        clipped = np.clip(np.asarray(selected, dtype=np.float64), 1e-6, 1.0)
        return {
            "candidate_word_count": len(candidate_indices),
            "forward_seen_words": len(seen_words),
            "forward_mean_prob": float(np.mean(clipped)),
            "forward_mean_logprob": float(np.mean(np.log(clipped))),
            "forward_window_start_word": window_start,
            "forward_window_end_word": window_end,
        }

    @staticmethod
    def _window_bounds(word_count: int, candidate_indices: list[int]) -> tuple[int, int]:
        budget = MAX_LENGTH - 2
        if word_count <= budget:
            return 0, word_count
        cand_start = min(candidate_indices)
        cand_end = max(candidate_indices) + 1
        cand_len = cand_end - cand_start
        if cand_len >= budget:
            return cand_start, min(word_count, cand_start + budget)
        left = (budget - cand_len) // 2
        start = max(0, cand_start - left)
        end = min(word_count, start + budget)
        start = max(0, end - budget)
        return start, end


def score_box(
    *,
    doc_dir: Path,
    masked_document: str,
    boxes_by_id: dict[str, dict[str, Any]],
    box_id: str,
    candidates: list[dict[str, Any]],
    scorer: ForwardScorer,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_box = boxes_by_id[box_id]
    if args.include_ground_truth_candidate:
        candidates = list(candidates) + [
            {
                "document_id": doc_dir.name,
                "pair_key": doc_dir.name,
                "box_id": box_id,
                "candidate_id": f"{box_id}_GROUND_TRUTH",
                "candidate_index": len(candidates) + 1,
                "text": str(target_box["truth_text_exact"]),
                "source": "ground_truth",
            }
        ]

    candidate_texts = [str(row.get("text", "")) for row in candidates]
    context = local_context(masked_document, box_id, args.context_window_chars)
    context_sims = vector_cosines(context, candidate_texts, analyzer="word", ngram_range=(1, 2))
    truth_text = str(target_box["truth_text_exact"])
    truth_sims = vector_cosines(truth_text, candidate_texts, analyzer="char_wb", ngram_range=(3, 5))

    scored: list[dict[str, Any]] = []
    for idx, candidate in enumerate(tqdm(candidates, desc=f"{doc_dir.name}/{box_id}", leave=False, dynamic_ncols=True)):
        text = str(candidate.get("text", ""))
        completed_text, target_start, target_end = completed_text_for_candidate(
            masked_document, boxes_by_id, box_id, text
        )
        forward = scorer.score_text_span(completed_text, target_start, target_end)
        actual_count = len(text)
        target_count = int(target_box["char_count"])
        forward_log = forward["forward_mean_logprob"]
        if forward_log is None:
            forward_log = math.log(1e-6)
        row = dict(candidate)
        row.update(
            {
                "document_id": doc_dir.name,
                "box_id": box_id,
                "target_char_count": target_count,
                "actual_char_count": actual_count,
                "length_delta": actual_count - target_count,
                "length_match": actual_count == target_count,
                "length_logprob": length_logprob(actual_count, target_count),
                "context_similarity": context_sims[idx] if idx < len(context_sims) else 0.0,
                "semantic_similarity_to_truth": truth_sims[idx] if idx < len(truth_sims) else 0.0,
                "exact_truth_match": normalize_space(text).lower() == normalize_space(truth_text).lower(),
                **forward,
            }
        )
        if row.get("source") == "ground_truth":
            row["semantic_similarity_to_truth"] = 1.0
            row["exact_truth_match"] = True
        row["score"] = (
            args.length_weight * float(row["length_logprob"])
            + args.context_weight * float(row["context_similarity"])
            + args.forward_weight * float(forward_log)
        )
        scored.append(row)

    probs = softmax([float(row["score"]) for row in scored], args.posterior_temperature)
    for row, prob in zip(scored, probs):
        row["posterior_probability"] = prob

    ranked = sorted(scored, key=lambda row: float(row["posterior_probability"]), reverse=True)
    semantic_ranked = sorted(scored, key=lambda row: float(row["semantic_similarity_to_truth"]), reverse=True)
    h, h_norm = entropy(probs)
    truth_rows = [row for row in ranked if row.get("source") == "ground_truth" or row.get("exact_truth_match")]
    truth_rank = None
    truth_probability = None
    if truth_rows:
        truth_id = truth_rows[0].get("candidate_id")
        truth_rank = next(
            (idx for idx, row in enumerate(ranked, start=1) if row.get("candidate_id") == truth_id),
            None,
        )
        truth_probability = truth_rows[0].get("posterior_probability")

    summary = {
        "document_id": doc_dir.name,
        "box_id": box_id,
        "target_char_count": int(target_box["char_count"]),
        "candidate_count": len(scored),
        "generated_candidate_count": sum(1 for row in scored if row.get("source") != "ground_truth"),
        "length_match_rate": float(np.mean([bool(row["length_match"]) for row in scored])) if scored else 0.0,
        "posterior_entropy": h,
        "posterior_entropy_normalized": h_norm,
        "best_candidate_id": ranked[0].get("candidate_id") if ranked else None,
        "best_candidate_probability": ranked[0].get("posterior_probability") if ranked else None,
        "best_candidate_text": ranked[0].get("text") if ranked else None,
        "best_semantic_candidate_id": semantic_ranked[0].get("candidate_id") if semantic_ranked else None,
        "best_semantic_similarity": semantic_ranked[0].get("semantic_similarity_to_truth") if semantic_ranked else None,
        "probability_weighted_semantic_similarity": float(
            sum(float(row["posterior_probability"]) * float(row["semantic_similarity_to_truth"]) for row in scored)
        )
        if scored
        else 0.0,
        "truth_candidate_rank": truth_rank,
        "truth_candidate_probability": truth_probability,
        "exact_truth_present": any(bool(row.get("exact_truth_match")) for row in scored),
    }
    return scored, summary


def aggregate_summaries(box_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not box_summaries:
        return {}

    def mean(key: str) -> float | None:
        vals = [float(row[key]) for row in box_summaries if row.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    truth_ranks = [int(row["truth_candidate_rank"]) for row in box_summaries if row.get("truth_candidate_rank")]
    return {
        "box_count": len(box_summaries),
        "mean_candidate_count": mean("candidate_count"),
        "mean_length_match_rate": mean("length_match_rate"),
        "mean_posterior_entropy": mean("posterior_entropy"),
        "mean_posterior_entropy_normalized": mean("posterior_entropy_normalized"),
        "mean_probability_weighted_semantic_similarity": mean("probability_weighted_semantic_similarity"),
        "mean_best_semantic_similarity": mean("best_semantic_similarity"),
        "truth_candidate_top1_rate": (sum(1 for rank in truth_ranks if rank == 1) / len(truth_ranks))
        if truth_ranks
        else None,
        "mean_truth_candidate_rank": float(np.mean(truth_ranks)) if truth_ranks else None,
        "exact_truth_present_rate": mean("exact_truth_present"),
    }


def write_method_note(path: Path) -> None:
    path.write_text(
        """# OpenAI Inverse Candidate Evaluation

This stage consumes `modeling_outputs/inverse_openai/<document_id>/candidate_fills.jsonl`.

For each generated candidate `y_k` for box `b`, the evaluator:

1. Inserts `y_k` into box `b`.
2. Fills every other masked box with the local answer key so the forward model sees a natural completed document.
3. Runs the trained forward token classifier on the completed document.
4. Reads the mean log probability assigned to the candidate span being redacted.
5. Combines length fit, local context similarity, and forward-model likelihood.
6. Normalizes scores with a softmax over candidates for the same box.

The score is:

```text
score(y_k) =
  beta_len * log p_len(len(y_k) | observed_char_count)
+ beta_ctx * sim(y_k, visible_local_context)
+ beta_fwd * mean_i log p_theta(m_i = 1 | completed_document[y_k])
```

The normalized value is:

```text
P(y_k | document, box) = softmax_k(score(y_k) / temperature)
```

The output is not a literal Bayesian posterior yet, because the generator's
proposal distribution is not modeled. It is a calibrated finite-candidate
posterior-style distribution that is good enough for ranking, entropy, and
ground-truth similarity checks.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc_dirs = candidate_dirs(args.input_root, args.max_documents)
    print(f"[evaluation] found {len(doc_dirs)} generated document folders under {args.input_root}", flush=True)
    print(f"[evaluation] loading forward model from {args.forward_model_dir}", flush=True)
    scorer = ForwardScorer(args.forward_model_dir)

    all_scores: list[dict[str, Any]] = []
    box_summaries: list[dict[str, Any]] = []
    for doc_dir in tqdm(doc_dirs, desc="Evaluating generated documents", dynamic_ncols=True):
        masked_document = (doc_dir / "masked_document.txt").read_text(encoding="utf-8")
        boxes = read_json(doc_dir / "box_manifest.local_answer_key.json")
        boxes_by_id = {str(box["box_id"]): box for box in boxes}
        candidates = read_jsonl(doc_dir / "candidate_fills.jsonl")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            grouped.setdefault(str(candidate["box_id"]), []).append(candidate)
        box_ids = sorted(grouped)
        if args.max_boxes is not None:
            box_ids = box_ids[: max(0, args.max_boxes)]
        print(
            f"[evaluation] {doc_dir.name}: {len(box_ids)} boxes, {len(candidates)} candidate rows",
            flush=True,
        )
        for box_id in box_ids:
            if box_id not in boxes_by_id:
                raise SystemExit(f"{doc_dir}: candidate box {box_id} missing from local answer key.")
            scored, summary = score_box(
                doc_dir=doc_dir,
                masked_document=masked_document,
                boxes_by_id=boxes_by_id,
                box_id=box_id,
                candidates=grouped[box_id],
                scorer=scorer,
                args=args,
            )
            all_scores.extend(scored)
            box_summaries.append(summary)
        print(f"[evaluation] {doc_dir.name}: completed {len(box_ids)} boxes", flush=True)

    manifest = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "forward_model_dir": str(args.forward_model_dir),
        "include_ground_truth_candidate": bool(args.include_ground_truth_candidate),
        "weights": {
            "length": args.length_weight,
            "context": args.context_weight,
            "forward": args.forward_weight,
            "posterior_temperature": args.posterior_temperature,
        },
        "document_count": len(doc_dirs),
        "summary": aggregate_summaries(box_summaries),
        "outputs": {
            "candidate_scores": str(args.output_dir / "candidate_scores.jsonl"),
            "box_summaries": str(args.output_dir / "box_summaries.jsonl"),
            "method_note": str(args.output_dir / "openai_inverse_evaluation_method.md"),
        },
    }
    write_jsonl(args.output_dir / "candidate_scores.jsonl", all_scores)
    write_jsonl(args.output_dir / "box_summaries.jsonl", box_summaries)
    write_json(args.output_dir / "inverse_openai_eval_summary.json", manifest)
    write_method_note(args.output_dir / "openai_inverse_evaluation_method.md")
    print(json.dumps(to_jsonable(manifest["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
