#!/usr/bin/env python3
"""Train/evaluate the forward token classifier p_theta(m_i = 1 | x)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 8192
THRESHOLD = 0.5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
MAX_POSITIVE_WEIGHT = 20.0


def progress(iterable, *, desc: str, total: int | None = None, leave: bool = False):
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=leave)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ModernBERT forward redaction classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("modeling_outputs/forward_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("modeling_outputs/forward_model"))
    parser.add_argument("--fold-id", type=int, default=None, help="Use cv_folds.json fold instead of fixed splits.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--max-train-docs", type=int, default=None)
    parser.add_argument("--max-val-docs", type=int, default=None)
    parser.add_argument("--max-test-docs", type=int, default=None)
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_env() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def words_from_record(row: dict[str, Any]) -> list[str]:
    return [str(x) for x in row.get("words", [])]


def labels_from_record(row: dict[str, Any]) -> list[int]:
    return [int(x) for x in row.get("word_labels", [])]


def select_by_ids(all_docs: list[dict[str, Any]], ids: list[str], max_docs: int | None = None) -> list[dict[str, Any]]:
    id_set = set(ids)
    rows = [row for row in all_docs if str(row["document_id"]) in id_set]
    if max_docs is not None:
        rows = rows[: max(0, max_docs)]
    return rows


def load_splits(dataset_dir: Path, fold_id: int | None) -> tuple[list[str], list[str], list[str], str]:
    if fold_id is None:
        splits = read_json(dataset_dir / "splits.json")
        return splits["train"], splits["val"], splits["test"], "fixed_split"
    folds = read_json(dataset_dir / "cv_folds.json")
    if fold_id < 0 or fold_id >= len(folds):
        raise ValueError(f"fold_id={fold_id} out of range for {len(folds)} folds")
    fold = folds[fold_id]
    return fold["train"], fold["val"], fold["test"], f"cv_fold_{fold_id}"


@dataclass
class Feature:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    word_ids: list[int]
    document_id: str
    original_word_count: int
    encoded_word_count: int


def build_features(
    docs: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    desc: str,
    label_all_subtokens: bool = True,
) -> list[Feature]:
    features: list[Feature] = []
    for row in progress(docs, desc=desc, total=len(docs), leave=False):
        doc_id = str(row["document_id"])
        words = words_from_record(row)
        labels = labels_from_record(row)
        if not words or len(words) != len(labels):
            continue
        enc = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        word_ids_raw = enc.word_ids()
        label_row: list[int] = []
        prev_word = None
        word_ids: list[int] = []
        max_word_idx = -1
        for word_idx in word_ids_raw:
            if word_idx is None:
                label_row.append(-100)
                word_ids.append(-1)
                prev_word = None
                continue
            y = int(labels[word_idx])
            if label_all_subtokens or word_idx != prev_word:
                label_row.append(y)
            else:
                label_row.append(-100)
            word_ids.append(int(word_idx))
            max_word_idx = max(max_word_idx, int(word_idx))
            prev_word = word_idx
        features.append(
            Feature(
                input_ids=list(enc["input_ids"]),
                attention_mask=list(enc["attention_mask"]),
                labels=label_row,
                word_ids=word_ids,
                document_id=doc_id,
                original_word_count=len(words),
                encoded_word_count=max_word_idx + 1,
            )
        )
    return features


class FeatureDataset:
    def __init__(self, rows: list[Feature]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Feature:
        return self.rows[idx]


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(rows: list[Feature]) -> dict[str, torch.Tensor | list[str] | list[list[int]]]:
        max_len = max(len(row.input_ids) for row in rows)
        input_ids = []
        attention_mask = []
        labels = []
        doc_ids: list[str] = []
        word_ids: list[list[int]] = []
        for row in rows:
            pad = max_len - len(row.input_ids)
            input_ids.append(row.input_ids + [pad_id] * pad)
            attention_mask.append(row.attention_mask + [0] * pad)
            labels.append(row.labels + [-100] * pad)
            doc_ids.append(row.document_id)
            word_ids.append(row.word_ids + [-1] * pad)
        out: dict[str, torch.Tensor | list[str] | list[list[int]]] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "document_ids": doc_ids,
            "word_ids": word_ids,
        }
        return out

    return collate


def label_counts(features: list[Feature]) -> tuple[int, int]:
    neg = 0
    pos = 0
    for row in features:
        for y in row.labels:
            if y == 0:
                neg += 1
            elif y == 1:
                pos += 1
    return neg, pos


def load_model_and_tokenizer():
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    configure_trainable_parameters(model)
    return tokenizer, model


def configure_trainable_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = True

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def parameter_counts(model) -> dict[str, int]:
    total = sum(int(param.numel()) for param in model.parameters())
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None) -> torch.Tensor:
    logits = logits.float()
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
    return loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return torch.softmax(logits, dim=-1)[..., 1]


def aggregate_word_predictions(
    docs: list[dict[str, Any]],
    features: list[Feature],
    prob_rows: list[list[float]],
) -> list[dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    for row in docs:
        words = words_from_record(row)
        labels = labels_from_record(row)
        store[str(row["document_id"])] = {
            "document_id": str(row["document_id"]),
            "words": words,
            "word_labels": labels,
            "prob_sums": [0.0] * len(words),
            "prob_counts": [0] * len(words),
            "max_seen_word": -1,
        }
    for feature, probs in zip(features, prob_rows):
        bucket = store.get(feature.document_id)
        if bucket is None:
            continue
        for idx, prob, label in zip(feature.word_ids, probs, feature.labels):
            if idx < 0 or label == -100:
                continue
            if idx >= len(bucket["prob_sums"]):
                continue
            bucket["prob_sums"][idx] += float(prob)
            bucket["prob_counts"][idx] += 1
            bucket["max_seen_word"] = max(int(bucket["max_seen_word"]), idx)
    out: list[dict[str, Any]] = []
    for doc_id, row in store.items():
        limit = int(row.pop("max_seen_word")) + 1
        if limit < 0:
            limit = 0
        words = row["words"][:limit]
        labels = row["word_labels"][:limit]
        probs = [
            (s / c if c else 0.0)
            for s, c in zip(row.pop("prob_sums")[:limit], row.pop("prob_counts")[:limit])
        ]
        row["words"] = words
        row["word_labels"] = labels
        row["word_probabilities"] = probs
        row["word_pred_labels"] = [1 if p >= 0.5 else 0 for p in probs]
        row["positive_words"] = int(sum(labels))
        row["predicted_positive_words"] = int(sum(row["word_pred_labels"]))
        out.append(row)
    return out


def metric_summary(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    if not y_true:
        return {}
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
        "token_count": int(len(y)),
        "positive_count": int((y == 1).sum()),
        "predicted_positive_count": int((pred == 1).sum()),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y, p))
    except ValueError:
        metrics["auroc"] = None
    try:
        metrics["auprc"] = float(average_precision_score(y, p))
    except ValueError:
        metrics["auprc"] = None
    return metrics


@torch.no_grad()
def evaluate_model(
    model,
    docs: list[dict[str, Any]],
    features: list[Feature],
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    desc: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    steps = 0
    prob_rows: list[list[float]] = []
    for batch in progress(loader, desc=desc, total=len(loader), leave=False):
        labels = batch["labels"].to(device)
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        with autocast_context(device):
            outputs = model(**inputs)
        loss = compute_loss(outputs.logits, labels, class_weights)
        total_loss += float(loss.detach().cpu())
        steps += 1
        probs = probabilities_from_logits(outputs.logits).detach().cpu().numpy().tolist()
        prob_rows.extend(probs)
    pred_records = aggregate_word_predictions(docs, features, prob_rows)
    y_true = [int(y) for row in pred_records for y in row["word_labels"]]
    y_prob = [float(p) for row in pred_records for p in row["word_probabilities"]]
    metrics = metric_summary(y_true, y_prob, THRESHOLD)
    metrics["loss"] = total_loss / steps if steps else 0.0
    return metrics, pred_records


def train_one(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs_all = load_jsonl(args.dataset_dir / "documents.jsonl")
    train_ids, val_ids, test_ids, split_name = load_splits(args.dataset_dir, args.fold_id)
    train_docs = select_by_ids(docs_all, train_ids, args.max_train_docs)
    val_docs = select_by_ids(docs_all, val_ids, args.max_val_docs)
    test_docs = select_by_ids(docs_all, test_ids, args.max_test_docs)
    device = device_from_env()
    tokenizer, model = load_model_and_tokenizer()
    param_counts = parameter_counts(model)
    training_strategy = "full_finetune"
    precision = "bf16_autocast" if device.type == "cuda" else "float32"

    train_features = build_features(train_docs, tokenizer, max_length=MAX_LENGTH, desc="Tokenizing train")
    val_features = build_features(val_docs, tokenizer, max_length=MAX_LENGTH, desc="Tokenizing val")
    test_features = build_features(test_docs, tokenizer, max_length=MAX_LENGTH, desc="Tokenizing test")
    if not train_features:
        raise RuntimeError("No training features were produced.")

    model.to(device)
    neg, pos = label_counts(train_features)
    class_weights = None
    if pos > 0:
        pos_weight = min(MAX_POSITIVE_WEIGHT, neg / max(pos, 1))
        class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32, device=device)
    optimizer_name = "adamw"

    setup_summary = {
        "device": str(device),
        "max_length": MAX_LENGTH,
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "test_documents": len(test_docs),
        "train_features": len(train_features),
        "val_features": len(val_features),
        "test_features": len(test_features),
        "truncated_documents": {
            "train": sum(1 for row in train_features if row.encoded_word_count < row.original_word_count),
            "val": sum(1 for row in val_features if row.encoded_word_count < row.original_word_count),
            "test": sum(1 for row in test_features if row.encoded_word_count < row.original_word_count),
        },
        "train_label_counts": {"label_0": neg, "label_1": pos},
        "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
        "optimizer": optimizer_name,
        "training_strategy": training_strategy,
        "precision": precision,
        "parameter_counts": param_counts,
    }
    print(json.dumps(to_jsonable({"setup": setup_summary}), ensure_ascii=False), flush=True)

    collate = make_collate(tokenizer)
    train_loader = DataLoader(FeatureDataset(train_features), batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(FeatureDataset(val_features), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(FeatureDataset(test_features), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)))
    total_updates = max(1, updates_per_epoch * max(1, args.epochs))
    warmup_steps = int(total_updates * WARMUP_RATIO)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_state = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        train_iter = progress(train_loader, desc=f"Epoch {epoch}/{args.epochs}", total=len(train_loader), leave=True)
        for step, batch in enumerate(train_iter, start=1):
            labels = batch["labels"].to(device)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            with autocast_context(device):
                outputs = model(**inputs)
            loss = compute_loss(outputs.logits, labels, class_weights)
            (loss / max(1, args.gradient_accumulation_steps)).backward()
            running += float(loss.detach().cpu())
            if hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                clear_device_cache(device)
                global_step += 1
        val_metrics, _ = evaluate_model(
            model,
            val_docs,
            val_features,
            val_loader,
            device,
            class_weights,
            desc=f"Validating epoch {epoch}",
        )
        row = {"epoch": epoch, "global_step": global_step, "train_loss": running / max(1, len(train_loader)), "val_metrics": val_metrics}
        history.append(row)
        val_f1 = float(val_metrics.get("f1") or 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(to_jsonable(row), ensure_ascii=False), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, val_preds = evaluate_model(
        model,
        val_docs,
        val_features,
        val_loader,
        device,
        class_weights,
        desc="Final validation",
    )
    test_metrics, test_preds = evaluate_model(
        model,
        test_docs,
        test_features,
        test_loader,
        device,
        class_weights,
        desc="Final test",
    )
    write_jsonl(args.output_dir / "val_word_predictions.jsonl", val_preds)
    write_jsonl(args.output_dir / "test_word_predictions.jsonl", test_preds)
    write_jsonl(args.output_dir / "training_history.jsonl", history)

    summary = {
        "split_name": split_name,
        "model_name": MODEL_NAME,
        "tokenizer_name": MODEL_NAME,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "truncation": "first_window",
        "threshold": THRESHOLD,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": optimizer_name,
        "training_strategy": training_strategy,
        "precision": precision,
        "parameter_counts": param_counts,
        "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "test_documents": len(test_docs),
        "train_features": len(train_features),
        "val_features": len(val_features),
        "test_features": len(test_features),
        "truncated_documents": {
            "train": sum(1 for row in train_features if row.encoded_word_count < row.original_word_count),
            "val": sum(1 for row in val_features if row.encoded_word_count < row.original_word_count),
            "test": sum(1 for row in test_features if row.encoded_word_count < row.original_word_count),
        },
        "train_label_counts": {"label_0": neg, "label_1": pos},
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    write_json(args.output_dir / "training_summary.json", summary)

    if args.save_model:
        model_dir = args.output_dir / "model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)


def main() -> None:
    args = parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
