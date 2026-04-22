#!/usr/bin/env python3
"""Generate OpenAI candidate fills for masked redaction boxes.

This script is the OpenAI generation step for the inverse model. It takes a
known document from the postprocessed bracketed redaction files, hides each
redacted span behind a marker that only reveals the character count, asks the
model for N plausible fills per box, and writes the results locally for later
scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


DEFAULT_POSTPROCESSED_ROOT = Path("postprocessed")
POSTPROCESSED_BRACKETED_FILE = Path("difference/unredacted_bracketed.filtered.aligned.txt")
DEFAULT_OUTPUT_ROOT = Path("modeling_outputs/inverse_openai")
DEFAULT_PROMPT_PATH = Path("modeling_scripts/prompts/openai_inverse_fill_prompt.md")
CLEAN_TAG_RE = re.compile(r"\[\[(/?)CLEAN_REDACTION_(\d+)\]\]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[A-Za-z0-9]+", re.UNICODE)
DIVERSITY_AXES = [
    {
        "axis_id": "baseline_local_continuation",
        "description": "The most locally expected continuation from the surrounding text.",
    },
    {
        "axis_id": "alternate_actor_or_source",
        "description": "A plausible fill that changes the main actor, source, or intelligence channel.",
    },
    {
        "axis_id": "escalation_or_heightened_risk",
        "description": "A version where the situation is more serious, urgent, or destabilizing.",
    },
    {
        "axis_id": "deescalation_or_limited_impact",
        "description": "A version where the situation is contained, less certain, or lower-impact.",
    },
    {
        "axis_id": "diplomatic_or_policy_angle",
        "description": "A version focused on diplomatic maneuvering, alliance politics, or policy response.",
    },
    {
        "axis_id": "military_security_angle",
        "description": "A version focused on military movement, security risk, intelligence collection, or force posture.",
    },
    {
        "axis_id": "domestic_political_angle",
        "description": "A version focused on internal politics, leadership, public opinion, party dynamics, or legitimacy.",
    },
    {
        "axis_id": "economic_logistical_angle",
        "description": "A version focused on aid, supplies, infrastructure, finance, trade, or logistics.",
    },
    {
        "axis_id": "uncertainty_or_source_dispute",
        "description": "A version centered on conflicting reports, uncertainty, source confidence, or unresolved interpretation.",
    },
    {
        "axis_id": "contrarian_document_level_hypothesis",
        "description": "A plausible but substantively different document-level hypothesis that fits the broader document more than the immediate local wording.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inverse redaction fill candidates with OpenAI.")
    parser.add_argument("--postprocessed-root", type=Path, default=DEFAULT_POSTPROCESSED_ROOT)
    parser.add_argument("--document-id", default=None, help="Document id / pair key to process.")
    parser.add_argument(
        "--document-index",
        type=int,
        default=0,
        help="Zero-based index among eligible documents when --document-id is omitted.",
    )
    parser.add_argument("--max-documents", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--candidates-per-box", type=int, default=10)
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=None,
        help="Optional test cap on boxes to request fills for; all boxes are still masked.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default because GPT-5.4 nano does not support it.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=24000)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="low",
        help="Use 'none' to omit the reasoning parameter.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--store", action="store_true", help="Allow the API to store the response.")
    parser.add_argument("--dry-run", action="store_true", help="Write the masked payload without calling OpenAI.")
    parser.add_argument(
        "--no-local-answer-key",
        action="store_true",
        help="Do not write local answer-key text next to the API outputs.",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def sanitize_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe[:180] or "document"


def diversity_axes_for_count(candidate_count: int) -> list[dict[str, str]]:
    axes = list(DIVERSITY_AXES)
    while len(axes) < candidate_count:
        idx = len(axes) + 1
        axes.append(
            {
                "axis_id": f"additional_distinct_hypothesis_{idx:02d}",
                "description": "An additional non-paraphrase hypothesis with a different main claim, actor, or implication.",
            }
        )
    return axes[:candidate_count]


def parse_postprocessed_bracketed_text(raw: str, document_id: str) -> tuple[str, list[dict[str, Any]]]:
    plain_parts: list[str] = []
    spans: list[dict[str, Any]] = []
    stack: list[tuple[int, int]] = []
    src = 0
    dst = 0

    for match in CLEAN_TAG_RE.finditer(raw):
        chunk = raw[src : match.start()]
        plain_parts.append(chunk)
        dst += len(chunk)

        closing = bool(match.group(1))
        redaction_id = int(match.group(2))
        if closing:
            for idx in range(len(stack) - 1, -1, -1):
                open_id, start = stack[idx]
                if open_id == redaction_id:
                    stack.pop(idx)
                    if dst > start:
                        spans.append(
                            {
                                "source_redaction_id": redaction_id,
                                "char_start": start,
                                "char_end": dst,
                                "closed": True,
                            }
                        )
                    break
            else:
                raise SystemExit(
                    f"Found closing CLEAN_REDACTION_{redaction_id} without an opener in {document_id}."
                )
        else:
            stack.append((redaction_id, dst))
        src = match.end()

    tail = raw[src:]
    plain_parts.append(tail)
    dst += len(tail)
    for redaction_id, start in stack:
        if dst > start:
            spans.append(
                {
                    "source_redaction_id": redaction_id,
                    "char_start": start,
                    "char_end": dst,
                    "closed": False,
                }
            )

    plain = "".join(plain_parts)
    spans = sorted(spans, key=lambda row: (int(row["char_start"]), int(row["char_end"])))
    last_end = -1
    for span in spans:
        start = int(span["char_start"])
        end = int(span["char_end"])
        if start < last_end:
            raise SystemExit(f"Overlapping CLEAN_REDACTION spans in {document_id}.")
        last_end = end
        truth = plain[start:end]
        span["truth_text_exact"] = truth
        span["truth_text_normalized"] = normalize_space(truth)
        span["char_count"] = len(truth)
        span["normalized_char_count"] = len(span["truth_text_normalized"])
        span["token_count"] = len(WORD_RE.findall(truth))
    return plain, spans


def load_postprocessed_documents(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.postprocessed_root
    if args.document_id:
        pair_dirs = [root / args.document_id]
    else:
        pair_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    docs: list[dict[str, Any]] = []
    for pair_dir in pair_dirs:
        source_file = pair_dir / POSTPROCESSED_BRACKETED_FILE
        if not source_file.exists():
            continue
        raw = source_file.read_text(encoding="utf-8", errors="replace")
        plain, spans = parse_postprocessed_bracketed_text(raw, pair_dir.name)
        if not spans:
            continue
        docs.append(
            {
                "document_id": pair_dir.name,
                "pair_key": pair_dir.name,
                "source": "postprocessed",
                "source_file": str(source_file),
                "text": plain,
                "_source_boxes": spans,
            }
        )

    if args.document_id and not docs:
        raise SystemExit(
            f"No postprocessed document found for --document-id {args.document_id!r} at "
            f"{root / args.document_id / POSTPROCESSED_BRACKETED_FILE}."
        )
    start = max(0, args.document_index) if not args.document_id else 0
    selected = docs[start : start + max(1, args.max_documents)]
    if not selected:
        raise SystemExit("No eligible postprocessed documents with CLEAN_REDACTION spans were found.")
    return selected


def load_documents(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    docs = load_postprocessed_documents(args)
    return docs, str(args.postprocessed_root / "*" / POSTPROCESSED_BRACKETED_FILE)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def select_boxes(doc: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    spans = sorted(doc.get("_source_boxes", []), key=lambda row: (int(row["char_start"]), int(row["char_end"])))
    for span_index, span in enumerate(spans, start=1):
        boxes.append(
            {
                "box_id": f"BOX_{len(boxes) + 1:03d}",
                "span_index": span_index,
                "source_redaction_id": int(span["source_redaction_id"]),
                "char_start": int(span["char_start"]),
                "char_end": int(span["char_end"]),
                "char_count": int(span["char_count"]),
                "normalized_char_count": int(span["normalized_char_count"]),
                "token_count": int(span["token_count"]),
                "source_tag_closed": bool(span.get("closed", True)),
                "truth_text_exact": str(span["truth_text_exact"]),
                "truth_text_normalized": str(span["truth_text_normalized"]),
            }
        )
    if not boxes:
        raise SystemExit(f"No CLEAN_REDACTION boxes found for document {doc.get('document_id')!r}.")
    return boxes


def mask_document(text: str, boxes: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for box in sorted(boxes, key=lambda row: int(row["char_start"])):
        start = int(box["char_start"])
        end = int(box["char_end"])
        pieces.append(text[cursor:start])
        pieces.append(f"[[{box['box_id']}: {box['char_count']} chars]]")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def public_box_manifest(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_keys = [
        "box_id",
        "span_index",
        "source_redaction_id",
        "char_start",
        "char_end",
        "char_count",
        "normalized_char_count",
        "token_count",
        "token_start",
        "token_end",
        "source_tag_closed",
    ]
    return [{key: box[key] for key in public_keys if key in box} for box in boxes]


def answer_key_manifest(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answer_keys = [
        "box_id",
        "span_index",
        "source_redaction_id",
        "char_start",
        "char_end",
        "char_count",
        "normalized_char_count",
        "token_count",
        "source_tag_closed",
        "truth_text_exact",
        "truth_text_normalized",
    ]
    return [{key: box[key] for key in answer_keys if key in box} for box in boxes]


def build_input_payload(doc: dict[str, Any], boxes: list[dict[str, Any]], masked_text: str, n: int) -> dict[str, Any]:
    return {
        "document_id": str(doc["document_id"]),
        "pair_key": str(doc.get("pair_key", doc["document_id"])),
        "source": str(doc.get("source", "")),
        "source_file": str(doc.get("source_file", "")),
        "candidate_count_per_box": n,
        "length_requirement": (
            "Non-negotiable: each candidate text must match its box char_count exactly under Python len(text). "
            "Spaces and punctuation count. Before returning JSON, count every candidate, revise if needed, and "
            "put the final counted length in the candidate char_count field. Only marginal 1-2 character errors "
            "are tolerable; exact matches are expected."
        ),
        "diversity_requirement": (
            "Generate non-paraphrase candidates. Each candidate for a box must use a different "
            "diversity axis and make a substantively different hypothesis about what the hidden text says."
        ),
        "long_box_guidance": (
            "For longer boxes, treat the full document, section headings, page order, and repeated topic structure "
            "as more important than only the immediate neighboring sentence. The local context anchors grammar and style, "
            "but the hidden paragraph may introduce a document-level development that is not simply a continuation of the nearby text."
        ),
        "diversity_axes": diversity_axes_for_count(n),
        "box_count": len(boxes),
        "boxes": public_box_manifest(boxes),
        "masked_document": masked_text,
    }


def build_response_schema(candidate_count: int, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    box_ids = [str(box["box_id"]) for box in boxes]
    axis_ids = [axis["axis_id"] for axis in diversity_axes_for_count(candidate_count)]
    candidate_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_id": {"type": "string"},
            "diversity_axis": {"type": "string", "enum": axis_ids},
            "text": {"type": "string"},
            "char_count": {"type": "integer"},
            "rationale": {"type": "string"},
            "distinctiveness": {"type": "string"},
        },
        "required": ["candidate_id", "diversity_axis", "text", "char_count", "rationale", "distinctiveness"],
    }
    box_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "box_id": {"type": "string", "enum": box_ids},
            "char_count": {"type": "integer"},
            "candidates": {
                "type": "array",
                "items": candidate_schema,
                "minItems": candidate_count,
                "maxItems": candidate_count,
            },
        },
        "required": ["box_id", "char_count", "candidates"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_id": {"type": "string"},
            "candidate_count_per_box": {"type": "integer"},
            "boxes": {
                "type": "array",
                "items": box_schema,
                "minItems": len(boxes),
                "maxItems": len(boxes),
            },
        },
        "required": ["document_id", "candidate_count_per_box", "boxes"],
    }


def build_request(
    args: argparse.Namespace,
    prompt_text: str,
    input_payload: dict[str, Any],
    boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": args.model,
        "input": [
            {"role": "developer", "content": prompt_text},
            {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False, indent=2)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "redaction_fill_candidates",
                "description": "Per-box candidate fills for masked redaction spans.",
                "schema": build_response_schema(args.candidates_per_box, boxes),
                "strict": True,
            },
            "verbosity": "medium",
        },
        "max_output_tokens": args.max_output_tokens,
        "store": bool(args.store),
    }
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.reasoning_effort != "none":
        request["reasoning"] = {"effort": args.reasoning_effort}
    return request


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {"response": str(response)}


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text is not None:
                parts.append(str(text))
            elif isinstance(content, dict) and content.get("text") is not None:
                parts.append(str(content["text"]))
    return "\n".join(parts).strip()


def parse_json_output(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate_candidates(
    parsed: dict[str, Any],
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {str(box["box_id"]): box for box in boxes}
    parsed_boxes = {str(box.get("box_id")): box for box in parsed.get("boxes", []) if isinstance(box, dict)}
    rows: list[dict[str, Any]] = []
    issues: list[str] = []

    for box_id, expected_box in expected.items():
        parsed_box = parsed_boxes.get(box_id)
        if not parsed_box:
            issues.append(f"{box_id}: missing from model output")
            continue
        candidates = parsed_box.get("candidates", [])
        if not isinstance(candidates, list):
            issues.append(f"{box_id}: candidates is not a list")
            continue
        seen_texts: set[str] = set()
        seen_axes: set[str] = set()
        for idx, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                issues.append(f"{box_id}: candidate {idx} is not an object")
                continue
            text = str(candidate.get("text", ""))
            actual_count = len(text)
            target_count = int(expected_box["char_count"])
            norm = normalize_space(text).lower()
            duplicate = norm in seen_texts
            seen_texts.add(norm)
            diversity_axis = str(candidate.get("diversity_axis", ""))
            duplicate_axis = diversity_axis in seen_axes
            if diversity_axis:
                seen_axes.add(diversity_axis)
            row = {
                "document_id": str(doc["document_id"]),
                "pair_key": str(doc.get("pair_key", doc["document_id"])),
                "source": str(doc.get("source", "")),
                "source_file": str(doc.get("source_file", "")),
                "model": model,
                "box_id": box_id,
                "span_index": int(expected_box["span_index"]),
                "source_redaction_id": expected_box.get("source_redaction_id"),
                "candidate_index": idx,
                "candidate_id": str(candidate.get("candidate_id", f"{box_id}_CAND_{idx:02d}")),
                "diversity_axis": diversity_axis,
                "text": text,
                "target_char_count": target_count,
                "actual_char_count": actual_count,
                "model_reported_char_count": candidate.get("char_count"),
                "length_delta": actual_count - target_count,
                "length_match": actual_count == target_count,
                "duplicate_within_box": duplicate,
                "duplicate_axis_within_box": duplicate_axis,
                "rationale": str(candidate.get("rationale", "")),
                "distinctiveness": str(candidate.get("distinctiveness", "")),
            }
            rows.append(row)
            if actual_count != target_count:
                issues.append(f"{box_id}/{row['candidate_id']}: length delta {row['length_delta']}")
            if duplicate:
                issues.append(f"{box_id}/{row['candidate_id']}: duplicate text within box")
            if duplicate_axis:
                issues.append(f"{box_id}/{row['candidate_id']}: duplicate diversity axis {diversity_axis}")

    expected_candidate_count = len(boxes) * int(parsed.get("candidate_count_per_box", 0) or 0)
    length_match_count = sum(1 for row in rows if row["length_match"])
    summary = {
        "document_id": str(doc["document_id"]),
        "model": model,
        "box_count": len(boxes),
        "candidate_rows": len(rows),
        "expected_candidate_rows_from_output": expected_candidate_count,
        "length_match_count": length_match_count,
        "length_match_rate": length_match_count / len(rows) if rows else 0.0,
        "issue_count": len(issues),
        "issues": issues[:200],
    }
    return rows, summary


def call_openai(request: dict[str, Any], timeout: float, max_retries: int, sleep_seconds: float) -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it before running without --dry-run.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The openai package is not installed in this environment.") from exc

    client = OpenAI(timeout=timeout)
    last_error: Exception | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            if attempt > 1:
                print(f"Retrying OpenAI request, attempt {attempt}/{max_retries}...", flush=True)
            return client.responses.create(**request)
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, max_retries):
                break
            print(f"OpenAI request failed on attempt {attempt}/{max_retries}: {exc}", flush=True)
            time.sleep(sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def copy_source_pdfs(doc: dict[str, Any], out_dir: Path) -> list[str]:
    source_file = Path(str(doc.get("source_file", "")))
    if not source_file:
        return []
    pair_dir = source_file.parents[1] if len(source_file.parents) >= 2 else source_file.parent
    source_pdf_dir = pair_dir / "source_pdfs"
    if not source_pdf_dir.exists():
        return []
    dest_dir = out_dir / "source_pdfs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for pdf in sorted(source_pdf_dir.glob("*.pdf")):
        dest = dest_dir / pdf.name
        shutil.copy2(pdf, dest)
        copied.append(str(dest.relative_to(out_dir)))
    return copied


def table_cell(value: Any) -> str:
    text = normalize_space(str(value))
    return text.replace("|", "\\|")


def write_candidate_comparison_doc(
    out_dir: Path,
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    copied_pdfs: list[str],
) -> Path:
    candidates_by_box: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        candidates_by_box.setdefault(str(row["box_id"]), []).append(row)

    lines: list[str] = [
        f"# Inverse Candidate Comparison: {doc['document_id']}",
        "",
        "This file is meant for human inspection. It pairs the local answer-key text with the generated candidates for each redaction box.",
        "",
        "## Source",
        "",
        f"- Cleaned text source: `{doc.get('source_file', '')}`",
    ]
    if copied_pdfs:
        lines.append("- Source PDFs:")
        for rel_pdf in copied_pdfs:
            lines.append(f"  - [{Path(rel_pdf).name}]({Path(rel_pdf).as_posix()})")
    else:
        lines.append("- Source PDFs: not found")

    for box in boxes:
        box_id = str(box["box_id"])
        rows = sorted(candidates_by_box.get(box_id, []), key=lambda row: int(row.get("candidate_index", 0)))
        lines.extend(
            [
                "",
                f"## {box_id}",
                "",
                f"- Source redaction id: `{box.get('source_redaction_id')}`",
                f"- Target character count: `{box.get('char_count')}`",
                f"- Token count: `{box.get('token_count')}`",
                "",
                "### Ground Truth",
                "",
                "```text",
                str(box.get("truth_text_exact", "")),
                "```",
                "",
            ]
        )
        if not rows:
            lines.extend(["### Candidates", "", "_No candidates were requested or returned for this box._", ""])
            continue

        lines.extend(
            [
                "### Candidate Summary",
                "",
                "| Candidate | Diversity axis | Chars | Delta | Exact length | Duplicate axis |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        table_cell(row.get("candidate_id", "")),
                        table_cell(row.get("diversity_axis", "")),
                        table_cell(row.get("actual_char_count", "")),
                        table_cell(row.get("length_delta", "")),
                        table_cell(row.get("length_match", "")),
                        table_cell(row.get("duplicate_axis_within_box", "")),
                    ]
                )
                + " |"
            )

        lines.extend(["", "### Candidate Texts", ""])
        for row in rows:
            lines.extend(
                [
                    f"#### {row.get('candidate_id', '')}: {row.get('diversity_axis', '')}",
                    "",
                    f"- Actual chars: `{row.get('actual_char_count')}`",
                    f"- Target chars: `{row.get('target_char_count')}`",
                    f"- Length delta: `{row.get('length_delta')}`",
                    f"- Rationale: {row.get('rationale', '')}",
                    f"- Distinctiveness: {row.get('distinctiveness', '')}",
                    "",
                    "```text",
                    str(row.get("text", "")),
                    "```",
                    "",
                ]
            )

    path = out_dir / "candidate_comparison.md"
    write_text(path, "\n".join(lines).rstrip() + "\n")
    return path


def write_document_artifacts(
    args: argparse.Namespace,
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    prompt_text: str,
    input_payload: dict[str, Any],
    request: dict[str, Any],
) -> Path:
    doc_id = str(doc["document_id"])
    out_dir = args.output_root / sanitize_path_component(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / "prompt.md", prompt_text)
    write_text(out_dir / "masked_document.txt", input_payload["masked_document"])
    write_json(out_dir / "input_payload.json", input_payload)
    write_json(out_dir / "box_manifest.public.json", public_box_manifest(boxes))
    if not args.no_local_answer_key:
        write_json(out_dir / "box_manifest.local_answer_key.json", answer_key_manifest(boxes))
    write_json(out_dir / "openai_request.preview.json", request)
    return out_dir


def process_document(args: argparse.Namespace, doc: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    all_boxes = select_boxes(doc, args)
    requested_boxes = all_boxes[: args.max_boxes] if args.max_boxes is not None else all_boxes
    if not requested_boxes:
        raise SystemExit(f"No boxes selected for document {doc.get('document_id')!r}.")
    masked_text = mask_document(str(doc["text"]), all_boxes)
    input_payload = build_input_payload(doc, requested_boxes, masked_text, args.candidates_per_box)
    input_payload["masked_box_count"] = len(all_boxes)
    input_payload["requested_box_count"] = len(requested_boxes)
    request = build_request(args, prompt_text, input_payload, requested_boxes)
    out_dir = write_document_artifacts(args, doc, all_boxes, prompt_text, input_payload, request)
    copied_pdfs = copy_source_pdfs(doc, out_dir)
    print(
        f"[generation] {doc['document_id']}: masked {len(all_boxes)} boxes, "
        f"requesting {len(requested_boxes)} x {args.candidates_per_box} candidates",
        flush=True,
    )

    run_summary: dict[str, Any] = {
        "document_id": str(doc["document_id"]),
        "source": str(doc.get("source", "")),
        "source_file": str(doc.get("source_file", "")),
        "output_dir": str(out_dir),
        "model": args.model,
        "dry_run": bool(args.dry_run),
        "masked_box_count": len(all_boxes),
        "requested_box_count": len(requested_boxes),
        "candidates_per_box": args.candidates_per_box,
        "source_pdf_copies": copied_pdfs,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.dry_run:
        run_summary["status"] = "dry_run_written"
        write_json(out_dir / "generation_summary.json", run_summary)
        print(f"[generation] {doc['document_id']}: dry-run artifacts written to {out_dir}", flush=True)
        return run_summary

    response = call_openai(request, args.timeout, args.max_retries, args.retry_sleep_seconds)
    raw_response = response_to_dict(response)
    response_text = extract_output_text(response)
    parsed = parse_json_output(response_text)
    candidate_rows, validation_summary = validate_candidates(parsed, doc, requested_boxes, args.model)

    write_json(out_dir / "raw_response.json", raw_response)
    write_text(out_dir / "response_text.txt", response_text + "\n")
    write_json(out_dir / "parsed_candidates.json", parsed)
    write_jsonl(out_dir / "candidate_fills.jsonl", candidate_rows)
    comparison_doc = None
    if not args.no_local_answer_key:
        comparison_doc = write_candidate_comparison_doc(out_dir, doc, all_boxes, candidate_rows, copied_pdfs)
    run_summary.update({"status": "completed", "validation": validation_summary})
    if comparison_doc is not None:
        run_summary["candidate_comparison_doc"] = str(comparison_doc)
    write_json(out_dir / "generation_summary.json", run_summary)
    print(
        f"[generation] {doc['document_id']}: wrote {len(candidate_rows)} candidates "
        f"({validation_summary['length_match_count']}/{len(candidate_rows)} exact length)",
        flush=True,
    )
    if comparison_doc is not None:
        print(f"[generation] {doc['document_id']}: wrote readable comparison {comparison_doc}", flush=True)
    return run_summary


def main() -> None:
    args = parse_args()
    if args.candidates_per_box < 1:
        raise SystemExit("--candidates-per-box must be at least 1.")
    prompt_text = args.prompt_path.read_text(encoding="utf-8")
    docs, source_path = load_documents(args)
    print(
        f"[generation] loaded {len(docs)} documents from canonical postprocessed source: {source_path}",
        flush=True,
    )
    summaries = [
        process_document(args, doc, prompt_text)
        for doc in tqdm(docs, desc="Generating OpenAI candidates", dynamic_ncols=True)
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "postprocessed",
        "source_path": source_path,
        "canonical_document_file": str(POSTPROCESSED_BRACKETED_FILE),
        "prompt_path": str(args.prompt_path),
        "model": args.model,
        "dry_run": bool(args.dry_run),
        "document_count": len(summaries),
        "summaries": summaries,
    }
    write_json(args.output_root / "latest_generation_manifest.json", manifest)
    print(json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
