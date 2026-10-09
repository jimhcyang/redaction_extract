"""Run the classical box detector against the frozen items-v4 source pages.

The detector remains single-page and answer-blind. After detection, this module
uses already-saved v4/Astra text and the later PDF text layer to audit whether
the detected geometry covers each known target. It makes no model or network
calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pymupdf
import pypdfium2 as pdfium

from .box_pipeline import (
    InputRecord,
    collect_pdf_pairs_with_stats,
    process_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = REPO_ROOT / "data" / "bench" / "items_v4_final.jsonl"
DEFAULT_ASTRA = REPO_ROOT / "results" / "full_v6_astra.jsonl"
DEFAULT_DOCS = REPO_ROOT / "data" / "docs"
DEFAULT_OUTPUT = REPO_ROOT / "box_results" / "items_v4"
DETECTOR_VERSION = "3.6.0"
AUDIT_VERSION = "v4-box-audit-4"
FULL_COVERAGE_THRESHOLD = 0.80
PARTIAL_COVERAGE_THRESHOLD = 0.30
MARKER = re.compile(r"\[(?:STILL\s+)?REDACTED\]", re.IGNORECASE)


@dataclass(frozen=True)
class LocatedTarget:
    status: str
    match_score: float
    combined_score: float
    runner_up_margin: float
    matched_text: str
    word_boxes_pdf: tuple[tuple[float, float, float, float], ...]
    page_size: tuple[float, float]


@dataclass(frozen=True)
class PageJob:
    key: str
    doc_id: str
    redacted_pdf: str
    later_pdf: str
    redacted_page: int
    later_page: int
    fragments: tuple[dict[str, Any], ...]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _normal_tokens(text: Any) -> list[str]:
    plain = unicodedata.normalize("NFKD", MARKER.sub(" ", str(text or "")))
    plain = plain.encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", plain)


def _passage_refs(item_id: str) -> tuple[str, list[tuple[int, int]]]:
    doc_id, suffix = item_id.split(":", 1)
    refs: list[tuple[int, int]] = []
    for part in suffix.split("+"):
        match = re.fullmatch(r"(\d+)\.(\d+)", part)
        if not match:
            raise ValueError(f"Unsupported v4 item id: {item_id}")
        refs.append((int(match.group(1)), int(match.group(2))))
    return doc_id, refs


def _job_key(doc_id: str, redacted_page: int, later_page: int) -> str:
    digest = hashlib.sha256(
        f"{doc_id}:{redacted_page}:{later_page}".encode("utf-8")
    ).hexdigest()[:10]
    return f"page_{redacted_page:03d}_{later_page:03d}_{digest}"


def build_scope(
    items_path: Path, astra_path: Path, docs_root: Path
) -> tuple[list[PageJob], list[dict[str, Any]], dict[str, Any]]:
    """Join final v4 items to their exact source-page pairs."""

    items = _jsonl(items_path)
    astra_rows = _jsonl(astra_path)
    astra = {
        (str(row["id"]).split(":", 1)[0], int(row["r_page"])): row
        for row in astra_rows
    }
    pairs, pair_stats = collect_pdf_pairs_with_stats(
        docs_root / "cibcia.csv",
        docs_root / "redacted_pdfs",
        docs_root / "unredacted_pdfs",
    )
    pair_index = {pair.redacted_doc_id: pair for pair in pairs}
    missing_docs = sorted({str(item["doc_id_r"]) for item in items} - pair_index.keys())
    if missing_docs:
        raise RuntimeError(
            f"v4 references {len(missing_docs)} unavailable PDF pairs; "
            f"first: {missing_docs[:3]}"
        )

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    item_rows: list[dict[str, Any]] = []
    for item in items:
        parent_id = str(item["id"])
        doc_id, refs = _passage_refs(parent_id)
        fragment_tokens: list[str] = []
        for fragment_index, (redacted_page, span_index) in enumerate(refs, start=1):
            source = astra.get((doc_id, redacted_page))
            if source is None:
                raise RuntimeError(f"Missing Astra source row for {doc_id}:{redacted_page}")
            spans = [
                span
                for span in source.get("result", {}).get("spans", [])
                if span.get("status") in {"revealed", "partial"}
            ]
            if span_index >= len(spans):
                raise RuntimeError(
                    f"Missing Astra span {span_index} for {doc_id}:{redacted_page}"
                )
            span = spans[span_index]
            fragment_answer = str(span.get("extracted", ""))
            fragment_tokens.extend(_normal_tokens(fragment_answer))
            later_page = int(source["u_page"])
            fragment = {
                "fragment_id": f"{parent_id}#fragment-{fragment_index}",
                "parent_item_id": parent_id,
                "fragment_index": fragment_index,
                "fragment_count": len(refs),
                "doc_id": doc_id,
                "redacted_page": redacted_page,
                "later_page": later_page,
                "span_index": span_index,
                "answer": fragment_answer,
                "left_context": str(span.get("left", "")),
                "right_context": str(span.get("right", "")),
                "revealed": item.get("revealed"),
                "length_bucket": item.get("length_bucket"),
                "date": item.get("date"),
                "label_cross_page": bool(item.get("label_cross_page")),
            }
            grouped[(doc_id, redacted_page, later_page)].append(fragment)
            item_rows.append(fragment)
        if fragment_tokens != _normal_tokens(item.get("answer", "")):
            raise RuntimeError(f"v4 answer/source mismatch for {parent_id}")

    jobs: list[PageJob] = []
    for doc_id, redacted_page, later_page in sorted(grouped):
        pair = pair_index[doc_id]
        jobs.append(
            PageJob(
                key=_job_key(doc_id, redacted_page, later_page),
                doc_id=doc_id,
                redacted_pdf=str(pair.redacted_pdf),
                later_pdf=str(pair.unredacted_pdf),
                redacted_page=redacted_page,
                later_page=later_page,
                fragments=tuple(grouped[(doc_id, redacted_page, later_page)]),
            )
        )
    scope = {
        "items": len(items),
        "item_fragments": len(item_rows),
        "cross_page_items": sum(bool(item.get("label_cross_page")) for item in items),
        "source_documents": len({str(item["doc_id_r"]) for item in items}),
        "unique_page_pairs": len(jobs),
        "pair_manifest": asdict(pair_stats),
        "revealed": dict(Counter(str(item.get("revealed")) for item in items)),
    }
    return jobs, items, scope


def _render_page(pdf_path: Path, page_1based: int, output: Path, dpi: int) -> None:
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_1based < 1 or page_1based > len(document):
            raise IndexError(
                f"Page {page_1based} outside 1..{len(document)} for {pdf_path.name}"
            )
        page = document.get_page(page_1based - 1)
        try:
            page.render(scale=dpi / 72.0).to_pil().save(output)
        finally:
            page.close()
    finally:
        document.close()


def _page_words(
    pdf_path: Path, page_1based: int
) -> tuple[list[dict[str, Any]], tuple[float, float]]:
    document = pymupdf.open(pdf_path)
    try:
        page = document[page_1based - 1]
        words: list[dict[str, Any]] = []
        for word_index, raw in enumerate(page.get_text("words", sort=True)):
            x1, y1, x2, y2, text, block, line, _ = raw
            for token in _normal_tokens(text):
                words.append(
                    {
                        "token": token,
                        "raw": str(text),
                        "word_index": word_index,
                        "bounds_pdf": (float(x1), float(y1), float(x2), float(y2)),
                        "block": int(block),
                        "line": int(line),
                    }
                )
        return words, (float(page.rect.width), float(page.rect.height))
    finally:
        document.close()


def _sequence_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    return max(
        SequenceMatcher(None, left, right, autojunk=False).ratio(),
        SequenceMatcher(None, "".join(left), "".join(right), autojunk=False).ratio(),
    )


def _candidate_starts(page_tokens: list[str], answer_tokens: list[str]) -> set[int]:
    starts: set[int] = set()
    if not answer_tokens:
        return starts
    answer_length = len(answer_tokens)
    if answer_length <= len(page_tokens):
        for index in range(len(page_tokens) - answer_length + 1):
            if page_tokens[index : index + answer_length] == answer_tokens:
                starts.add(index)
    matcher = SequenceMatcher(None, page_tokens, answer_tokens, autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size:
            predicted = block.a - block.b
            starts.update(
                range(max(0, predicted - 4), min(len(page_tokens), predicted + 5))
            )
    return starts


def _anchor_starts(
    page_tokens: list[str], anchor: list[str], *, after: bool
) -> set[int]:
    if not anchor or not page_tokens:
        return set()
    length = min(len(anchor), 10)
    anchor = anchor[:length] if after else anchor[-length:]
    scored: list[tuple[float, int]] = []
    for start in range(max(1, len(page_tokens) - length + 1)):
        score = _sequence_ratio(anchor, page_tokens[start : start + length])
        if score >= 0.55:
            scored.append((score, start if after else start + length))
    starts: set[int] = set()
    for _, target_start in sorted(scored, reverse=True)[:8]:
        starts.update(
            range(max(0, target_start - 3), min(len(page_tokens), target_start + 4))
        )
    return starts


def locate_target(
    words: list[dict[str, Any]],
    page_size: tuple[float, float],
    fragment: dict[str, Any],
) -> LocatedTarget:
    answer = _normal_tokens(fragment.get("answer"))
    page_tokens = [str(word["token"]) for word in words]
    if not words or not answer:
        return LocatedTarget("NO_TEXT_LAYER_OR_ANSWER", 0, 0, 0, "", (), page_size)
    before = _normal_tokens(fragment.get("left_context"))[-10:]
    after = _normal_tokens(fragment.get("right_context"))[:10]
    starts = _candidate_starts(page_tokens, answer)
    starts.update(_anchor_starts(page_tokens, before, after=False))
    starts.update(_anchor_starts(page_tokens, after, after=True))
    if not starts:
        starts.update(range(0, len(page_tokens), max(1, len(answer) // 3)))
    delta = max(2, round(len(answer) * 0.25))
    candidates: list[tuple[float, float, int, int]] = []
    for start in sorted(starts):
        for length in range(max(1, len(answer) - delta), len(answer) + delta + 1):
            if start + length > len(page_tokens):
                continue
            observed = page_tokens[start : start + length]
            answer_score = _sequence_ratio(answer, observed)
            before_score = _sequence_ratio(
                before, page_tokens[max(0, start - len(before)) : start]
            )
            after_score = _sequence_ratio(
                after, page_tokens[start + length : start + length + len(after)]
            )
            context_weight = 0.22 if len(answer) <= 3 else 0.10
            combined = answer_score * (1 - context_weight) + max(
                before_score, after_score
            ) * context_weight
            candidates.append((combined, answer_score, start, length))
    if not candidates:
        return LocatedTarget("TARGET_NOT_IN_TEXT_LAYER", 0, 0, 0, "", (), page_size)
    candidates.sort(reverse=True)
    combined, answer_score, start, length = candidates[0]
    runner = next(
        (candidate for candidate in candidates[1:] if abs(candidate[2] - start) > 2),
        None,
    )
    selected = words[start : start + length]
    unique_ids: list[int] = []
    by_id: dict[int, dict[str, Any]] = {}
    for word in selected:
        word_id = int(word["word_index"])
        by_id[word_id] = word
        if not unique_ids or unique_ids[-1] != word_id:
            unique_ids.append(word_id)
    selected_words = [by_id[word_id] for word_id in unique_ids]
    status = (
        "PASS"
        if answer_score >= 0.70
        else "PARTIAL_TEXT_MATCH"
        if answer_score >= 0.50
        else "TARGET_NOT_IN_TEXT_LAYER"
    )
    return LocatedTarget(
        status=status,
        match_score=round(float(answer_score), 6),
        combined_score=round(float(combined), 6),
        runner_up_margin=round(float(combined - (runner[0] if runner else 0)), 6),
        matched_text=" ".join(str(word["raw"]) for word in selected_words),
        word_boxes_pdf=tuple(
            tuple(float(value) for value in word["bounds_pdf"])
            for word in selected_words
        ),
        page_size=page_size,
    )


def _registration_features(
    gray: np.ndarray, maximum_dimension: int = 1600
) -> tuple[np.ndarray, float]:
    scale = min(1.0, maximum_dimension / max(gray.shape))
    if scale == 1:
        return gray, 1.0
    return (
        cv2.resize(
            gray,
            (round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def _hull_coverage(points: np.ndarray, image_shape: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))) / max(
        1.0, float(np.prod(image_shape))
    )


def register_pages(
    earlier: np.ndarray, later: np.ndarray
) -> tuple[np.ndarray | None, dict[str, Any]]:
    earlier_small, earlier_scale = _registration_features(earlier)
    later_small, later_scale = _registration_features(later)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.025)
    ek, ed = sift.detectAndCompute(earlier_small, None)
    lk, ld = sift.detectAndCompute(later_small, None)
    diagnostics: dict[str, Any] = {
        "method": "SIFT_RANSAC_HOMOGRAPHY",
        "status": "FAILED_NO_DESCRIPTORS",
        "earlier_keypoints": len(ek),
        "later_keypoints": len(lk),
    }
    if ed is None or ld is None:
        return None, diagnostics
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ld, ed, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    diagnostics["ratio_test_matches"] = len(good)
    if len(good) < 20:
        diagnostics["status"] = "FAILED_TOO_FEW_MATCHES"
        return None, diagnostics
    later_points = np.float32([lk[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    earlier_points = np.float32([ek[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    later_points /= later_scale
    earlier_points /= earlier_scale
    homography, mask = cv2.findHomography(later_points, earlier_points, cv2.RANSAC, 4.5)
    if homography is None or mask is None:
        diagnostics["status"] = "FAILED_HOMOGRAPHY"
        return None, diagnostics
    inlier_flags = mask.ravel().astype(bool)
    inliers = int(np.count_nonzero(inlier_flags))
    ratio = inliers / max(1, len(good))
    projected = cv2.perspectiveTransform(later_points, homography)
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - earlier_points.reshape(-1, 2), axis=1
    )
    median_error = float(np.median(errors[inlier_flags])) if inliers else float("inf")
    later_hull = _hull_coverage(later_points.reshape(-1, 2)[inlier_flags], later.shape)
    earlier_hull = _hull_coverage(
        earlier_points.reshape(-1, 2)[inlier_flags], earlier.shape
    )
    corners = np.float32(
        [[[0, 0]], [[later.shape[1] - 1, 0]], [[later.shape[1] - 1, later.shape[0] - 1]], [[0, later.shape[0] - 1]]]
    )
    warped_corners = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    area_ratio = abs(float(cv2.contourArea(warped_corners))) / max(1.0, earlier.size)
    bounds_ok = bool(
        warped_corners[:, 0].min() >= -0.25 * earlier.shape[1]
        and warped_corners[:, 0].max() <= 1.25 * earlier.shape[1]
        and warped_corners[:, 1].min() >= -0.25 * earlier.shape[0]
        and warped_corners[:, 1].max() <= 1.25 * earlier.shape[0]
    )
    plausible = bool(
        cv2.isContourConvex(warped_corners.astype(np.float32))
        and 0.50 <= area_ratio <= 1.90
        and bounds_ok
    )
    accepted = bool(
        plausible
        and inliers >= 15
        and median_error <= 4.5
        and (
            ratio >= 0.35
            or (
                inliers >= 50
                and median_error <= 1.25
                and later_hull >= 0.02
                and earlier_hull >= 0.02
            )
        )
    )
    diagnostics.update(
        {
            "status": "PASS" if accepted else "FAILED_WEAK_GEOMETRY",
            "inliers": inliers,
            "inlier_ratio": round(ratio, 6),
            "median_reprojection_error": round(median_error, 6),
            "later_inlier_hull_coverage": round(later_hull, 6),
            "earlier_inlier_hull_coverage": round(earlier_hull, 6),
            "projected_page_area_ratio": round(area_ratio, 6),
        }
    )
    return (homography, diagnostics) if accepted else (None, diagnostics)


def _original_to_deskewed(payload: dict[str, Any]) -> np.ndarray:
    inverse = np.asarray(
        payload["detector_diagnostics"]["inverse_affine_to_original"],
        dtype=np.float32,
    )
    return cv2.invertAffineTransform(inverse)


def _word_polygons(
    target: LocatedTarget,
    later_payload: dict[str, Any],
    homography: np.ndarray,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    original = cv2.imread(
        str(later_payload["output_files"]["original_source_png"]),
        cv2.IMREAD_GRAYSCALE,
    )
    if original is None or not target.word_boxes_pdf:
        return [], []
    page_width, page_height = target.page_size
    transform = _original_to_deskewed(later_payload)
    later_polygons: list[list[list[float]]] = []
    earlier_polygons: list[list[list[float]]] = []
    for x1, y1, x2, y2 in target.word_boxes_pdf:
        points = np.float32(
            [
                [x1 / page_width * original.shape[1], y1 / page_height * original.shape[0]],
                [x2 / page_width * original.shape[1], y1 / page_height * original.shape[0]],
                [x2 / page_width * original.shape[1], y2 / page_height * original.shape[0]],
                [x1 / page_width * original.shape[1], y2 / page_height * original.shape[0]],
            ]
        ).reshape(-1, 1, 2)
        later_points = cv2.transform(points, transform)
        earlier_points = cv2.perspectiveTransform(later_points, homography)
        later_polygons.append(later_points.reshape(-1, 2).round(3).tolist())
        earlier_polygons.append(earlier_points.reshape(-1, 2).round(3).tolist())
    return later_polygons, earlier_polygons


def target_coverage(
    polygons: list[list[list[float]]], payload: dict[str, Any]
) -> dict[str, Any]:
    if not polygons:
        return {
            "target_words": 0,
            "covered_words": 0,
            "coverage": 0.0,
            "regions": [],
            "components": [],
        }
    line_height = float(
        payload.get("detector_diagnostics", {}).get("estimated_text_line_height", 12)
    )
    padding = max(3.0, line_height * 0.28)
    counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    covered = 0
    regions = payload.get("redaction_regions", [])
    for polygon in polygons:
        center = tuple(np.asarray(polygon, dtype=np.float32).mean(axis=0).tolist())
        best_region = None
        best_component = None
        best_distance = -float("inf")
        for region in regions:
            for component in region.get("components", []):
                contour = np.asarray(component["polygon_xy"], dtype=np.float32)
                distance = float(
                    cv2.pointPolygonTest(contour.reshape(-1, 1, 2), center, True)
                )
                if distance > best_distance:
                    best_distance = distance
                    best_region = str(region["region_id"])
                    best_component = str(component["component_id"])
        if best_region is not None and best_distance >= -padding:
            covered += 1
            counts[best_region] += 1
            if best_component is not None:
                component_counts[best_component] += 1
    return {
        "target_words": len(polygons),
        "covered_words": covered,
        "coverage": round(covered / len(polygons), 6),
        "regions": sorted(counts),
        "components": sorted(component_counts),
    }


def _assignment_status_from_values(
    target_status: str,
    coverage: float,
    registered: bool,
    *,
    full_coverage_threshold: float = FULL_COVERAGE_THRESHOLD,
    partial_coverage_threshold: float = PARTIAL_COVERAGE_THRESHOLD,
) -> str:
    if not registered:
        return "REGISTRATION_FAILED"
    if target_status in {"NO_TEXT_LAYER_OR_ANSWER", "TARGET_NOT_IN_TEXT_LAYER"}:
        return target_status
    if coverage >= full_coverage_threshold:
        return "ASSIGNED_FULL"
    if coverage >= partial_coverage_threshold:
        return "ASSIGNED_PARTIAL"
    if target_status == "PARTIAL_TEXT_MATCH":
        return "TARGET_LOCALIZATION_UNCERTAIN"
    return "TARGET_LINKED_GEOMETRY_MISS"


def _assignment_status(
    target: LocatedTarget,
    coverage: dict[str, Any],
    registered: bool,
    *,
    full_coverage_threshold: float = FULL_COVERAGE_THRESHOLD,
    partial_coverage_threshold: float = PARTIAL_COVERAGE_THRESHOLD,
) -> str:
    return _assignment_status_from_values(
        target.status,
        float(coverage["coverage"]),
        registered,
        full_coverage_threshold=full_coverage_threshold,
        partial_coverage_threshold=partial_coverage_threshold,
    )


REGION_COLORS = (
    (50, 126, 36),
    (146, 92, 25),
    (126, 57, 126),
    (33, 121, 151),
    (92, 84, 188),
    (145, 110, 29),
)

RENDER_VERSION = "polygon-aware-collision-free-labels-3"
LabelBox = tuple[int, int, int, int]


def _boxes_overlap(first: LabelBox, second: LabelBox, gap: int = 2) -> bool:
    return not (
        first[2] + gap <= second[0]
        or second[2] + gap <= first[0]
        or first[3] + gap <= second[1]
        or second[3] + gap <= first[1]
    )


def _overlap_area(first: LabelBox, second: LabelBox) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _label_metrics(
    text: str, line_height: float
) -> tuple[float, int, int, int, int, int, int]:
    """Size a readable label below one measured text-line height."""

    maximum_height = max(12, int(np.floor(max(1.0, line_height) * 0.82)))
    thickness = 1
    padding_x = 3
    padding_y = 2
    scale = min(0.74, max(0.28, maximum_height / 34.0))
    while True:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        box_height = text_height + baseline + 2 * padding_y
        if box_height <= maximum_height or scale <= 0.28:
            break
        scale = max(0.28, scale - 0.02)
    return (
        scale,
        thickness,
        text_width + 2 * padding_x,
        min(box_height, maximum_height),
        text_height,
        baseline,
        padding_x,
    )


def _label_candidates(
    bounds: LabelBox,
    label_size: tuple[int, int],
    image_shape: tuple[int, ...],
    preferred_corner: int,
) -> list[LabelBox]:
    x1, y1, x2, y2 = bounds
    label_width, label_height = label_size
    image_height, image_width = image_shape[:2]
    inset = 2
    corners = (
        (x1 + inset, y1 + inset),
        (x2 - label_width - inset, y1 + inset),
        (x1 + inset, y2 - label_height - inset),
        (x2 - label_width - inset, y2 - label_height - inset),
    )
    ordered_corners = corners[preferred_corner % 4 :] + corners[: preferred_corner % 4]
    exterior = (
        (x1, y1 - label_height - inset),
        (x2 - label_width, y1 - label_height - inset),
        (x1, y2 + inset),
        (x2 - label_width, y2 + inset),
        (x1 - label_width - inset, y1),
        (x2 + inset, y1),
    )
    raw = list(ordered_corners) + list(exterior)
    # Extra lanes make collision freedom possible even for several nested boxes.
    for lane in range(1, 5):
        offset = lane * (label_height + inset)
        raw.extend(
            (
                (x1, y1 - label_height - inset - offset),
                (x2 - label_width, y1 - label_height - inset - offset),
                (x1, y2 + inset + offset),
                (x2 - label_width, y2 + inset + offset),
            )
        )

    candidates: list[LabelBox] = []
    seen: set[LabelBox] = set()
    for x, y in raw:
        x = max(0, min(int(round(x)), max(0, image_width - label_width)))
        y = max(0, min(int(round(y)), max(0, image_height - label_height)))
        candidate = (x, y, x + label_width, y + label_height)
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _draw_label(
    image: np.ndarray,
    text: str,
    bounds: LabelBox,
    color: tuple[int, int, int],
    *,
    line_height: float,
    occupied: list[LabelBox] | None = None,
    preferred_corner: int = 0,
    contour: np.ndarray | None = None,
) -> LabelBox:
    occupied = occupied if occupied is not None else []
    (
        scale,
        thickness,
        label_width,
        label_height,
        text_height,
        baseline,
        padding_x,
    ) = _label_metrics(text, line_height)
    candidates = _label_candidates(
        bounds,
        (label_width, label_height),
        image.shape,
        preferred_corner,
    )
    polygon_mask: np.ndarray | None = None
    if contour is not None and contour.size:
        polygon_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [contour.astype(np.int32)], 255)
        image_height, image_width = image.shape[:2]
        vertex_candidates: list[LabelBox] = []
        for point in contour.reshape(-1, 2):
            vertex_x, vertex_y = map(int, point)
            for x, y in (
                (vertex_x + 2, vertex_y + 2),
                (vertex_x - label_width - 2, vertex_y + 2),
                (vertex_x + 2, vertex_y - label_height - 2),
                (vertex_x - label_width - 2, vertex_y - label_height - 2),
            ):
                x = max(0, min(x, max(0, image_width - label_width)))
                y = max(0, min(y, max(0, image_height - label_height)))
                vertex_candidates.append(
                    (x, y, x + label_width, y + label_height)
                )
        distance = cv2.distanceTransform(polygon_mask, cv2.DIST_L2, 3)
        _, _, _, peak = cv2.minMaxLoc(distance)
        peak_x = max(
            0,
            min(
                int(peak[0] - label_width / 2),
                max(0, image_width - label_width),
            ),
        )
        peak_y = max(
            0,
            min(
                int(peak[1] - label_height / 2),
                max(0, image_height - label_height),
            ),
        )
        vertex_candidates.append(
            (peak_x, peak_y, peak_x + label_width, peak_y + label_height)
        )
        candidates = vertex_candidates + candidates

        def polygon_fit(candidate: LabelBox) -> tuple[bool, float]:
            x1, y1, x2, y2 = candidate
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            center_inside = polygon_mask[center[1], center[0]] > 0
            coverage = float(
                np.mean(polygon_mask[y1:y2, x1:x2] > 0)
            )
            return bool(center_inside), coverage

        unique: list[LabelBox] = []
        seen: set[LabelBox] = set()
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        candidates = sorted(
            unique,
            key=lambda candidate: (
                not polygon_fit(candidate)[0],
                -polygon_fit(candidate)[1],
            ),
        )
        interior = [
            candidate
            for candidate in candidates
            if polygon_fit(candidate)[0]
        ]
        if interior:
            candidates = interior
    selected = next(
        (
            candidate
            for candidate in candidates
            if not any(_boxes_overlap(candidate, prior) for prior in occupied)
        ),
        None,
    )
    if selected is None and contour is None:
        image_height, image_width = image.shape[:2]
        center_x = (bounds[0] + bounds[2]) / 2.0
        center_y = (bounds[1] + bounds[3]) / 2.0
        grid = [
            (x, y, x + label_width, y + label_height)
            for y in range(0, max(1, image_height - label_height + 1), label_height + 2)
            for x in range(0, max(1, image_width - label_width + 1), label_width + 2)
        ]
        grid.sort(
            key=lambda candidate: abs((candidate[0] + candidate[2]) / 2.0 - center_x)
            + abs((candidate[1] + candidate[3]) / 2.0 - center_y)
        )
        selected = next(
            (
                candidate
                for candidate in grid
                if not any(_boxes_overlap(candidate, prior) for prior in occupied)
            ),
            min(
                candidates,
                key=lambda candidate: sum(
                    _overlap_area(candidate, prior) for prior in occupied
                ),
            ),
        )
    if selected is None:
        # Every candidate remains anchored inside the component. If all
        # interior positions collide, choose the least-overlapping one rather
        # than moving the marker to unrelated page space.
        selected = min(
            candidates,
            key=lambda candidate: sum(
                _overlap_area(candidate, prior) for prior in occupied
            ),
        )
    x1, y1, x2, y2 = selected
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        -1,
    )
    baseline_y = min(y2 - baseline, y1 + 1 + text_height)
    cv2.putText(
        image,
        text,
        (x1 + padding_x, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    occupied.append(selected)
    return selected


def _coordinate_scale(
    image: np.ndarray, payload: dict[str, Any]
) -> tuple[float, float]:
    size = payload.get("image_size_wh") or [image.shape[1], image.shape[0]]
    width = max(1.0, float(size[0]))
    height = max(1.0, float(size[1]))
    return image.shape[1] / width, image.shape[0] / height


def _scaled_contour(
    polygon: Any, coordinate_scale: tuple[float, float]
) -> np.ndarray:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    contour[:, 0] *= coordinate_scale[0]
    contour[:, 1] *= coordinate_scale[1]
    return np.rint(contour).astype(np.int32)


def _contour_bounds(contour: np.ndarray) -> LabelBox:
    points = contour.reshape(-1, 2)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def _draw_regions(
    image: np.ndarray,
    payload: dict[str, Any],
    *,
    coordinate_scale: tuple[float, float] | None = None,
) -> tuple[list[LabelBox], float]:
    coordinate_scale = coordinate_scale or _coordinate_scale(image, payload)
    line_height = float(
        payload.get("detector_diagnostics", {}).get("estimated_text_line_height", 32)
    ) * float(coordinate_scale[1])
    overlay = image.copy()
    labels: list[
        tuple[str, LabelBox, tuple[int, int, int], np.ndarray]
    ] = []
    for region_index, region in enumerate(payload.get("redaction_regions", [])):
        color = REGION_COLORS[region_index % len(REGION_COLORS)]
        for component in region.get("components", []):
            contour = _scaled_contour(component["polygon_xy"], coordinate_scale)
            cv2.fillPoly(overlay, [contour], color, cv2.LINE_AA)
            labels.append(
                (
                    str(component["component_id"]),
                    _contour_bounds(contour),
                    color,
                    contour,
                )
            )
    cv2.addWeighted(overlay, 0.11, image, 0.89, 0, dst=image)
    for region_index, region in enumerate(payload.get("redaction_regions", [])):
        color = REGION_COLORS[region_index % len(REGION_COLORS)]
        for component in region.get("components", []):
            contour = _scaled_contour(component["polygon_xy"], coordinate_scale)
            width = max(1, int(round(4 * min(coordinate_scale))))
            cv2.polylines(image, [contour], True, color, width, cv2.LINE_AA)
    occupied: list[LabelBox] = []
    for index, (text, bounds, color, contour) in enumerate(labels):
        _draw_label(
            image,
            text,
            bounds,
            color,
            line_height=line_height,
            occupied=occupied,
            preferred_corner=index % 4,
            contour=contour,
        )
    return occupied, line_height


STATUS_COLORS = {
    "ASSIGNED_FULL": (24, 176, 239),
    "ASSIGNED_PARTIAL": (0, 118, 255),
    "TARGET_LINKED_GEOMETRY_MISS": (45, 45, 214),
    "TARGET_LOCALIZATION_UNCERTAIN": (125, 74, 181),
    "TARGET_NOT_IN_TEXT_LAYER": (95, 95, 95),
    "NO_TEXT_LAYER_OR_ANSWER": (95, 95, 95),
    "REGISTRATION_FAILED": (95, 95, 95),
}


def _draw_targets(
    image: np.ndarray,
    rows: list[dict[str, Any]],
    polygon_key: str,
    *,
    coordinate_scale: tuple[float, float] = (1.0, 1.0),
) -> None:
    for index, row in enumerate(rows, start=1):
        color = STATUS_COLORS.get(str(row["status"]), (0, 118, 255))
        exact_polygons = [
            _scaled_contour(polygon, coordinate_scale)
            for polygon in row.get(polygon_key, [])
            if len(polygon) >= 3
        ]
        if exact_polygons:
            overlay = image.copy()
            cv2.fillPoly(overlay, exact_polygons, color, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.19, image, 0.81, 0, dst=image)
        for contour in exact_polygons:
            width = max(1, int(round(2 * min(coordinate_scale))))
            cv2.polylines(image, [contour], True, color, width, cv2.LINE_AA)


def _draw_target_labels(
    image: np.ndarray,
    rows: list[dict[str, Any]],
    polygon_key: str,
    *,
    occupied: list[LabelBox],
    line_height: float,
    coordinate_scale: tuple[float, float] = (1.0, 1.0),
) -> None:
    for index, row in enumerate(rows, start=1):
        color = STATUS_COLORS.get(str(row["status"]), (0, 118, 255))
        exact_polygons = [
            _scaled_contour(polygon, coordinate_scale)
            for polygon in row.get(polygon_key, [])
            if len(polygon) >= 3
        ]
        if exact_polygons:
            points = np.concatenate(exact_polygons, axis=0)
            _draw_label(
                image,
                f"T{index}",
                _contour_bounds(points),
                color,
                line_height=line_height,
                occupied=occupied,
                preferred_corner=(index + 1) % 4,
            )


def _save_preview(image: np.ndarray, path: Path, max_dimension: int = 1800) -> None:
    scale = min(1.0, max_dimension / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 82])


def _compact_geometry(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(payload.get("detector_diagnostics", {}))
    return {
        "detector_version": payload.get("detector_version"),
        "source_path": payload.get("source_path"),
        "source_kind": payload.get("source_kind"),
        "page_no_1based": payload.get("page_no_1based"),
        "image_size_wh": payload.get("image_size_wh"),
        "coordinate_system": payload.get("coordinate_system"),
        "redaction_region_count": payload.get("redaction_region_count"),
        "physical_component_count": payload.get("physical_component_count"),
        "redaction_regions": payload.get("redaction_regions", []),
        "detector_diagnostics": diagnostics,
    }


def _refresh_saved_page(payload: dict[str, str]) -> str:
    output = Path(payload["output"])
    page = _read_json(Path(payload["page_record"]))
    for release, polygon_key in (
        ("earlier", "target_polygons_earlier"),
        ("later", "target_polygons_later"),
    ):
        raw_path = output / str(page[f"{release}_raw_asset"])
        asset_path = output / str(page[f"{release}_asset"])
        geometry_path = output / str(page[f"{release}_geometry"])
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read saved raw preview: {raw_path}")
        geometry = _read_json(geometry_path)
        coordinate_scale = _coordinate_scale(image, geometry)
        _draw_targets(
            image,
            page.get("fragments", []),
            polygon_key,
            coordinate_scale=coordinate_scale,
        )
        occupied, line_height = _draw_regions(
            image,
            geometry,
            coordinate_scale=coordinate_scale,
        )
        _draw_target_labels(
            image,
            page.get("fragments", []),
            polygon_key,
            occupied=occupied,
            line_height=line_height,
            coordinate_scale=coordinate_scale,
        )
        _save_preview(image, asset_path)
    return str(page["page_key"])


def refresh_render(output: Path, workers: int) -> dict[str, Any]:
    output = output.resolve()
    page_records = sorted((output / "page_records").glob("*.json"))
    if not page_records:
        raise SystemExit(f"No saved page records found under: {output}")
    payloads = [
        {"output": str(output), "page_record": str(path)} for path in page_records
    ]
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_refresh_saved_page, payload): payload
            for payload in payloads
        }
        with _progress(len(futures), description="Refresh v4 overlays") as progress:
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append(
                        {
                            "page_record": payload["page_record"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                progress.update(1)
    if errors:
        _write_json(output / "render_errors.json", errors)
        raise RuntimeError(f"Overlay refresh failed for {len(errors)} page(s)")
    render_errors = output / "render_errors.json"
    if render_errors.exists():
        render_errors.unlink()
    for path in (output / "run_config.json", output / "summary.json"):
        if path.exists():
            value = _read_json(path)
            value["render_version"] = RENDER_VERSION
            _write_json(path, value)
    result = {
        "page_pairs_refreshed": len(page_records),
        "render_version": RENDER_VERSION,
        "output": str(output),
    }
    print(json.dumps(result, indent=2))
    print(f"Open: {(output / 'index.html').as_uri()}")
    return result


def _run_page_job(payload: dict[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    job = PageJob(**payload["job"])
    output = Path(payload["output"])
    dpi = int(payload["dpi"])
    full_coverage_threshold = float(payload["full_coverage_threshold"])
    partial_coverage_threshold = float(payload["partial_coverage_threshold"])
    page_record = output / "page_records" / f"{job.key}.json"
    earlier_asset = output / "assets" / f"{job.key}_earlier.jpg"
    later_asset = output / "assets" / f"{job.key}_later.jpg"
    earlier_raw_asset = output / "assets" / f"{job.key}_earlier_raw.jpg"
    later_raw_asset = output / "assets" / f"{job.key}_later_raw.jpg"
    if payload["resume"] and all(
        path.is_file()
        for path in (
            page_record,
            earlier_asset,
            later_asset,
            earlier_raw_asset,
            later_raw_asset,
        )
    ):
        return json.loads(page_record.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix=f"box_v4_{job.key}_") as temporary:
        temp = Path(temporary)
        earlier_render = temp / "earlier.png"
        later_render = temp / "later.png"
        _render_page(Path(job.redacted_pdf), job.redacted_page, earlier_render, dpi)
        _render_page(Path(job.later_pdf), job.later_page, later_render, dpi)
        detector_output = temp / "detector"
        detector_payloads: dict[str, dict[str, Any]] = {}
        for release, source, rendered, page_number in (
            ("earlier", Path(job.redacted_pdf), earlier_render, job.redacted_page),
            ("later", Path(job.later_pdf), later_render, job.later_page),
        ):
            detector_payloads[release] = process_record(
                InputRecord(
                    item_key=f"{job.key}_{release}",
                    source_path=source,
                    rendered_image_path=rendered,
                    source_kind=f"v4_{release}_pdf",
                    page_no_1based=page_number,
                    pair_key=job.key,
                ),
                out_root=detector_output,
                save_debug_masks=False,
            )
        earlier_payload = detector_payloads["earlier"]
        later_payload = detector_payloads["later"]
        earlier_gray = cv2.imread(
            str(earlier_payload["output_files"]["source_png"]), cv2.IMREAD_GRAYSCALE
        )
        later_gray = cv2.imread(
            str(later_payload["output_files"]["source_png"]), cv2.IMREAD_GRAYSCALE
        )
        if earlier_gray is None or later_gray is None:
            raise RuntimeError(f"Detector did not emit source images for {job.key}")
        homography, registration = register_pages(earlier_gray, later_gray)
        words, page_size = _page_words(Path(job.later_pdf), job.later_page)
        fragment_results: list[dict[str, Any]] = []
        for fragment in job.fragments:
            target = locate_target(words, page_size, fragment)
            later_polygons: list[list[list[float]]] = []
            earlier_polygons: list[list[list[float]]] = []
            if homography is not None and target.status in {"PASS", "PARTIAL_TEXT_MATCH"}:
                later_polygons, earlier_polygons = _word_polygons(
                    target, later_payload, homography
                )
            coverage = target_coverage(earlier_polygons, earlier_payload)
            status = _assignment_status(
                target,
                coverage,
                homography is not None,
                full_coverage_threshold=full_coverage_threshold,
                partial_coverage_threshold=partial_coverage_threshold,
            )
            fragment_results.append(
                {
                    **fragment,
                    "page_key": job.key,
                    "status": status,
                    "expected_answer": fragment["answer"],
                    "located_text": target.matched_text,
                    "text_location_status": target.status,
                    "text_match_score": target.match_score,
                    "text_combined_score": target.combined_score,
                    "runner_up_margin": target.runner_up_margin,
                    "coverage": coverage["coverage"],
                    "covered_words": coverage["covered_words"],
                    "target_words": coverage["target_words"],
                    "matched_region_ids": coverage["regions"],
                    "matched_component_ids": coverage["components"],
                    "target_polygons_earlier": earlier_polygons,
                    "target_polygons_later": later_polygons,
                    "registration_status": registration["status"],
                }
            )
        earlier_display = cv2.cvtColor(earlier_gray, cv2.COLOR_GRAY2BGR)
        later_display = cv2.cvtColor(later_gray, cv2.COLOR_GRAY2BGR)
        _save_preview(earlier_gray, earlier_raw_asset)
        _save_preview(later_gray, later_raw_asset)
        _draw_targets(earlier_display, fragment_results, "target_polygons_earlier")
        _draw_targets(later_display, fragment_results, "target_polygons_later")
        # Detector components are the primary output, so their boundaries and
        # labels remain legible above the translucent post-hoc target layer.
        earlier_occupied, earlier_line_height = _draw_regions(
            earlier_display, earlier_payload
        )
        later_occupied, later_line_height = _draw_regions(
            later_display, later_payload
        )
        _draw_target_labels(
            earlier_display,
            fragment_results,
            "target_polygons_earlier",
            occupied=earlier_occupied,
            line_height=earlier_line_height,
        )
        _draw_target_labels(
            later_display,
            fragment_results,
            "target_polygons_later",
            occupied=later_occupied,
            line_height=later_line_height,
        )
        _save_preview(earlier_display, earlier_asset)
        _save_preview(later_display, later_asset)
        earlier_geometry = output / "geometry" / f"{job.key}_earlier.json"
        later_geometry = output / "geometry" / f"{job.key}_later.json"
        _write_json(earlier_geometry, _compact_geometry(earlier_payload))
        _write_json(later_geometry, _compact_geometry(later_payload))

    row = {
        "page_key": job.key,
        "doc_id": job.doc_id,
        "redacted_page": job.redacted_page,
        "later_page": job.later_page,
        "redacted_pdf": job.redacted_pdf,
        "later_pdf": job.later_pdf,
        "registration": registration,
        "earlier_region_count": earlier_payload["redaction_region_count"],
        "later_region_count": later_payload["redaction_region_count"],
        "earlier_component_count": earlier_payload["physical_component_count"],
        "later_component_count": later_payload["physical_component_count"],
            "earlier_asset": str(earlier_asset.relative_to(output)),
            "later_asset": str(later_asset.relative_to(output)),
            "earlier_raw_asset": str(earlier_raw_asset.relative_to(output)),
            "later_raw_asset": str(later_raw_asset.relative_to(output)),
        "earlier_geometry": str(earlier_geometry.relative_to(output)),
        "later_geometry": str(later_geometry.relative_to(output)),
        "full_coverage_threshold": full_coverage_threshold,
        "partial_coverage_threshold": partial_coverage_threshold,
        "fragments": fragment_results,
    }
    _write_json(page_record, row)
    return row


def _aggregate_items(
    items: list[dict[str, Any]], pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fragments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_links: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        for fragment in page.get("fragments", []):
            item_id = str(fragment["parent_item_id"])
            fragments[item_id].append(fragment)
            page_links[item_id].append(str(page["page_key"]))
    rows: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item["id"])
        expected_fragments = len(_passage_refs(item_id)[1])
        item_fragments = sorted(
            fragments.get(item_id, []), key=lambda row: int(row["fragment_index"])
        )
        statuses = [str(row["status"]) for row in item_fragments]
        total_words = sum(int(row.get("target_words", 0)) for row in item_fragments)
        covered_words = sum(int(row.get("covered_words", 0)) for row in item_fragments)
        aggregate_coverage = covered_words / total_words if total_words else 0.0
        if len(item_fragments) != expected_fragments:
            status = "INCOMPLETE_RUN"
        elif item_fragments and all(status == "ASSIGNED_FULL" for status in statuses):
            status = "ASSIGNED_FULL"
        elif any(status in {"ASSIGNED_FULL", "ASSIGNED_PARTIAL"} for status in statuses):
            status = "ASSIGNED_PARTIAL"
        elif "TARGET_LINKED_GEOMETRY_MISS" in statuses:
            status = "TARGET_LINKED_GEOMETRY_MISS"
        elif "TARGET_LOCALIZATION_UNCERTAIN" in statuses:
            status = "TARGET_LOCALIZATION_UNCERTAIN"
        elif "REGISTRATION_FAILED" in statuses:
            status = "REGISTRATION_FAILED"
        elif statuses:
            status = statuses[0]
        else:
            status = "NOT_PROCESSED"
        rows.append(
            {
                "item_id": item_id,
                "doc_id": item["doc_id_r"],
                "date": item.get("date"),
                "revealed": item.get("revealed"),
                "length_bucket": item.get("length_bucket"),
                "cross_page": bool(item.get("label_cross_page")),
                "answer": item.get("answer", ""),
                "status": status,
                "coverage": round(aggregate_coverage, 6),
                "covered_words": covered_words,
                "target_words": total_words,
                "fragment_count": len(item_fragments),
                "expected_fragment_count": expected_fragments,
                "fragment_statuses": statuses,
                "page_keys": sorted(set(page_links.get(item_id, []))),
            }
        )
    return rows


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _status_class(status: str) -> str:
    return re.sub(r"[^a-z]+", "-", status.lower()).strip("-")


def _short(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


CSS = """
:root{--paper:#f4f0e6;--ink:#14231f;--muted:#60706a;--line:#d5d0c4;--card:#fffdf7;--green:#1f8456;--gold:#c77d16;--red:#b8332a;--violet:#7950a6}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Gill Sans",sans-serif}.shell{max-width:1500px;margin:auto;padding:28px}h1,h2,h3{font-family:Charter,"Iowan Old Style",serif;letter-spacing:-.02em}h1{font-size:clamp(2rem,4vw,4rem);margin:.15em 0}.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:800;color:var(--green)}.lede{max-width:980px;color:var(--muted);font-size:1.05rem;line-height:1.55}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}.stat,.panel,.callout{background:var(--card);border:1px solid var(--line);border-radius:15px;box-shadow:0 6px 24px #293a2c0c}.stat{padding:16px}.stat strong{display:block;font:700 1.7rem Charter,serif}.controls{position:sticky;top:0;z-index:5;display:flex;gap:10px;flex-wrap:wrap;padding:12px;background:#f4f0e6eb;backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}input,select,button{background:#fff;border:1px solid #b9b4a9;border-radius:9px;padding:10px 12px;font:inherit}button{cursor:pointer;font-size:.78rem;font-weight:750}button.active{background:var(--ink);color:#fff;border-color:var(--ink)}input{min-width:320px;flex:1}.table-wrap{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:11px 13px;border-bottom:1px solid #e6e1d7;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#ebe5d8;z-index:2;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}a{color:#126644;text-decoration-thickness:1px;text-underline-offset:3px}.badge{display:inline-block;border-radius:99px;padding:4px 9px;font-size:.7rem;font-weight:800;letter-spacing:.04em;background:#ddd}.assigned-full{background:#d9f1df;color:#145b37}.assigned-partial{background:#f9e7bd;color:#82510d}.target-linked-geometry-miss{background:#f4d1cc;color:#86271f}.target-localization-uncertain{background:#eadcf5;color:#563573}.registration-failed,.target-not-in-text-layer,.no-text-layer-or-answer,.not-processed{background:#e1e1dd;color:#4f5753}.page-head{display:flex;justify-content:space-between;gap:20px;align-items:end}.viewer{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{padding:16px}.panel img{display:block;width:100%;height:auto;background:#ddd;border-radius:8px}.image-tools{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.explain-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}.callout{padding:15px}.callout b{display:block;margin-bottom:5px}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:.85rem;margin:12px 0}.swatch{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}.target-table td:nth-child(1){width:64px}.quote{font-family:Charter,"Iowan Old Style",serif;font-size:1rem;line-height:1.48}.context{color:var(--muted);line-height:1.45}.meta{display:flex;gap:8px;flex-wrap:wrap;color:var(--muted);font-size:.82rem}.nav{display:flex;gap:12px;margin:8px 0 20px}.small{font-size:.82rem;color:var(--muted)}code{font-family:"SFMono-Regular",Consolas,monospace;font-size:.84em}.nowrap{white-space:nowrap}@media(max-width:850px){.viewer,.explain-grid{grid-template-columns:1fr}.shell{padding:17px}input{min-width:100%}.page-head{align-items:start;flex-direction:column}}
"""


def _page_html(
    page: dict[str, Any], previous_key: str | None, next_key: str | None
) -> str:
    full_threshold = float(
        page.get("full_coverage_threshold", FULL_COVERAGE_THRESHOLD)
    )
    fragments: list[str] = []
    for index, row in enumerate(page["fragments"], start=1):
        left_context = _short(row.get("left_context", ""), 220)
        right_context = _short(row.get("right_context", ""), 220)
        component_ids = row.get("matched_component_ids", [])
        fragments.append(
            f"""<tr><td><b>T{index}</b><br><span class="badge {_status_class(row['status'])}">{_escape(row['status'])}</span></td>
<td><div class="quote">{_escape(row['expected_answer'])}</div><p class="context"><b>Before:</b> {_escape(left_context or 'not saved')}<br><b>After:</b> {_escape(right_context or 'not saved')}</p></td>
<td><div class="quote">{_escape(row['located_text'] or 'No reliable text-layer location')}</div><p class="small">Saved v4 text match: {float(row['text_match_score']):.1%}</p></td>
<td><b>{float(row['coverage']):.1%}</b> ({row['covered_words']}/{row['target_words']} words)<br><span class="small">CV components: {_escape(', '.join(component_ids) or 'none')}<br>Regions: {_escape(', '.join(row['matched_region_ids']) or 'none')}</span></td></tr>"""
        )
    nav = ["<a href='../../index.html'>Box results</a>", "<a href='../index.html'>V4 audit index</a>"]
    if previous_key:
        nav.append(f"<a href='{_escape(previous_key)}.html'>Previous page</a>")
    if next_key:
        nav.append(f"<a href='{_escape(next_key)}.html'>Next page</a>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(page['doc_id'])} - box audit</title><style>{CSS}</style></head><body><main class="shell">
<nav class="nav">{' / '.join(nav)}</nav><div class="page-head"><div><div class="eyebrow">Items v4 physical-source audit</div><h1>{_escape(page['doc_id'])}</h1><p class="lede">Earlier page {page['redacted_page']} is the more-redacted source; later page {page['later_page']} is the less-redacted comparison. The CV detector runs independently on each page. Saved v4 text is attached only after detection.</p></div><span class="badge {_status_class(page['page_status'])}">{_escape(page['page_status'])}</span></div>
<section class="explain-grid"><article class="callout"><b>1. CV geometry</b>Colored outlines and labels such as <code>R4.1</code> are physical components found from one page's pixels. The detector receives no answer text or paired page.</article><article class="callout"><b>2. V4/Astra target</b><code>T1</code> marks exact word boxes located in the later PDF text layer, then projected onto the earlier scan. These are audit annotations, not detector inputs.</article><article class="callout"><b>3. Assignment</b>Coverage is the fraction of localized target-word centers inside, or within a line-scaled tolerance of, detected CV geometry. Full requires at least {full_threshold:.0%} coverage for every fragment.</article></section>
<div class="legend"><span><i class="swatch" style="background:#327e24"></i>CV components (colors distinguish regions)</span><span><i class="swatch" style="background:#efb018"></i>full target words</span><span><i class="swatch" style="background:#ff7600"></i>partial target words</span><span><i class="swatch" style="background:#d62d2d"></i>geometry miss</span></div>
<section class="viewer"><article class="panel"><h2>Earlier release</h2><p class="small">More redacted - source page {page['redacted_page']}</p><div class="image-tools"><button class="active" data-view="overlay" data-image="earlier">Detection + target</button><button data-view="raw" data-image="earlier">Original pixels</button></div><a id="earlier-link" href="../{_escape(page['earlier_asset'])}"><img id="earlier-image" src="../{_escape(page['earlier_asset'])}" data-overlay="../{_escape(page['earlier_asset'])}" data-raw="../{_escape(page['earlier_raw_asset'])}" loading="lazy" alt="Earlier release"></a><p><a href="../{_escape(page['earlier_geometry'])}">Earlier geometry JSON</a></p></article>
<article class="panel"><h2>Later release</h2><p class="small">Less redacted - source page {page['later_page']}</p><div class="image-tools"><button class="active" data-view="overlay" data-image="later">Detection + target</button><button data-view="raw" data-image="later">Original pixels</button></div><a id="later-link" href="../{_escape(page['later_asset'])}"><img id="later-image" src="../{_escape(page['later_asset'])}" data-overlay="../{_escape(page['later_asset'])}" data-raw="../{_escape(page['later_raw_asset'])}" loading="lazy" alt="Later release"></a><p><a href="../{_escape(page['later_geometry'])}">Later geometry JSON</a></p></article></section>
<section><h2>Target provenance and assignment</h2><div class="table-wrap"><table class="target-table"><thead><tr><th>Target</th><th>Saved v4 target and context</th><th>Located later-PDF text</th><th>CV coverage</th></tr></thead><tbody>{''.join(fragments)}</tbody></table></div></section>
<p class="small">Registration: {_escape(page['registration'].get('status'))}; inliers: {_escape(page['registration'].get('inliers'))}; median reprojection error: {_escape(page['registration'].get('median_reprojection_error'))}. Source files: <code>{_escape(Path(page['redacted_pdf']).name)}</code> / <code>{_escape(Path(page['later_pdf']).name)}</code>.</p></main>
<script>document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{{const id=b.dataset.image,img=document.getElementById(id+'-image'),link=document.getElementById(id+'-link');document.querySelectorAll('[data-image="'+id+'"]').forEach(x=>x.classList.toggle('active',x===b));const src=img.dataset[b.dataset.view];img.src=src;link.href=src}}));</script></body></html>"""


def _index_html(summary: dict[str, Any], items: list[dict[str, Any]]) -> str:
    rows = []
    for row in items:
        first_page = row["page_keys"][0] if row["page_keys"] else ""
        link = f"pages/{first_page}.html" if first_page else "#"
        search = " ".join(
            [str(row["item_id"]), str(row["doc_id"]), str(row["answer"]), str(row["status"])]
        ).lower()
        rows.append(
            f"<tr data-status='{_escape(row['status'])}' data-revealed='{_escape(row['revealed'])}' data-search='{_escape(search)}'><td><a href='{_escape(link)}'>{_escape(row['item_id'])}</a><br><span class='small'>{_escape(row['date'])}</span></td><td><span class='badge {_status_class(row['status'])}'>{_escape(row['status'])}</span></td><td>{float(row['coverage']):.1%}<br><span class='small'>{row['covered_words']}/{row['target_words']} located words</span></td><td>{_escape(row['revealed'])}<br><span class='small'>{_escape(row['length_bucket'])}</span></td><td>{_escape(_short(row['answer']))}</td></tr>"
        )
    counts = summary["item_status_counts"]
    full_threshold = float(
        summary.get("full_coverage_threshold", FULL_COVERAGE_THRESHOLD)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Items v4 - Box CV audit</title><style>{CSS}</style></head><body><main class="shell"><nav class="nav"><a href="../index.html">Box results</a> / <a href="../manual_validation/index.html">Manual validation</a></nav><div class="eyebrow">Redaction Extract / detector {DETECTOR_VERSION}</div><h1>Items v4 box audit</h1><p class="lede">Every frozen v4 redaction is traced back to its earlier and later PDF pages. The CV detector sees one page at a time and does not receive answers, OCR, document IDs, paired-release pixels, or benchmark labels. Saved v4 text is located as exact word boxes in the later PDF text layer and attached afterward to measure geometric coverage.</p>
<section class="stats"><div class="stat"><strong>{summary['items_processed']:,}</strong>v4 items</div><div class="stat"><strong>{summary['page_pairs_processed']:,}</strong>page pairs</div><div class="stat"><strong>{counts.get('ASSIGNED_FULL',0):,}</strong>fully assigned</div><div class="stat"><strong>{counts.get('ASSIGNED_PARTIAL',0):,}</strong>partial</div><div class="stat"><strong>{summary['registration_pass']:,}</strong>registrations passed</div><div class="stat"><strong>{summary['elapsed_seconds']/60:.1f}</strong>minutes</div></section>
<section class="panel" style="padding:16px;margin-bottom:18px"><b>Interpretation.</b> Full means every page fragment of an item achieved at least {full_threshold:.0%} exact target-word-center coverage. Partial means at least one fragment had meaningful coverage but the complete item did not pass. A geometry miss is not proof that the v4 text is wrong; open the paired scans to distinguish detector, text-localization, and scan-registration failures.</section>
<div class="controls"><input id="query" placeholder="Search document, answer, status..."><select id="status"><option value="">All statuses</option>{''.join(f'<option>{_escape(key)}</option>' for key in sorted(counts))}</select><select id="revealed"><option value="">All reveal types</option><option>fully</option><option>partly</option></select><span id="visible" class="small"></span></div>
<div class="table-wrap"><table><thead><tr><th>Item</th><th>Status</th><th>Coverage</th><th>Type</th><th>Recovered target</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="small">Machine-readable outputs: <a href="summary.json">summary.json</a> / <a href="item_results.csv">item_results.csv</a> / <a href="fragment_results.csv">fragment_results.csv</a> / <a href="page_results.jsonl">page_results.jsonl</a>.</p></main>
<script>const q=document.getElementById('query'),s=document.getElementById('status'),r=document.getElementById('revealed'),v=document.getElementById('visible');function f(){{let n=0;document.querySelectorAll('tbody tr').forEach(x=>{{const ok=(!q.value||x.dataset.search.includes(q.value.toLowerCase()))&&(!s.value||x.dataset.status===s.value)&&(!r.value||x.dataset.revealed===r.value);x.hidden=!ok;if(ok)n++}});v.textContent=n.toLocaleString()+' visible'}}[q,s,r].forEach(x=>x.addEventListener('input',f));f();</script></body></html>"""


def _progress(total: int, description: str = "Items v4 Box CV") -> Any:
    try:
        from tqdm import tqdm
    except ImportError:
        class Fallback:
            def __enter__(self) -> "Fallback": return self
            def __exit__(self, *_: Any) -> None: return None
            def update(self, _: int = 1) -> None: return None
            def set_postfix_str(self, _: str) -> None: return None
        return Fallback()
    return tqdm(total=total, desc=description, unit="page-pair", dynamic_ncols=True)


def _prepare_output(
    output: Path,
    config: dict[str, Any],
    *,
    overwrite: bool,
    resume: bool,
) -> None:
    config_path = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if overwrite:
            shutil.rmtree(output)
        elif not resume:
            raise SystemExit(f"Output exists; pass --resume or --overwrite: {output}")
        elif not config_path.is_file():
            raise SystemExit(f"Cannot resume without {config_path}")
        else:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != config:
                raise SystemExit("Resume configuration differs from the existing run")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, config)


def run(args: argparse.Namespace) -> dict[str, Any]:
    items_path = args.items.resolve()
    astra_path = args.astra.resolve()
    docs_root = args.docs_root.resolve()
    output = args.out.resolve()
    jobs, items, scope = build_scope(items_path, astra_path, docs_root)
    if args.max_page_pairs is not None:
        jobs = jobs[: max(0, args.max_page_pairs)]
    config = {
        "audit_version": AUDIT_VERSION,
        "render_version": RENDER_VERSION,
        "detector_version": DETECTOR_VERSION,
        "detector_code_sha256": _code_sha256(),
        "items": str(items_path),
        "items_sha256": _sha256(items_path),
        "astra": str(astra_path),
        "astra_sha256": _sha256(astra_path),
        "docs_root": str(docs_root),
        "dpi": args.dpi,
        "full_coverage_threshold": args.full_coverage_threshold,
        "partial_coverage_threshold": PARTIAL_COVERAGE_THRESHOLD,
        "selected_page_pairs": len(jobs),
    }
    if args.plan:
        print(json.dumps({"scope": scope, "run": config}, indent=2))
        return {"scope": scope, "run": config}
    _prepare_output(output, config, overwrite=args.overwrite, resume=args.resume)
    started = time.perf_counter()
    payloads = [
        {
            "job": asdict(job),
            "output": str(output),
            "dpi": args.dpi,
            "full_coverage_threshold": args.full_coverage_threshold,
            "partial_coverage_threshold": PARTIAL_COVERAGE_THRESHOLD,
            "resume": args.resume,
        }
        for job in jobs
    ]
    pages: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_page_job, payload): payload for payload in payloads}
        with _progress(len(futures)) as progress:
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    page = future.result()
                    page["page_status"] = (
                        "ASSIGNED_FULL"
                        if page["fragments"]
                        and all(row["status"] == "ASSIGNED_FULL" for row in page["fragments"])
                        else "ASSIGNED_PARTIAL"
                        if any(
                            row["status"] in {"ASSIGNED_FULL", "ASSIGNED_PARTIAL"}
                            for row in page["fragments"]
                        )
                        else page["fragments"][0]["status"]
                        if page["fragments"]
                        else "NOT_PROCESSED"
                    )
                    pages.append(page)
                    progress.set_postfix_str(page["page_status"])
                except Exception as exc:
                    errors.append(
                        {
                            "page_key": payload["job"]["key"],
                            "doc_id": payload["job"]["doc_id"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    progress.set_postfix_str("error")
                progress.update(1)
    pages.sort(key=lambda row: (row["doc_id"], row["redacted_page"], row["later_page"]))
    processed_parent_ids = {
        str(fragment["parent_item_id"])
        for page in pages
        for fragment in page.get("fragments", [])
    }
    selected_items = [item for item in items if str(item["id"]) in processed_parent_ids]
    item_results = _aggregate_items(selected_items, pages)
    fragment_results = [fragment for page in pages for fragment in page["fragments"]]
    page_dir = output / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(pages):
        previous_key = pages[index - 1]["page_key"] if index else None
        next_key = pages[index + 1]["page_key"] if index + 1 < len(pages) else None
        (page_dir / f"{page['page_key']}.html").write_text(
            _page_html(page, previous_key, next_key), encoding="utf-8"
        )
    elapsed = time.perf_counter() - started
    summary = {
        **scope,
        "items_processed": len(item_results),
        "fragments_processed": len(fragment_results),
        "page_pairs_processed": len(pages),
        "page_pair_errors": len(errors),
        "registration_pass": sum(
            page.get("registration", {}).get("status") == "PASS" for page in pages
        ),
        "item_status_counts": dict(
            sorted(Counter(row["status"] for row in item_results).items())
        ),
        "fragment_status_counts": dict(
            sorted(Counter(row["status"] for row in fragment_results).items())
        ),
        "elapsed_seconds": round(elapsed, 3),
        "detector_version": DETECTOR_VERSION,
        "audit_version": AUDIT_VERSION,
        "render_version": RENDER_VERSION,
        "full_coverage_threshold": args.full_coverage_threshold,
        "partial_coverage_threshold": PARTIAL_COVERAGE_THRESHOLD,
        "complete_scope": (
            len(jobs) == scope["unique_page_pairs"]
            and len(pages) == len(jobs)
            and not errors
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "errors.json", errors)
    _write_jsonl(output / "page_results.jsonl", pages)
    _write_csv(output / "item_results.csv", item_results)
    _write_csv(output / "fragment_results.csv", fragment_results)
    (output / "index.html").write_text(
        _index_html(summary, item_results), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Open: {(output / 'index.html').as_uri()}")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    result.add_argument("--astra", type=Path, default=DEFAULT_ASTRA)
    result.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS)
    result.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--dpi", type=int, choices=[200], default=200)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument(
        "--full-coverage-threshold",
        type=float,
        default=FULL_COVERAGE_THRESHOLD,
        help=(
            "Minimum per-fragment target-word coverage for ASSIGNED_FULL "
            f"(default: {FULL_COVERAGE_THRESHOLD:.2f})."
        ),
    )
    result.add_argument("--max-page-pairs", type=int)
    result.add_argument("--plan", action="store_true")
    result.add_argument(
        "--refresh-render",
        action="store_true",
        help="Redraw saved overlays from existing raw previews and geometry only.",
    )
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least one")
    if not PARTIAL_COVERAGE_THRESHOLD < args.full_coverage_threshold <= 1.0:
        raise SystemExit(
            "--full-coverage-threshold must be greater than 0.30 and at most 1.0"
        )
    if args.refresh_render:
        if args.plan or args.overwrite or args.resume or args.max_page_pairs is not None:
            raise SystemExit(
                "--refresh-render cannot be combined with --plan, --overwrite, "
                "--resume, or --max-page-pairs"
            )
        refresh_render(args.out, args.workers)
        return
    run(args)


if __name__ == "__main__":
    main()
