from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


@dataclass
class PDFPair:
    pair_key: str
    unredacted_numeric_id: str
    redacted_doc_id: str
    row: list[str]
    row_index_1based: int
    redacted_pdf: Path
    unredacted_pdf: Path


@dataclass
class PairCollectionStats:
    manifest_schema: str = "unknown"
    rows_total: int = 0
    rows_missing_required_fields: int = 0
    rows_invalid_unredacted_id_format: int = 0
    rows_missing_redacted_pdf: int = 0
    rows_missing_unredacted_pdf: int = 0
    rows_duplicate_unredacted_id: int = 0
    rows_duplicate_redacted_id: int = 0
    rows_valid: int = 0


@dataclass
class InputRecord:
    item_key: str
    source_path: Path
    rendered_image_path: Path
    source_kind: str
    page_no_1based: int | None
    pair_key: str | None


@dataclass(frozen=True)
class BoxComponent:
    """One geometric component contributing to a redaction region."""

    box: tuple[int, int, int, int]
    polygon: tuple[tuple[int, int], ...]
    source: str
    score: float


@dataclass(frozen=True)
class RedactionRegion:
    """A semantic redaction region made from touching/overlapping components."""

    components: tuple[BoxComponent, ...]

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (
            min(component.box[0] for component in self.components),
            min(component.box[1] for component in self.components),
            max(component.box[2] for component in self.components),
            max(component.box[3] for component in self.components),
        )


def _progress(iterable: Any, *, total: int | None, desc: str) -> Any:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.2)


def _clean_filename_component(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "x"


def _normalize_csv_row(row: list[str], width: int = 11) -> list[str]:
    out = list(row)
    if len(out) < width:
        out.extend([""] * (width - len(out)))
    return out


def _parse_unredacted_id(raw: str) -> int | None:
    s = str(raw).strip()
    if not s or not s.isdigit():
        return None
    if len(s) > 8:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _canonical_unredacted_pdf_path(unredacted_id_int: int, unredacted_dir: Path) -> Path:
    return unredacted_dir / f"cib_{unredacted_id_int:08d}.pdf"


def collect_pdf_pairs_with_stats(
    csv_path: Path,
    redacted_dir: Path,
    unredacted_dir: Path,
    max_pairs: int | None = None,
) -> tuple[list[PDFPair], PairCollectionStats]:
    pairs: list[PDFPair] = []
    stats = PairCollectionStats()
    seen_unred_ids: set[str] = set()
    seen_red_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return pairs, stats

        normalized_header = {str(value).strip().lower(): idx for idx, value in enumerate(first_row)}
        is_headered = {"record_id", "document_id"}.issubset(normalized_header)
        if is_headered:
            stats.manifest_schema = "headered"
            record_idx = normalized_header["record_id"]
            document_idx = normalized_header["document_id"]
            source_rows = enumerate(reader, start=2)
        else:
            stats.manifest_schema = "legacy_headerless"
            record_idx = 1
            document_idx = 5
            source_rows = enumerate(chain([first_row], reader), start=1)

        for i, raw_row in source_rows:
            stats.rows_total += 1
            row = _normalize_csv_row(raw_row)
            unred_id_raw = str(row[record_idx]).strip() if record_idx < len(row) else ""
            red_doc_id = str(row[document_idx]).strip() if document_idx < len(row) else ""

            if not unred_id_raw or not red_doc_id:
                stats.rows_missing_required_fields += 1
                continue

            unred_id_int = _parse_unredacted_id(unred_id_raw)
            if unred_id_int is None:
                stats.rows_invalid_unredacted_id_format += 1
                continue

            unred_id_norm = str(unred_id_int)
            if unred_id_norm in seen_unred_ids:
                stats.rows_duplicate_unredacted_id += 1
                continue
            if red_doc_id in seen_red_ids:
                stats.rows_duplicate_redacted_id += 1
                continue

            red_pdf = redacted_dir / f"{red_doc_id}.pdf"
            if not red_pdf.exists():
                stats.rows_missing_redacted_pdf += 1
                continue

            unred_pdf = _canonical_unredacted_pdf_path(unred_id_int, unredacted_dir)
            if not unred_pdf.exists():
                stats.rows_missing_unredacted_pdf += 1
                continue

            seen_unred_ids.add(unred_id_norm)
            seen_red_ids.add(red_doc_id)
            pair_key = _clean_filename_component(f"{unred_id_norm}_{red_doc_id}")
            pairs.append(
                PDFPair(
                    pair_key=pair_key,
                    unredacted_numeric_id=unred_id_norm,
                    redacted_doc_id=red_doc_id,
                    row=row,
                    row_index_1based=i,
                    redacted_pdf=red_pdf,
                    unredacted_pdf=unred_pdf,
                )
            )
            stats.rows_valid += 1
            if max_pairs is not None and len(pairs) >= max_pairs:
                break

    return pairs, stats


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _clip_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(1, min(width, int(x2))),
        max(1, min(height, int(y2))),
    )


def _rect_polygon(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    x1, y1, x2, y2 = box
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _edge_density(
    edges: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 3
) -> float:
    xa, ya, xb, yb = x1 + pad, y1 + pad, x2 - pad, y2 - pad
    if xb <= xa or yb <= ya:
        return 1.0
    roi = edges[ya:yb, xa:xb]
    return float(np.mean(roi > 0)) if roi.size else 1.0


def _binarize_dark(gray: np.ndarray) -> np.ndarray:
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hard_inv = ((gray < 170).astype(np.uint8) * 255)
    return cv2.bitwise_or(otsu_inv, hard_inv)


def _binarize_faint(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expose low-contrast gray outlines without lowering the global dark cutoff."""
    h, w = gray.shape
    tile = max(4, min(16, round(min(h, w) / 180)))
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile)).apply(gray)
    block = max(21, min(71, (min(h, w) // 10) | 1))
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        9,
    )
    return enhanced, adaptive


def _estimate_skew_degrees(gray: np.ndarray) -> float:
    """Estimate modest scan skew from long near-horizontal structures."""
    h, w = gray.shape
    work = gray
    if w > 1200:
        scale = 1200.0 / w
        work = cv2.resize(
            gray,
            (1200, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    wh, ww = work.shape
    edges = cv2.Canny(work, 45, 140)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=max(25, ww // 20),
        minLineLength=max(40, ww // 4),
        maxLineGap=max(5, ww // 80),
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        angle = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        if abs(angle) <= 4:
            angles.append(angle)
    if len(angles) < 2:
        return 0.0
    return float(np.median(np.asarray(angles, dtype=np.float32)))


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Deskew for proposal extraction and return a map back to original pixels."""
    angle = _estimate_skew_degrees(gray)
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if abs(angle) < 0.25 or abs(angle) > 3.0:
        return gray, identity, 0.0
    h, w = gray.shape
    center = (w / 2.0, h / 2.0)
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for rotation in (-angle, angle):
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        rotated = cv2.warpAffine(
            gray,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        residual = abs(_estimate_skew_degrees(rotated))
        candidates.append((residual, rotated, matrix, rotation))
    _, rotated, matrix, rotation = min(candidates, key=lambda value: value[0])
    return rotated, cv2.invertAffineTransform(matrix), float(rotation)


def _map_component(
    component: BoxComponent, inverse: np.ndarray, width: int, height: int
) -> BoxComponent:
    points = np.asarray(component.polygon, dtype=np.float32).reshape((-1, 1, 2))
    mapped = cv2.transform(points, inverse).reshape((-1, 2))
    polygon = tuple(
        (
            max(0, min(width - 1, int(round(x)))),
            max(0, min(height - 1, int(round(y)))),
        )
        for x, y in mapped
    )
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    box = _clip_box((min(xs), min(ys), max(xs) + 1, max(ys) + 1), width, height)
    return BoxComponent(box=box, polygon=polygon, source=component.source, score=component.score)


def _inner_region(
    x1: int, y1: int, x2: int, y2: int, pad: int
) -> tuple[int, int, int, int]:
    return x1 + pad, y1 + pad, x2 - pad, y2 - pad


def _interior_stats(
    gray: np.ndarray,
    dark: np.ndarray,
    box: tuple[int, int, int, int],
    pad: int = 3,
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    xa, ya, xb, yb = _inner_region(x1, y1, x2, y2, pad)
    if xb <= xa or yb <= ya:
        return 0.0, 1.0
    interior_gray = gray[ya:yb, xa:xb]
    interior_dark = dark[ya:yb, xa:xb]
    if interior_gray.size == 0:
        return 0.0, 1.0
    return float(interior_gray.mean()), float(np.mean(interior_dark > 0))


def _band_support(mask: np.ndarray, box: tuple[int, int, int, int], tolerance: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    h, w = mask.shape
    x1, y1, x2, y2 = _clip_box(box, w, h)
    t = max(1, tolerance)
    top = mask[max(0, y1 - t):min(h, y1 + t + 1), x1:x2]
    bottom = mask[max(0, y2 - 1 - t):min(h, y2 + t), x1:x2]
    left = mask[y1:y2, max(0, x1 - t):min(w, x1 + t + 1)]
    right = mask[y1:y2, max(0, x2 - 1 - t):min(w, x2 + t)]

    def horizontal_support(band: np.ndarray) -> float:
        return float(np.mean(np.any(band > 0, axis=0))) if band.size else 0.0

    def vertical_support(band: np.ndarray) -> float:
        return float(np.mean(np.any(band > 0, axis=1))) if band.size else 0.0

    return (
        horizontal_support(top),
        horizontal_support(bottom),
        vertical_support(left),
        vertical_support(right),
    )


def _multi_scale_lines(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    horizontal_lengths = sorted(
        {max(18, round(w * ratio)) for ratio in (0.025, 0.06, 0.12)}
    )
    vertical_lengths = sorted(
        {max(6, round(h * ratio)) for ratio in (0.006, 0.015, 0.04)}
    )
    horizontal = np.zeros_like(mask)
    vertical = np.zeros_like(mask)
    for length in horizontal_lengths:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
        horizontal = cv2.bitwise_or(
            horizontal, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        )
    for length in vertical_lengths:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        vertical = cv2.bitwise_or(
            vertical, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        )
    return horizontal, vertical


def _outline_score(
    gray: np.ndarray,
    dark: np.ndarray,
    edges: np.ndarray,
    line_mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> float | None:
    image_h, image_w = gray.shape
    x1, y1, x2, y2 = _clip_box(box, image_w, image_h)
    width = x2 - x1
    height = y2 - y1
    if width < 20 or height < 7 or width * height < 160:
        return None
    interior_mean, interior_dark = _interior_stats(gray, dark, (x1, y1, x2, y2), pad=3)
    if interior_mean < 188 or interior_dark > 0.24:
        return None
    edge_density = _edge_density(edges, x1, y1, x2, y2, pad=3)
    if edge_density > 0.18:
        return None
    tolerance = max(2, round(min(image_h, image_w) * 0.0025))
    supports = _band_support(line_mask, (x1, y1, x2, y2), tolerance)
    present = [value >= 0.16 for value in supports]
    margin = max(3, round(min(image_h, image_w) * 0.015))
    boundary = [y1 <= margin, y2 >= image_h - margin, x1 <= margin, x2 >= image_w - margin]
    effective = [seen or edge for seen, edge in zip(present, boundary)]
    horizontal_pair = effective[0] and effective[1]
    vertical_pair = effective[2] and effective[3]
    if sum(effective) < 3 or sum(present) < 2:
        return None
    if not (horizontal_pair or vertical_pair):
        return None
    missing_internal = sum(not seen and not edge for seen, edge in zip(present, boundary))
    if missing_internal and width * height < image_w * image_h * 0.02:
        return None
    blank_score = min(1.0, max(0.0, (interior_mean - 188.0) / 55.0))
    return float(sum(supports) + blank_score + (1.0 - min(1.0, edge_density / 0.18)))


def _contour_polygon(contour: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return _rect_polygon(box)
    approx = cv2.approxPolyDP(contour, max(1.0, 0.012 * perimeter), True).reshape((-1, 2))
    if 4 <= len(approx) <= 20:
        return tuple((int(x), int(y)) for x, y in approx)
    return _rect_polygon(box)


def _outline_candidates(
    gray: np.ndarray,
    dark: np.ndarray,
    edges: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> list[BoxComponent]:
    line_mask = cv2.bitwise_or(horizontal, vertical)
    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    candidates: list[BoxComponent] = []
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        box = (x, y, x + width, y + height)
        score = _outline_score(gray, dark, edges, line_mask, box)
        if score is None:
            continue
        candidates.append(
            BoxComponent(
                box=box,
                polygon=_contour_polygon(contour, box),
                source="multiscale_outline_contour",
                score=score,
            )
        )
    return candidates


def _solid_candidates(gray: np.ndarray) -> tuple[list[BoxComponent], np.ndarray]:
    """Find filled redactions from thick dark cores, not merely dark blobs.

    Ordinary words often occupy half of their bounding rectangle, so fill ratio
    alone is not enough.  A redaction bar also contains a long uninterrupted
    horizontal or vertical core.  We use that core as a seed and recover only
    the connected dark component around it.
    """
    image_h, image_w = gray.shape
    raw_solid = ((gray < 72).astype(np.uint8) * 255)
    solid = cv2.morphologyEx(
        raw_solid,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    horizontal_length = max(18, round(image_w * 0.012))
    horizontal_height = max(2, round(image_h * 0.0015))
    vertical_width = max(2, round(image_w * 0.0015))
    vertical_length = max(14, round(image_h * 0.008))
    horizontal_seed = cv2.morphologyEx(
        raw_solid,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_length, horizontal_height)
        ),
        iterations=1,
    )
    vertical_seed = cv2.morphologyEx(
        raw_solid,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (vertical_width, vertical_length)),
        iterations=1,
    )
    core_seed = cv2.bitwise_or(horizontal_seed, vertical_seed)
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[BoxComponent] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        box = (x, y, x + width, y + height)
        area = _area(box)
        minimum_area = max(180, round(image_w * image_h * 0.00006))
        minimum_height = max(7, round(image_h * 0.006))
        if (
            width < max(20, round(height * 1.25))
            or height < minimum_height
            or area < minimum_area
        ):
            continue
        roi = solid[y:y + height, x:x + width]
        seed_roi = core_seed[y:y + height, x:x + width]
        if not seed_roi.size or not np.any(seed_roi > 0):
            continue
        dark_fraction = float(np.mean(roi > 0)) if roi.size else 0.0
        contour_fraction = float(cv2.contourArea(contour) / area) if area else 0.0
        seed_fraction = float(np.mean(seed_roi > 0))
        row_occupancy = np.mean(roi > 0, axis=1)
        coherent_rows = float(np.mean(row_occupancy >= 0.72))
        if dark_fraction < 0.68 or contour_fraction < 0.60:
            continue
        if seed_fraction < 0.08 or coherent_rows < 0.30:
            continue
        score = (
            3.0
            + dark_fraction
            + contour_fraction
            + seed_fraction
            + coherent_rows
            + min(1.0, area / 5000.0)
        )
        candidates.append(
            BoxComponent(
                box=box,
                polygon=_contour_polygon(contour, box),
                source="solid_fill",
                score=score,
            )
        )
    return candidates, core_seed


def _near_duplicate(a: BoxComponent, b: BoxComponent, tolerance: int) -> bool:
    if _iou(a.box, b.box) >= 0.86:
        return True
    return max(abs(av - bv) for av, bv in zip(a.box, b.box)) <= tolerance


def _contains_box(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    margin: int = 3,
) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (
        ox1 <= ix1 + margin
        and oy1 <= iy1 + margin
        and ox2 >= ix2 - margin
        and oy2 >= iy2 - margin
    )


def _prune_outline_artifacts(candidates: list[BoxComponent]) -> list[BoxComponent]:
    """Remove intersection slivers and synthetic envelopes around real parts.

    Connected line masks can emit both the physical rectangles and (a) their
    small overlap or (b) one loose contour enclosing the complete stair-step.
    The former is an intersection artifact.  The latter adds no geometry when
    two or more children already span almost its full width and height.  A
    genuinely larger overlapping redaction is retained when it extends well
    beyond its contained components.
    """
    if len(candidates) < 3:
        return candidates

    remove: set[int] = set()
    for index, candidate in enumerate(candidates):
        containers = [
            other
            for other_index, other in enumerate(candidates)
            if other_index != index
            and _area(other.box) > _area(candidate.box) * 1.15
            and _contains_box(other.box, candidate.box)
        ]
        if len(containers) >= 2:
            remove.add(index)

    for index, candidate in enumerate(candidates):
        if index in remove:
            continue
        children = [
            other
            for other_index, other in enumerate(candidates)
            if other_index != index
            and other_index not in remove
            and _area(candidate.box) > _area(other.box) * 1.15
            and _contains_box(candidate.box, other.box)
        ]
        if len(children) < 2:
            continue
        cx1 = min(child.box[0] for child in children)
        cy1 = min(child.box[1] for child in children)
        cx2 = max(child.box[2] for child in children)
        cy2 = max(child.box[3] for child in children)
        x1, y1, x2, y2 = candidate.box
        width_coverage = (cx2 - cx1) / max(1, x2 - x1)
        height_coverage = (cy2 - cy1) / max(1, y2 - y1)
        if width_coverage >= 0.72 and height_coverage >= 0.72:
            remove.add(index)

    return [candidate for index, candidate in enumerate(candidates) if index not in remove]


def _dedupe_components(
    candidates: list[BoxComponent], image_shape: tuple[int, int]
) -> list[BoxComponent]:
    tolerance = max(3, round(min(image_shape) * 0.004))
    ordered = sorted(candidates, key=lambda item: (item.score, _area(item.box)), reverse=True)
    kept: list[BoxComponent] = []
    for candidate in ordered:
        if any(_near_duplicate(candidate, previous, tolerance) for previous in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.box[1], item.box[0], -_area(item.box)))


def _components_connected(a: BoxComponent, b: BoxComponent, tolerance: int) -> bool:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0, max(ay1, by1) - min(ay2, by2))
    if _iou(a.box, b.box) > 0:
        return True
    if x_gap <= tolerance and y_overlap >= 0.20 * min(ay2 - ay1, by2 - by1):
        return True
    if y_gap <= tolerance and x_overlap >= 0.20 * min(ax2 - ax1, bx2 - bx1):
        return True
    return False


def _group_components(
    components: list[BoxComponent], image_shape: tuple[int, int]
) -> list[RedactionRegion]:
    if not components:
        return []
    tolerance = max(2, round(min(image_shape) * 0.0025))
    parents = list(range(len(components)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            if _components_connected(components[left], components[right], tolerance):
                union(left, right)
    grouped: dict[int, list[BoxComponent]] = {}
    for index, component in enumerate(components):
        grouped.setdefault(find(index), []).append(component)
    regions = [
        RedactionRegion(
            components=tuple(sorted(values, key=lambda item: (item.box[1], item.box[0])))
        )
        for values in grouped.values()
    ]
    return sorted(regions, key=lambda region: (region.box[1], region.box[0]))


def detect_redaction_regions_with_artifacts(
    gray: np.ndarray,
) -> tuple[list[RedactionRegion], dict[str, np.ndarray], dict[str, Any]]:
    original_h, original_w = gray.shape
    work_gray, inverse, rotation = _deskew(gray)
    dark = _binarize_dark(work_gray)
    enhanced, faint = _binarize_faint(work_gray)
    strong_h, strong_v = _multi_scale_lines(dark)
    faint_h, faint_v = _multi_scale_lines(faint)
    horizontal = cv2.bitwise_or(strong_h, faint_h)
    vertical = cv2.bitwise_or(strong_v, faint_v)
    edges = cv2.Canny(enhanced, 25, 100)
    outline = _outline_candidates(work_gray, dark, edges, horizontal, vertical)
    outline = _prune_outline_artifacts(outline)
    solid, solid_mask = _solid_candidates(work_gray)
    work_components = _dedupe_components(outline + solid, work_gray.shape)
    mapped = [
        _map_component(component, inverse, original_w, original_h)
        for component in work_components
    ]
    components = _dedupe_components(mapped, gray.shape)
    regions = _group_components(components, gray.shape)
    line_mask = cv2.bitwise_or(horizontal, vertical)
    return regions, {
        "dark_mask": dark,
        "faint_mask": faint,
        "horizontal_lines": horizontal,
        "vertical_lines": vertical,
        "lines_mask": line_mask,
        "solid_mask": solid_mask,
        "edges": edges,
    }, {
        "estimated_rotation_degrees": rotation,
        "outline_candidate_count": len(outline),
        "solid_candidate_count": len(solid),
        "component_count_before_grouping": len(components),
        "region_count": len(regions),
    }


def detect_redaction_boxes_with_artifacts(
    gray: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], dict[str, np.ndarray]]:
    """Compatibility API: return grouped region bounds rather than raw components."""
    regions, artifacts, _ = detect_redaction_regions_with_artifacts(gray)
    return [region.box for region in regions], artifacts


def render_pdf_to_images(pdf_path: Path, out_dir: Path, prefix: str, dpi: int, max_pages: int | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF rendering backend missing. Install pypdfium2.") from exc

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(doc)
        if max_pages is not None:
            page_count = min(page_count, max(0, int(max_pages)))
        images: list[Path] = []
        scale = max(0.1, float(dpi) / 72.0)
        for page_index in range(page_count):
            page = doc.get_page(page_index)
            try:
                bmp = page.render(scale=scale)
                pil = bmp.to_pil()
                out_path = out_dir / f"{prefix}_p{page_index + 1:04d}.png"
                pil.save(out_path)
                images.append(out_path)
            finally:
                page.close()
        return images
    finally:
        doc.close()


def iter_input_records(
    *,
    input_path: Path | None,
    docs_root: Path | None,
    docs_subdir: str,
    out_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    source_kind: str,
    dpi: int,
    max_files: int | None,
    max_pages: int | None,
) -> tuple[list[InputRecord], dict[str, Any]]:
    records: list[InputRecord] = []
    stats: dict[str, Any] = {"mode": None}
    render_root = out_root / "_rendered_inputs"
    render_root.mkdir(parents=True, exist_ok=True)

    if input_path is not None:
        stats["mode"] = "input_path"
        files: list[Path] = []
        if input_path.is_file():
            files = [input_path]
        elif input_path.is_dir():
            files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS))
        else:
            raise SystemExit(f"Input path not found: {input_path}")
        if max_files is not None:
            files = files[: max(0, int(max_files))]

        for src in files:
            suffix = src.suffix.lower()
            if suffix in IMAGE_EXTS:
                item_key = _clean_filename_component(src.stem)
                records.append(InputRecord(item_key=item_key, source_path=src, rendered_image_path=src, source_kind="image", page_no_1based=None, pair_key=None))
            elif suffix in PDF_EXTS:
                pdf_render_dir = render_root / _clean_filename_component(src.stem)
                pages = render_pdf_to_images(src, pdf_render_dir, _clean_filename_component(src.stem), dpi=dpi, max_pages=max_pages)
                for page_idx, image_path in enumerate(pages, start=1):
                    item_key = _clean_filename_component(f"{src.stem}_p{page_idx:04d}")
                    records.append(InputRecord(item_key=item_key, source_path=src, rendered_image_path=image_path, source_kind="pdf", page_no_1based=page_idx, pair_key=None))
        stats["input_file_count"] = len(files)
        stats["record_count"] = len(records)
        return records, stats

    if docs_root is None:
        raise SystemExit("Provide either --input or --docs_root.")

    stats["mode"] = "docs_root"
    pairs, pair_stats = collect_pdf_pairs_with_stats(
        csv_path=docs_root / csv_name,
        redacted_dir=docs_root / redacted_dir_name,
        unredacted_dir=docs_root / unredacted_dir_name,
        max_pairs=None,
    )
    stats["pair_stats"] = pair_stats.__dict__
    if max_files is not None:
        pairs = pairs[: max(0, int(max_files))]

    for pair in pairs:
        if source_kind in {"redacted", "both"}:
            pdf_path = pair.redacted_pdf
            pdf_render_dir = render_root / pair.pair_key / "redacted"
            pages = render_pdf_to_images(pdf_path, pdf_render_dir, f"{pair.pair_key}_redacted", dpi=dpi, max_pages=max_pages)
            for page_idx, image_path in enumerate(pages, start=1):
                item_key = _clean_filename_component(f"{pair.pair_key}_redacted_p{page_idx:04d}")
                records.append(InputRecord(item_key=item_key, source_path=pdf_path, rendered_image_path=image_path, source_kind="redacted_pdf", page_no_1based=page_idx, pair_key=pair.pair_key))
        if source_kind in {"unredacted", "both"}:
            pdf_path = pair.unredacted_pdf
            pdf_render_dir = render_root / pair.pair_key / "unredacted"
            pages = render_pdf_to_images(pdf_path, pdf_render_dir, f"{pair.pair_key}_unredacted", dpi=dpi, max_pages=max_pages)
            for page_idx, image_path in enumerate(pages, start=1):
                item_key = _clean_filename_component(f"{pair.pair_key}_unredacted_p{page_idx:04d}")
                records.append(InputRecord(item_key=item_key, source_path=pdf_path, rendered_image_path=image_path, source_kind="unredacted_pdf", page_no_1based=page_idx, pair_key=pair.pair_key))

    stats["record_count"] = len(records)
    return records, stats


def process_record(record: InputRecord, *, out_root: Path, save_debug_masks: bool) -> dict[str, Any]:
    gray = cv2.imread(str(record.rendered_image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {record.rendered_image_path}")

    regions, artifacts, diagnostics = detect_redaction_regions_with_artifacts(gray)
    item_dir = out_root / record.item_key
    item_dir.mkdir(parents=True, exist_ok=True)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    palette = [(0, 180, 0), (220, 90, 0), (180, 0, 180), (0, 140, 220), (160, 120, 0)]
    for region_index, region in enumerate(regions, start=1):
        color = palette[(region_index - 1) % len(palette)]
        for component_index, component in enumerate(region.components, start=1):
            polygon = np.asarray(component.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [polygon], True, color, 2, cv2.LINE_AA)
            x1, y1, _, _ = component.box
            cv2.putText(
                overlay,
                f"G{region_index}.{component_index}",
                (x1 + 2, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    source_copy_path = item_dir / f"{record.item_key}.source.png"
    overlay_path = item_dir / f"{record.item_key}.redaction_boxes.png"
    metadata_path = item_dir / f"{record.item_key}.redaction_boxes.json"
    cv2.imwrite(str(source_copy_path), gray)
    cv2.imwrite(str(overlay_path), overlay)

    debug_paths: dict[str, str] = {}
    if save_debug_masks:
        debug_dir = item_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for key, mask in artifacts.items():
            out_path = debug_dir / f"{key}.png"
            cv2.imwrite(str(out_path), mask)
            debug_paths[key] = str(out_path)

    payload = {
        "item_key": record.item_key,
        "source_path": str(record.source_path),
        "rendered_image_path": str(record.rendered_image_path),
        "source_kind": record.source_kind,
        "pair_key": record.pair_key,
        "page_no_1based": record.page_no_1based,
        "image_size_wh": [int(gray.shape[1]), int(gray.shape[0])],
        "redaction_box_count": len(regions),
        "redaction_region_count": len(regions),
        "physical_component_count": sum(len(region.components) for region in regions),
        "redaction_boxes_xyxy": [list(map(int, region.box)) for region in regions],
        "redaction_regions": [
            {
                "region_id": f"G{region_index}",
                "bounds_xyxy": list(map(int, region.box)),
                "components": [
                    {
                        "component_id": f"G{region_index}.{component_index}",
                        "bounds_xyxy": list(map(int, component.box)),
                        "polygon_xy": [list(map(int, point)) for point in component.polygon],
                        "proposal_source": component.source,
                        "proposal_score": round(float(component.score), 6),
                    }
                    for component_index, component in enumerate(region.components, start=1)
                ],
            }
            for region_index, region in enumerate(regions, start=1)
        ],
        "detector_diagnostics": diagnostics,
        "output_files": {
            "source_png": str(source_copy_path),
            "overlay_png": str(overlay_path),
            "metadata_json": str(metadata_path),
            "debug_masks": debug_paths,
        },
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_box_pipeline(
    *,
    out_root: Path,
    input_path: Path | None = None,
    docs_root: Path | None = None,
    docs_subdir: str = "redacted_pdfs",
    source_kind: str = "redacted",
    csv_name: str = "cibcia.csv",
    redacted_dir_name: str = "redacted_pdfs",
    unredacted_dir_name: str = "unredacted_pdfs",
    dpi: int = 300,
    max_files: int | None = None,
    max_pages: int | None = None,
    save_debug_masks: bool = False,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    records, stats = iter_input_records(
        input_path=input_path,
        docs_root=docs_root,
        docs_subdir=docs_subdir,
        out_root=out_root,
        csv_name=csv_name,
        redacted_dir_name=redacted_dir_name,
        unredacted_dir_name=unredacted_dir_name,
        source_kind=source_kind,
        dpi=dpi,
        max_files=max_files,
        max_pages=max_pages,
    )
    manifest_rows: list[dict[str, Any]] = []
    for record in _progress(records, total=len(records), desc="Detect redaction boxes"):
        try:
            manifest_rows.append(process_record(record, out_root=out_root, save_debug_masks=save_debug_masks))
        except Exception as exc:
            manifest_rows.append({
                "item_key": record.item_key,
                "source_path": str(record.source_path),
                "rendered_image_path": str(record.rendered_image_path),
                "error": str(exc),
            })

    manifest_json = out_root / "box_manifest.json"
    manifest_jsonl = out_root / "box_manifest.jsonl"
    summary_json = out_root / "box_summary.json"
    manifest_json.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    success_count = sum(1 for row in manifest_rows if "redaction_box_count" in row)
    error_count = len(manifest_rows) - success_count
    summary = {
        "stats": stats,
        "record_count": len(records),
        "success_count": success_count,
        "error_count": error_count,
        "manifest_json": str(manifest_json),
        "manifest_jsonl": str(manifest_jsonl),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
