from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from . import _core_detection as core_stage


# Corpus pairing and input iteration are independent of detector geometry.
PDFPair = core_stage.PDFPair
PairCollectionStats = core_stage.PairCollectionStats
InputRecord = core_stage.InputRecord
BoxComponent = core_stage.BoxComponent
RedactionRegion = core_stage.RedactionRegion
collect_pdf_pairs_with_stats = core_stage.collect_pdf_pairs_with_stats
render_pdf_to_images = core_stage.render_pdf_to_images
iter_input_records = core_stage.iter_input_records


@dataclass(frozen=True)
class AxisSegment:
    orientation: str
    position: int
    start: int
    end: int
    source: str

    @property
    def length(self) -> int:
        return max(0, self.end - self.start + 1)


@dataclass(frozen=True)
class RectangleProposal:
    component: BoxComponent
    observed_sides: tuple[bool, bool, bool, bool]
    virtual_sides: tuple[bool, bool, bool, bool]
    zone_id: int


@dataclass(frozen=True)
class OrientedSegment:
    """A locally straight border segment retained before axis rectification."""

    orientation: str
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    angle_degrees: float

    @property
    def midpoint(self) -> tuple[float, float]:
        return (
            (self.start[0] + self.end[0]) / 2.0,
            (self.start[1] + self.end[1]) / 2.0,
        )


def _rect_polygon(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    x1, y1, x2, y2 = box
    return ((x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1))


def _polygon_box(
    polygon: tuple[tuple[int, int], ...]
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _simplify_polygon(
    contour: np.ndarray, image_shape: tuple[int, int], line_height: int
) -> tuple[tuple[int, int], ...]:
    """Return a stable outline without replacing a notch by its envelope."""

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return tuple()
    epsilon = max(1.0, min(line_height * 0.16, perimeter * 0.006))
    points = cv2.approxPolyDP(contour, epsilon, True).reshape((-1, 2))
    height, width = image_shape
    polygon: list[tuple[int, int]] = []
    for raw_x, raw_y in points:
        point = (
            max(0, min(width - 1, int(raw_x))),
            max(0, min(height - 1, int(raw_y))),
        )
        if not polygon or point != polygon[-1]:
            polygon.append(point)
    if len(polygon) > 2 and polygon[0] == polygon[-1]:
        polygon.pop()
    return tuple(polygon) if 4 <= len(polygon) <= 20 else tuple()


def _area(box: tuple[int, int, int, int]) -> int:
    return core_stage._area(box)


def _interval_union_length(intervals: Iterable[tuple[int, int]], gap: int = 0) -> tuple[int, int]:
    ordered = sorted((int(a), int(b)) for a, b in intervals if b >= a)
    if not ordered:
        return 0, 0
    total = 0
    longest = 0
    left, right = ordered[0]
    for next_left, next_right in ordered[1:]:
        if next_left <= right + gap + 1:
            right = max(right, next_right)
            continue
        length = right - left + 1
        total += length
        longest = max(longest, length)
        left, right = next_left, next_right
    length = right - left + 1
    return total + length, max(longest, length)


def _axis_open(mask: np.ndarray, orientation: str) -> np.ndarray:
    h, w = mask.shape
    if orientation == "h":
        lengths = sorted({max(18, round(w * ratio)) for ratio in (0.025, 0.055, 0.12)})
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 240), 1))
    else:
        lengths = sorted({max(8, round(h * ratio)) for ratio in (0.010, 0.025, 0.06)})
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 240)))
    out = np.zeros_like(mask)
    for length in lengths:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (length, 1) if orientation == "h" else (1, length)
        )
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        out = cv2.bitwise_or(out, opened)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, close_kernel, iterations=1)


def _snap_edge(value: int, limit: int, margin: int) -> int:
    if value <= margin:
        return 0
    if value >= limit - 1 - margin:
        return limit - 1
    return int(value)


def _segments_from_mask(
    mask: np.ndarray,
    orientation: str,
    source: str,
) -> list[AxisSegment]:
    h, w = mask.shape
    edge_margin = max(3, round(min(h, w) * 0.018))
    minimum = max(18, round(w * 0.025)) if orientation == "h" else max(8, round(h * 0.010))
    maximum_thickness = max(5, round((h if orientation == "h" else w) * 0.008))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segments: list[AxisSegment] = []
    for label in range(1, count):
        x, y, width, height, _ = map(int, stats[label])
        major = width if orientation == "h" else height
        thickness = height if orientation == "h" else width
        if major < minimum or thickness > maximum_thickness:
            continue
        if orientation == "h":
            position = int(round(float(centroids[label][1])))
            start = _snap_edge(x, w, edge_margin)
            end = _snap_edge(x + width - 1, w, edge_margin)
        else:
            position = int(round(float(centroids[label][0])))
            start = _snap_edge(y, h, edge_margin)
            end = _snap_edge(y + height - 1, h, edge_margin)
        segments.append(AxisSegment(orientation, position, start, end, source))
    return segments


def _merge_segments(
    segments: list[AxisSegment], image_shape: tuple[int, int]
) -> list[AxisSegment]:
    if not segments:
        return []
    position_tolerance = max(1, round(min(image_shape) * 0.0025))
    gap_tolerance = max(3, round(min(image_shape) * 0.008))
    remaining = sorted(segments, key=lambda item: (item.position, item.start, item.end))
    merged: list[AxisSegment] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        changed = True
        while changed:
            changed = False
            low = min(item.start for item in group)
            high = max(item.end for item in group)
            center = int(round(np.median([item.position for item in group])))
            keep: list[AxisSegment] = []
            for item in remaining:
                if (
                    item.orientation == seed.orientation
                    and abs(item.position - center) <= position_tolerance
                    and item.start <= high + gap_tolerance
                    and item.end >= low - gap_tolerance
                ):
                    group.append(item)
                    changed = True
                else:
                    keep.append(item)
            remaining = keep
        source = "strong" if any(item.source == "strong" for item in group) else "faint"
        merged.append(
            AxisSegment(
                seed.orientation,
                int(round(np.median([item.position for item in group]))),
                min(item.start for item in group),
                max(item.end for item in group),
                source,
            )
        )
    return sorted(merged, key=lambda item: (item.position, item.start, item.end))


def _extract_axis_segments(
    dark: np.ndarray, faint: np.ndarray
) -> tuple[list[AxisSegment], list[AxisSegment], np.ndarray, np.ndarray]:
    strong_h = _axis_open(dark, "h")
    strong_v = _axis_open(dark, "v")
    faint_h = _axis_open(faint, "h")
    faint_v = _axis_open(faint, "v")
    h_segments = _merge_segments(
        _segments_from_mask(strong_h, "h", "strong")
        + _segments_from_mask(faint_h, "h", "faint"),
        dark.shape,
    )
    v_segments = _merge_segments(
        _segments_from_mask(strong_v, "v", "strong")
        + _segments_from_mask(faint_v, "v", "faint"),
        dark.shape,
    )
    horizontal = np.zeros_like(dark)
    vertical = np.zeros_like(dark)
    for segment in h_segments:
        cv2.line(
            horizontal,
            (segment.start, segment.position),
            (segment.end, segment.position),
            255,
            2,
        )
    for segment in v_segments:
        cv2.line(
            vertical,
            (segment.position, segment.start),
            (segment.position, segment.end),
            255,
            2,
        )
    return h_segments, v_segments, horizontal, vertical


def _line_zones(horizontal: np.ndarray, vertical: np.ndarray) -> list[tuple[int, int, int, int]]:
    line_mask = cv2.bitwise_or(horizontal, vertical)
    radius = max(2, round(min(line_mask.shape) * 0.005))
    joined = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1)),
        iterations=1,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones: list[tuple[int, int, int, int]] = []
    h, w = line_mask.shape
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        pad = radius + 2
        zones.append((max(0, x - pad), max(0, y - pad), min(w, x + width + pad), min(h, y + height + pad)))
    return sorted(zones, key=lambda value: (value[1], value[0]))


def _segment_intersects_zone(segment: AxisSegment, zone: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = zone
    if segment.orientation == "h":
        return y1 <= segment.position <= y2 and segment.end >= x1 and segment.start <= x2
    return x1 <= segment.position <= x2 and segment.end >= y1 and segment.start <= y2


def _line_support(
    segments: list[AxisSegment],
    position: int,
    start: int,
    end: int,
    position_tolerance: int,
    gap_tolerance: int,
) -> tuple[float, float, bool]:
    length = max(1, end - start + 1)
    intervals: list[tuple[int, int]] = []
    strong = False
    for segment in segments:
        if abs(segment.position - position) > position_tolerance:
            continue
        left = max(start, segment.start)
        right = min(end, segment.end)
        if right < left:
            continue
        intervals.append((left, right))
        strong = strong or segment.source == "strong"
    total, longest = _interval_union_length(intervals, gap=gap_tolerance)
    return min(1.0, total / length), min(1.0, longest / length), strong


def _segment_reaches(
    segments: list[AxisSegment],
    position: int,
    endpoint: int,
    position_tolerance: int,
    endpoint_tolerance: int,
) -> bool:
    return any(
        abs(segment.position - position) <= position_tolerance
        and segment.start - endpoint_tolerance <= endpoint <= segment.end + endpoint_tolerance
        for segment in segments
    )


def _text_line_height(gray: np.ndarray, line_mask: np.ndarray) -> int:
    ink = ((gray < 175).astype(np.uint8) * 255)
    ink[line_mask > 0] = 0
    projection = np.count_nonzero(ink, axis=1)

    # Scanned full pages often have a dark frame or edge noise on every row.
    # A fixed near-zero threshold then turns the whole page into one apparent
    # text band and makes every legitimate box look too short. Estimate that
    # persistent background first, then require enough additional horizontal
    # ink to identify a real line of text.
    persistent_ink = float(np.percentile(projection, 45))
    threshold = max(
        3,
        int(round(persistent_ink + max(3.0, gray.shape[1] * 0.008))),
    )
    bands: list[int] = []
    start: int | None = None
    maximum_band_height = max(12, round(gray.shape[0] * 0.08))
    for index, value in enumerate(projection):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            if 3 <= index - start <= maximum_band_height:
                bands.append(index - start)
            start = None
    if start is not None and 3 <= len(projection) - start <= maximum_band_height:
        bands.append(len(projection) - start)
    if not bands:
        return max(10, round(min(gray.shape) * 0.025))

    measured = float(np.median(bands)) * 1.20
    scale_floor = min(gray.shape) * 0.018
    scale_ceiling = min(gray.shape) * 0.055
    return max(7, int(round(min(scale_ceiling, max(scale_floor, measured)))))


def _interior_quality(
    gray: np.ndarray,
    edges: np.ndarray,
    box: tuple[int, int, int, int],
    line_height: int,
) -> tuple[float, float, float] | None:
    x1, y1, x2, y2 = box
    pad = max(2, min(line_height // 4, max(2, (y2 - y1) // 5)))
    if x2 - x1 <= 2 * pad or y2 - y1 <= 2 * pad:
        return None
    interior = gray[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    interior_edges = edges[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    if interior.size == 0:
        return None
    mean = float(np.mean(interior))
    dark_fraction = float(np.mean(interior < 175))
    edge_fraction = float(np.mean(interior_edges > 0))
    if mean < 184 or dark_fraction > 0.20 or edge_fraction > 0.15:
        return None
    return mean, dark_fraction, edge_fraction


def _visible_text_row_occupancy(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    line_height: int,
) -> tuple[float, int]:
    """Measure word-like ink rows inside a proposed outline.

    Mean darkness is a weak test for a large enclosure: several lines of text
    can occupy little of its total area. This test instead asks how much of
    the interior height contains horizontally distributed ink. Border rows
    are removed, and a one-row closing tolerates scan breaks inside glyphs.
    The measure is scale-relative and uses no OCR or case data.
    """

    x1, y1, x2, y2 = box
    height = y2 - y1
    pad = max(2, min(4, max(2, height // 7)))
    if x2 - x1 <= 2 * pad or height <= 2 * pad:
        return 0.0, 0
    interior = gray[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    if interior.size == 0:
        return 0.0, 0

    row_density = np.mean(interior < 175, axis=1)
    active = (row_density > 0.08).astype(np.uint8).reshape((-1, 1))
    if active.shape[0] >= 3:
        active = cv2.morphologyEx(
            active,
            cv2.MORPH_CLOSE,
            np.ones((3, 1), dtype=np.uint8),
            iterations=1,
        )
    values = active.reshape(-1).astype(bool)
    longest = 0
    run = 0
    for value in values:
        if value:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return float(np.mean(values)) if values.size else 0.0, longest


def _enumerate_zone_rectangles(
    gray: np.ndarray,
    edges: np.ndarray,
    horizontal_segments: list[AxisSegment],
    vertical_segments: list[AxisSegment],
    zone: tuple[int, int, int, int],
    zone_id: int,
    line_height: int,
) -> list[RectangleProposal]:
    h, w = gray.shape
    hs = [segment for segment in horizontal_segments if _segment_intersects_zone(segment, zone)]
    vs = [segment for segment in vertical_segments if _segment_intersects_zone(segment, zone)]
    if not hs or not vs:
        return []
    edge_margin = max(3, round(min(h, w) * 0.018))
    y_coords = {segment.position for segment in hs}
    if any(segment.start == 0 for segment in vs) or any(segment.position == 0 for segment in hs):
        y_coords.add(0)
    if any(segment.end == h - 1 for segment in vs) or any(segment.position == h - 1 for segment in hs):
        y_coords.add(h - 1)

    # A line that terminates within the crop-edge tolerance invokes that edge.
    if any(segment.start <= edge_margin for segment in vs):
        y_coords.add(0)
    if any(segment.end >= h - 1 - edge_margin for segment in vs):
        y_coords.add(h - 1)

    y_values = sorted(y_coords)
    position_tolerance = max(2, round(min(h, w) * 0.004))
    gap_tolerance = max(3, round(min(h, w) * 0.008))
    minimum_height = max(8, round(line_height * 1.10))
    minimum_width = max(22, round(line_height * 1.65))
    proposals: list[RectangleProposal] = []

    for y1_index, y1 in enumerate(y_values):
        for y2 in y_values[y1_index + 1:]:
            height = y2 - y1
            if height < minimum_height:
                continue

            # Only vertical runs that span this particular horizontal pair can
            # become sides. This prevents thousands of glyph stems elsewhere
            # in a connected text zone from entering the rectangle search.
            x_coords = set()
            for segment in vs:
                overlap = max(0, min(segment.end, y2) - max(segment.start, y1) + 1)
                if overlap >= height * 0.55:
                    x_coords.add(segment.position)
            top_lines = [segment for segment in hs if abs(segment.position - y1) <= position_tolerance]
            bottom_lines = [segment for segment in hs if abs(segment.position - y2) <= position_tolerance]
            if any(segment.start <= edge_margin for segment in top_lines + bottom_lines):
                x_coords.add(0)
            if any(segment.end >= w - 1 - edge_margin for segment in top_lines + bottom_lines):
                x_coords.add(w - 1)
            x_values = sorted(x_coords)
            for x1_index, x1 in enumerate(x_values):
                for x2 in x_values[x1_index + 1:]:
                    width = x2 - x1
                    if width < minimum_width or width < height * 0.70:
                        continue
                    top = _line_support(hs, y1, x1, x2, position_tolerance, gap_tolerance)
                    bottom = _line_support(hs, y2, x1, x2, position_tolerance, gap_tolerance)
                    left = _line_support(vs, x1, y1, y2, position_tolerance, gap_tolerance)
                    right = _line_support(vs, x2, y1, y2, position_tolerance, gap_tolerance)
                    side_values = (top, bottom, left, right)
                    observed = tuple(
                        coverage >= (0.57 if strong else 0.67)
                        and longest >= (0.46 if strong else 0.55)
                        for coverage, longest, strong in side_values
                    )
                    boundary = (y1 == 0, y2 == h - 1, x1 == 0, x2 == w - 1)
                    virtual = tuple(edge and not seen for edge, seen in zip(boundary, observed))
                    effective = tuple(seen or edge for seen, edge in zip(observed, boundary))
                    if not all(effective):
                        continue
                    if sum(observed) < 2:
                        continue
                    if not (observed[0] or observed[1]) or not (observed[2] or observed[3]):
                        continue
                    # Two observed sides must be perpendicular when two sides are virtual.
                    if sum(virtual) >= 2 and not any(
                        observed[h_side] and observed[v_side]
                        for h_side in (0, 1)
                        for v_side in (2, 3)
                    ):
                        continue
                    # A short, narrow outline cut off by the bottom of a crop
                    # is usually a paragraph bracket or furniture fragment,
                    # not a redaction. Large page-edge masks remain eligible.
                    if (
                        virtual[1]
                        and height < line_height * 2.5
                        and width < w * 0.35
                    ):
                        continue
                    near_bottom = y2 >= h - 1 - edge_margin
                    near_right = x2 >= w - 1 - edge_margin
                    if (
                        near_bottom
                        and near_right
                        and height < line_height * 2.5
                        and width < w * 0.35
                    ):
                        continue
                    # Coverage alone is insufficient: a text stem beyond a real
                    # horizontal endpoint must not enlarge the rectangle. Every
                    # observed/virtual corner must be reached by both adjacent
                    # sides in the rectilinear line graph.
                    corner_tolerance = gap_tolerance + position_tolerance
                    corner_checks = (
                        (y1, x1, 0, 2),
                        (y1, x2, 0, 3),
                        (y2, x1, 1, 2),
                        (y2, x2, 1, 3),
                    )
                    corners_valid = True
                    for y_corner, x_corner, h_side, v_side in corner_checks:
                        h_reaches = boundary[h_side] or _segment_reaches(
                            hs, y_corner, x_corner, position_tolerance, corner_tolerance
                        )
                        v_reaches = boundary[v_side] or _segment_reaches(
                            vs, x_corner, y_corner, position_tolerance, corner_tolerance
                        )
                        if not (h_reaches and v_reaches):
                            corners_valid = False
                            break
                    if not corners_valid:
                        continue
                    box = (x1, y1, x2 + 1, y2 + 1)
                    quality = _interior_quality(gray, edges, box, line_height)
                    if quality is None:
                        continue
                    # A crop edge may replace a genuinely missing box side, but
                    # it can also combine with a title underline and tall glyphs
                    # to manufacture an apparent rectangle. Because a virtual
                    # side has no observed ink, require stronger interior
                    # blankness for page-edge-completed candidates.
                    if any(virtual):
                        occupied_rows, longest_text_run = _visible_text_row_occupancy(
                            gray, box, line_height
                        )
                        if (
                            occupied_rows > 0.30
                            and longest_text_run >= max(5, round(line_height * 0.35))
                        ):
                            continue
                    mean, dark_fraction, edge_fraction = quality
                    if (
                        boundary[0]
                        and width < line_height * 3.25
                        and (dark_fraction > 0.08 or edge_fraction > 0.08)
                    ):
                        continue
                    side_score = sum(coverage + longest for coverage, longest, _ in side_values)
                    score = (
                        side_score
                        + sum(observed) * 0.8
                        - sum(virtual) * 0.20
                        + min(1.0, max(0.0, (mean - 184.0) / 60.0))
                        + (1.0 - min(1.0, dark_fraction / 0.20))
                        + (1.0 - min(1.0, edge_fraction / 0.15))
                    )
                    proposals.append(
                        RectangleProposal(
                            component=BoxComponent(
                                box=box,
                                polygon=_rect_polygon(box),
                                source=(
                                    f"rectilinear_zone_{zone_id}:observed={sum(observed)}:"
                                    f"virtual={sum(virtual)}"
                                ),
                                score=float(score),
                            ),
                            observed_sides=observed,
                            virtual_sides=virtual,
                            zone_id=zone_id,
                        )
                    )
    return proposals


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], margin: int = 2) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 + margin and oy1 <= iy1 + margin and ox2 >= ix2 - margin and oy2 >= iy2 - margin


def _near_same_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int], tolerance: int) -> bool:
    if core_stage._iou(a, b) >= 0.78:
        return True
    if core_stage._iou(a, b) >= 0.62:
        aligned_sides = sum(abs(left - right) <= tolerance for left, right in zip(a, b))
        if aligned_sides >= 3:
            return True
    return max(abs(left - right) for left, right in zip(a, b)) <= tolerance


def _compact_maximal_cover(
    proposals: list[RectangleProposal], image_shape: tuple[int, int]
) -> list[BoxComponent]:
    tolerance = max(2, round(min(image_shape) * 0.004))
    ordered = sorted(
        proposals,
        key=lambda item: (
            sum(item.observed_sides),
            -sum(item.virtual_sides),
            _area(item.component.box),
            item.component.score,
        ),
        reverse=True,
    )
    unique: list[RectangleProposal] = []
    for proposal in ordered:
        if any(_near_same_box(proposal.component.box, prior.component.box, tolerance) for prior in unique):
            continue
        unique.append(proposal)

    maximal: list[RectangleProposal] = []
    for proposal in unique:
        # A virtual crop-edge completion must not replace a nearly identical
        # fully observed rectangle just because it is a few pixels larger.
        if any(
            other.zone_id == proposal.zone_id
            and sum(other.observed_sides) > sum(proposal.observed_sides)
            and _contains(proposal.component.box, other.component.box, margin=tolerance)
            and _area(other.component.box) >= _area(proposal.component.box) * 0.72
            for other in unique
        ):
            continue
        if any(
            other.zone_id == proposal.zone_id
            and _area(other.component.box) > _area(proposal.component.box) * 1.08
            and _contains(other.component.box, proposal.component.box, margin=tolerance)
            and sum(other.observed_sides) >= sum(proposal.observed_sides)
            for other in unique
        ):
            continue
        maximal.append(proposal)

    # Remove any rectangle whose area is already covered almost completely by
    # larger rectangles in the same line zone. This is the compact set-cover
    # step that discards overlap/intersection artifacts while preserving two
    # genuinely maximal crossing rectangles.
    kept: list[RectangleProposal] = []
    for proposal in maximal:
        covering = [
            other
            for other in maximal
            if other is not proposal
            and other.zone_id == proposal.zone_id
            and _area(other.component.box) > _area(proposal.component.box) * 1.08
        ]
        if covering:
            x1, y1, x2, y2 = proposal.component.box
            coverage = np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.uint8)
            for other in covering:
                ox1, oy1, ox2, oy2 = other.component.box
                ix1, iy1 = max(x1, ox1), max(y1, oy1)
                ix2, iy2 = min(x2, ox2), min(y2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    coverage[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = 1
            if float(np.mean(coverage)) >= 0.92:
                continue
        kept.append(proposal)
    return sorted(
        (proposal.component for proposal in kept),
        key=lambda component: (component.box[1], component.box[0], -_area(component.box)),
    )


def _rectified_solid_candidates(
    gray: np.ndarray, line_height: int
) -> tuple[list[BoxComponent], np.ndarray]:
    candidates, solid_mask = core_stage._solid_candidates(gray)
    h, w = gray.shape
    edge_margin = max(3, round(min(h, w) * 0.018))
    rectified: list[BoxComponent] = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.box
        if (
            x1 <= edge_margin
            and x2 >= w - edge_margin
            and y2 - y1 < line_height * 2.5
        ):
            continue
        rectified.append(
            BoxComponent(
                box=candidate.box,
                polygon=_rect_polygon(candidate.box),
                source="solid_fill_rectified",
                score=candidate.score,
            )
        )
    return rectified, solid_mask


def _orthogonal_polygon_partition(
    polygon: tuple[tuple[int, int], ...],
    box: tuple[int, int, int, int],
    line_height: int,
) -> list[tuple[int, int, int, int]]:
    """Partition a simple right-angle notch into a compact rectangle cover."""

    if len(polygon) < 6:
        return []
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return []
    local = np.asarray([(x - x1, y - y1) for x, y in polygon], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 1)
    fill_ratio = float(np.mean(mask))
    if fill_ratio >= 0.93 or fill_ratio < 0.38:
        return []
    # Contour tracing walks around the ink thickness at every corner. That
    # creates a few tapered rows/columns even for a clean orthogonal polygon.
    # Detect only material endpoint changes, then absorb those short boundary
    # artifacts into the neighbouring stable band.
    change_threshold = max(5, round(line_height * 0.65), round(min(width, height) * 0.015))
    minimum_band = max(3, round(line_height * 0.40))

    def bands(horizontal: bool) -> list[tuple[int, int, int, int]]:
        primary = mask if horizontal else mask.T
        spans: list[tuple[int, int, int]] = []
        for position, values in enumerate(primary):
            occupied = np.flatnonzero(values)
            if occupied.size:
                spans.append((position, int(occupied[0]), int(occupied[-1]) + 1))
        if not spans:
            return []
        first_position, last_position = spans[0][0], spans[-1][0]
        edge_trim = min(
            max(2, round(line_height * 0.30)),
            max(0, (last_position - first_position - 1) // 5),
        )
        core = [
            span
            for span in spans
            if first_position + edge_trim <= span[0] <= last_position - edge_trim
        ]
        if len(core) < 3:
            core = spans

        # A five-row median suppresses raster stair-steps without erasing a
        # genuine rectangular notch, whose endpoint shift persists.
        smoothed: list[tuple[int, int, int]] = []
        radius = 2
        for index, (position, _, _) in enumerate(core):
            window = core[max(0, index - radius):min(len(core), index + radius + 1)]
            smoothed.append(
                (
                    position,
                    int(round(float(np.median([item[1] for item in window])))),
                    int(round(float(np.median([item[2] for item in window])))),
                )
            )

        grouped_spans: list[list[tuple[int, int, int]]] = [[smoothed[0]]]
        for span in smoothed[1:]:
            group = grouped_spans[-1]
            reference_low = int(round(float(np.median([item[1] for item in group]))))
            reference_high = int(round(float(np.median([item[2] for item in group]))))
            contiguous = span[0] == group[-1][0] + 1
            stable = (
                abs(span[1] - reference_low) <= change_threshold
                and abs(span[2] - reference_high) <= change_threshold
            )
            if contiguous and stable:
                group.append(span)
            else:
                grouped_spans.append([span])

        # Corner tapers can survive smoothing as a tiny first/last group. Fold
        # them into the closest neighbour rather than inventing a sliver box.
        changed = True
        while changed and len(grouped_spans) > 1:
            changed = False
            for index, group in enumerate(grouped_spans):
                group_height = group[-1][0] - group[0][0] + 1
                if group_height >= minimum_band:
                    continue
                neighbours: list[tuple[float, int]] = []
                group_low = float(np.median([item[1] for item in group]))
                group_high = float(np.median([item[2] for item in group]))
                for neighbour_index in (index - 1, index + 1):
                    if not 0 <= neighbour_index < len(grouped_spans):
                        continue
                    neighbour = grouped_spans[neighbour_index]
                    neighbour_low = float(np.median([item[1] for item in neighbour]))
                    neighbour_high = float(np.median([item[2] for item in neighbour]))
                    neighbours.append(
                        (
                            abs(group_low - neighbour_low) + abs(group_high - neighbour_high),
                            neighbour_index,
                        )
                    )
                _, neighbour_index = min(neighbours)
                merged = sorted(group + grouped_spans[neighbour_index], key=lambda item: item[0])
                lower_index = min(index, neighbour_index)
                upper_index = max(index, neighbour_index)
                grouped_spans[lower_index] = merged
                del grouped_spans[upper_index]
                changed = True
                break

        grouped: list[tuple[int, int, int, int]] = []
        for index, group in enumerate(grouped_spans):
            lows = [item[1] for item in group]
            highs = [item[2] for item in group]
            start = first_position if index == 0 else group[0][0]
            end = last_position + 1 if index == len(grouped_spans) - 1 else group[-1][0] + 1
            grouped.append(
                (
                    int(round(float(np.percentile(lows, 10)))),
                    start,
                    int(round(float(np.percentile(highs, 90)))),
                    end,
                )
            )
        if horizontal:
            return [(x1 + a, y1 + b, x1 + c, y1 + d) for a, b, c, d in grouped]
        return [(x1 + b, y1 + a, x1 + d, y1 + c) for a, b, c, d in grouped]

    row_cover = bands(True)
    column_cover = bands(False)
    options: list[tuple[float, float, list[tuple[int, int, int, int]]]] = []
    for cover in (row_cover, column_cover):
        if not 2 <= len(cover) <= 4:
            continue
        if not all(
            right - left >= max(16, round(line_height * 0.9))
            and bottom - top >= max(5, round(line_height * 0.28))
            for left, top, right, bottom in cover
        ):
            continue
        cover_mask = np.zeros_like(mask)
        for left, top, right, bottom in cover:
            cover_mask[
                max(0, top - y1):min(height, bottom - y1),
                max(0, left - x1):min(width, right - x1),
            ] = 1
        covered = float(np.sum((cover_mask > 0) & (mask > 0))) / max(1.0, float(np.sum(mask > 0)))
        overfill = float(np.sum((cover_mask > 0) & (mask == 0))) / max(1.0, float(np.sum(cover_mask > 0)))
        if covered >= 0.94 and overfill <= 0.12:
            options.append((covered, overfill, cover))
    if not options:
        return []
    return min(
        options,
        key=lambda item: (len(item[2]), item[1], -item[0], -max(_area(box) for box in item[2])),
    )[2]


def _enclosed_blank_candidates(
    gray: np.ndarray,
    dark: np.ndarray,
    faint: np.ndarray,
    edges: np.ndarray,
    line_height: int,
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover the blank region enclosed by observed border ink.

    Unlike a bounding-box fallback, this representation retains an open notch
    around visible text. The page frame is treated as a valid physical border,
    which also supports masks clipped by one or more page edges.
    """

    height, width = gray.shape
    strong_h = _axis_open(dark, "h")
    strong_v = _axis_open(dark, "v")
    faint_h = _axis_open(faint, "h")
    faint_v = _axis_open(faint, "v")
    observed_lines = cv2.bitwise_or(
        cv2.bitwise_or(strong_h, strong_v), cv2.bitwise_or(faint_h, faint_v)
    )
    observed_lines = cv2.morphologyEx(
        observed_lines,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    barrier_radius = max(1, round(min(height, width) * 0.0018))
    barrier = cv2.dilate(
        observed_lines,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * barrier_radius + 1, 2 * barrier_radius + 1)
        ),
        iterations=1,
    )
    frame = max(2, barrier_radius + 1)
    barrier[:frame, :] = 255
    barrier[-frame:, :] = 255
    barrier[:, :frame] = 255
    barrier[:, -frame:] = 255
    free = np.where(barrier == 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    minimum_width = max(20, round(line_height * 1.20))
    minimum_height = max(6, round(line_height * 0.30))
    candidates: list[BoxComponent] = []
    rejected_text = 0
    rejected_shape = 0

    for label in range(1, count):
        x, y, box_width, box_height, free_area = map(int, stats[label])
        box_area = box_width * box_height
        if (
            box_width < minimum_width
            or box_height < minimum_height
            or box_area < max(150, round(width * height * 0.000045))
            or box_area > width * height * 0.75
        ):
            continue
        fill_ratio = free_area / max(1, box_area)
        if fill_ratio < 0.50:
            rejected_shape += 1
            continue

        local = np.where(labels[y:y + box_height, x:x + box_width] == label, 255, 0).astype(
            np.uint8
        )
        # Put the observed border back into the shape before tracing. This
        # yields coordinates at the physical line rather than several pixels
        # inside the blank interior.
        local = cv2.dilate(
            local,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * barrier_radius + 1, 2 * barrier_radius + 1)
            ),
            iterations=1,
        )
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour = contour + np.asarray([[[x, y]]], dtype=np.int32)
        polygon = _simplify_polygon(contour, gray.shape, line_height)
        if not polygon:
            rejected_shape += 1
            continue
        polygon_box = _polygon_box(polygon)
        px1, py1, px2, py2 = polygon_box
        polygon_width, polygon_height = px2 - px1, py2 - py1
        polygon_area = abs(
            float(cv2.contourArea(np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))))
        )
        polygon_fill = polygon_area / max(1.0, float(polygon_width * polygon_height))
        if polygon_fill < 0.50:
            rejected_shape += 1
            continue
        touches_page = x <= frame or y <= frame or x + box_width >= width - frame or y + box_height >= height - frame
        if touches_page and polygon_height < max(6, round(line_height * 0.42)):
            rejected_shape += 1
            continue

        polygon_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [np.asarray(polygon, dtype=np.int32)], 255)
        interior_mask = cv2.erode(
            polygon_mask,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(3, barrier_radius * 2 + 1), max(3, barrier_radius * 2 + 1)),
            ),
            iterations=1,
        )
        interior_values = gray[interior_mask > 0]
        interior_edges = edges[interior_mask > 0]
        if not interior_values.size:
            continue
        mean = float(np.mean(interior_values))
        dark_fraction = float(np.mean(interior_values < 175))
        edge_fraction = float(np.mean(interior_edges > 0))

        # A shape may be blank overall while enclosing several visible text
        # rows. Those rows are precisely the outer-envelope failure mode.
        # True notches leave the text outside polygon_mask and pass this gate.
        if mean < 187 or dark_fraction > 0.115 or edge_fraction > 0.105:
            rejected_text += 1
            continue

        perimeter_mask = cv2.subtract(
            cv2.dilate(polygon_mask, np.ones((3, 3), np.uint8), iterations=1),
            cv2.erode(polygon_mask, np.ones((3, 3), np.uint8), iterations=1),
        )
        non_frame = perimeter_mask.copy()
        non_frame[:frame, :] = 0
        non_frame[-frame:, :] = 0
        non_frame[:, :frame] = 0
        non_frame[:, -frame:] = 0
        perimeter_pixels = int(np.count_nonzero(non_frame))
        observed_support = (
            int(np.count_nonzero((non_frame > 0) & (cv2.dilate(observed_lines, np.ones((5, 5), np.uint8)) > 0)))
            / max(1, perimeter_pixels)
        )
        touches_frame = (
            px1 <= frame + 1
            or py1 <= frame + 1
            or px2 >= width - frame - 1
            or py2 >= height - frame - 1
        )
        minimum_support = 0.40 if touches_frame else 0.52
        if observed_support < minimum_support:
            rejected_shape += 1
            continue

        score = (
            4.0
            + observed_support * 2.0
            + polygon_fill
            + min(1.0, max(0.0, (mean - 187.0) / 60.0))
            + (1.0 - dark_fraction / 0.115)
            + (1.0 - edge_fraction / 0.105)
        )
        candidates.append(
            BoxComponent(
                box=polygon_box,
                polygon=polygon,
                source=(
                    "enclosed_blank_outline:"
                    f"support={observed_support:.2f}:fill={polygon_fill:.2f}"
                ),
                score=float(score),
            )
        )

    candidates = _dedupe_components(candidates, gray.shape)
    candidate_mask = np.zeros_like(gray)
    for candidate in candidates:
        cv2.polylines(
            candidate_mask,
            [np.asarray(candidate.polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )
    return candidates, candidate_mask, {
        "enclosed_blank_component_count": len(candidates),
        "enclosed_blank_rejected_text_count": rejected_text,
        "enclosed_blank_rejected_shape_count": rejected_shape,
    }


def _closed_blank_contour_candidates(
    gray: np.ndarray,
    dark: np.ndarray,
    edges: np.ndarray,
    line_height: int,
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover a blank shape from the observed inside edge of a closed outline.

    ``RETR_TREE`` distinguishes the blank hole inside border ink from the
    border's noisier outside contour. Following that inside contour avoids the
    old failure mode where a notched or mildly slanted mask was replaced by a
    bounding rectangle that swallowed nearby visible text.

    This is intentionally a closed-outline rescue. Crop-edge masks and boxes
    with missing sides remain the responsibility of the rectilinear/page-frame
    branch, where virtual sides are explicit and auditable.
    """

    height, width = gray.shape
    contours, hierarchy = cv2.findContours(
        dark.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    candidate_mask = np.zeros_like(gray)
    if hierarchy is None:
        return [], candidate_mask, {
            "closed_blank_contour_count": 0,
            "closed_blank_rejected_text_count": 0,
            "closed_blank_rejected_shape_count": 0,
        }

    hierarchy = hierarchy[0]
    minimum_width = max(20, round(line_height * 1.20))
    minimum_height = max(6, round(line_height * 0.30))
    minimum_area = max(150, round(width * height * 0.000045))
    erosion_radius = max(2, round(line_height * 0.10))
    ring_radius = max(2, round(line_height * 0.10))
    candidates: list[BoxComponent] = []
    rejected_text = 0
    rejected_shape = 0

    for contour_index, contour in enumerate(contours):
        # A contour with a parent is the inside edge (a hole) in a connected
        # foreground component. It therefore has observed ink around the
        # boundary rather than a border synthesized across blank space.
        if int(hierarchy[contour_index][3]) < 0:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        contour_area = abs(float(cv2.contourArea(contour)))
        if (
            box_width < minimum_width
            or box_height < minimum_height
            or box_area < minimum_area
            or box_area > width * height * 0.75
            or contour_area < minimum_area * 0.72
        ):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        simplified = cv2.approxPolyDP(
            contour,
            max(1.0, min(line_height * 0.14, perimeter * 0.004)),
            True,
        )
        polygon = _simplify_polygon(simplified, gray.shape, line_height)
        if not polygon:
            rejected_shape += 1
            continue
        polygon_box = _polygon_box(polygon)
        px1, py1, px2, py2 = polygon_box
        polygon_width, polygon_height = px2 - px1, py2 - py1
        polygon_area = abs(
            float(cv2.contourArea(np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))))
        )
        polygon_fill = polygon_area / max(1.0, float(polygon_width * polygon_height))
        # Sparse glyph cavities and underlined title fragments can also be
        # closed holes, but they occupy much less of their bounding envelope
        # than a physical redaction outline. The reviewed true contours are
        # comfortably above this fail-closed floor.
        if polygon_fill < 0.50:
            rejected_shape += 1
            continue

        polygon_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [np.asarray(polygon, dtype=np.int32)], 255)
        interior_mask = cv2.erode(
            polygon_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * erosion_radius + 1, 2 * erosion_radius + 1),
            ),
            iterations=1,
        )
        interior_values = gray[interior_mask > 0]
        interior_edges = edges[interior_mask > 0]
        if not interior_values.size:
            rejected_shape += 1
            continue
        mean = float(np.mean(interior_values))
        dark_fraction = float(np.mean(interior_values < 175))
        edge_fraction = float(np.mean(interior_edges > 0))
        # Closed-contour rescue is stricter than the ordinary outline branch:
        # visible glyphs inside the shape indicate a table, page frame, or the
        # very outer envelope that this branch is designed not to emit.
        if mean < 205 or dark_fraction > 0.055 or edge_fraction > 0.060:
            rejected_text += 1
            continue

        outer = cv2.dilate(
            polygon_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
            ),
            iterations=1,
        )
        inner = cv2.erode(
            polygon_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
            ),
            iterations=1,
        )
        perimeter_mask = cv2.subtract(outer, inner)
        perimeter_pixels = int(np.count_nonzero(perimeter_mask))
        observed_support = (
            int(np.count_nonzero((perimeter_mask > 0) & (dark > 0)))
            / max(1, perimeter_pixels)
        )
        if observed_support < 0.30:
            rejected_shape += 1
            continue

        score = (
            7.0
            + observed_support * 2.0
            + polygon_fill
            + min(1.0, max(0.0, (mean - 205.0) / 45.0))
            + (1.0 - dark_fraction / 0.055)
            + (1.0 - edge_fraction / 0.060)
        )
        candidates.append(
            BoxComponent(
                box=polygon_box,
                polygon=polygon,
                source=(
                    "closed_blank_contour:"
                    f"support={observed_support:.2f}:fill={polygon_fill:.2f}"
                ),
                score=float(score),
            )
        )

    candidates = _dedupe_components(candidates, gray.shape)
    for candidate in candidates:
        cv2.polylines(
            candidate_mask,
            [np.asarray(candidate.polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )
    return candidates, candidate_mask, {
        "closed_blank_contour_count": len(candidates),
        "closed_blank_rejected_text_count": rejected_text,
        "closed_blank_rejected_shape_count": rejected_shape,
    }


def _rectified_outline_fallback(
    gray: np.ndarray,
    dark: np.ndarray,
    faint: np.ndarray,
    edges: np.ndarray,
    line_height: int,
) -> list[BoxComponent]:
    """Recover locally warped outlines when the strict line graph is empty.

    The older contour branch is used only as a fail-closed rescue. Its
    arbitrary contours are never emitted: every accepted candidate is
    converted to an axis-aligned rectangle and contained slivers are removed.
    """
    strong_h, strong_v = core_stage._multi_scale_lines(dark)
    faint_h, faint_v = core_stage._multi_scale_lines(faint)
    horizontal = cv2.bitwise_or(strong_h, faint_h)
    vertical = cv2.bitwise_or(strong_v, faint_v)
    candidates = core_stage._prune_outline_artifacts(
        core_stage._outline_candidates(gray, dark, edges, horizontal, vertical)
    )
    h, w = gray.shape
    rectified: list[BoxComponent] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        x1, y1, x2, y2 = candidate.box
        width, height = x2 - x1, y2 - y1
        if height < max(8, round(line_height * 1.10)):
            continue
        if width < max(22, round(height * 0.70), round(line_height * 3.5)):
            continue
        if y2 == h and height < line_height * 2.5 and width < w * 0.35:
            continue
        partition = _orthogonal_polygon_partition(
            candidate.polygon, candidate.box, line_height
        )
        if partition:
            rectified.extend(
                BoxComponent(
                    box=part,
                    polygon=_rect_polygon(part),
                    source=f"rectilinear_contour_partition:{candidate_index}",
                    score=float(candidate.score) + 0.10,
                )
                for part in partition
            )
        else:
            rectified.append(
                BoxComponent(
                    box=candidate.box,
                    polygon=_rect_polygon(candidate.box),
                    source="rectified_warp_fallback",
                    score=float(candidate.score) - 0.25,
                )
            )

    # Prefer the largest valid rectangle when the contour detector emitted
    # both a union envelope and intersection/contained fragments.
    ordered = sorted(rectified, key=lambda item: (_area(item.box), item.score), reverse=True)
    kept: list[BoxComponent] = []
    for candidate in ordered:
        if any(_contains(previous.box, candidate.box, margin=3) for previous in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.box[1], item.box[0]))


def _oriented_line_segments(
    gray: np.ndarray,
) -> tuple[list[OrientedSegment], list[OrientedSegment]]:
    """Retain mildly slanted local borders that axis morphology can erase."""

    height, width = gray.shape
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    horizontal: list[OrientedSegment] = []
    vertical: list[OrientedSegment] = []
    if detected is None:
        return horizontal, vertical

    for raw in detected[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in raw)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        axis_angle = abs(((angle + 90.0) % 180.0) - 90.0)
        if axis_angle <= 5.5 and length >= max(16.0, width * 0.018):
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            horizontal.append(
                OrientedSegment("h", (x1, y1), (x2, y2), length, angle)
            )
        elif abs(axis_angle - 90.0) <= 5.5 and length >= max(7.0, height * 0.006):
            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            vertical.append(
                OrientedSegment("v", (x1, y1), (x2, y2), length, angle)
            )
    return horizontal, vertical


def _merge_oriented_segments(
    segments: list[OrientedSegment],
    orientation: str,
    line_height: int,
    image_shape: tuple[int, int],
) -> list[OrientedSegment]:
    """Join collinear pieces split by transparent box intersections."""

    if not segments:
        return []
    perpendicular_tolerance = max(2.5, min(image_shape) * 0.0022)
    gap_tolerance = max(6.0, line_height * 0.55)
    parents = list(range(len(segments)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(segments):
        for right_index in range(left_index + 1, len(segments)):
            right = segments[right_index]
            angle_difference = (
                abs(left.angle_degrees - right.angle_degrees)
                if orientation == "h"
                else abs(abs(left.angle_degrees) - abs(right.angle_degrees))
            )
            if angle_difference > 3.5:
                continue
            if orientation == "h":
                perpendicular = abs(left.midpoint[1] - right.midpoint[1])
                left_interval = (left.start[0], left.end[0])
                right_interval = (right.start[0], right.end[0])
            else:
                perpendicular = abs(left.midpoint[0] - right.midpoint[0])
                left_interval = (left.start[1], left.end[1])
                right_interval = (right.start[1], right.end[1])
            gap = max(
                0.0,
                max(left_interval[0], right_interval[0])
                - min(left_interval[1], right_interval[1]),
            )
            if perpendicular <= perpendicular_tolerance and gap <= gap_tolerance:
                union(left_index, right_index)

    groups: dict[int, list[OrientedSegment]] = {}
    for index, segment in enumerate(segments):
        groups.setdefault(find(index), []).append(segment)
    merged: list[OrientedSegment] = []
    for group in groups.values():
        points = np.asarray(
            [point for segment in group for point in (segment.start, segment.end)],
            dtype=np.float64,
        )
        if orientation == "h":
            start_axis, end_axis = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
            if np.ptp(points[:, 0]) >= 1.0:
                slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
                start = (start_axis, float(slope * start_axis + intercept))
                end = (end_axis, float(slope * end_axis + intercept))
            else:
                y = float(np.median(points[:, 1]))
                start, end = (start_axis, y), (end_axis, y)
        else:
            start_axis, end_axis = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
            if np.ptp(points[:, 1]) >= 1.0:
                slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
                start = (float(slope * start_axis + intercept), start_axis)
                end = (float(slope * end_axis + intercept), end_axis)
            else:
                x = float(np.median(points[:, 0]))
                start, end = (x, start_axis), (x, end_axis)
        dx, dy = end[0] - start[0], end[1] - start[1]
        merged.append(
            OrientedSegment(
                orientation=orientation,
                start=start,
                end=end,
                length=math.hypot(dx, dy),
                angle_degrees=math.degrees(math.atan2(dy, dx)),
            )
        )
    return merged


def _oriented_vertical_overlap(
    segment: OrientedSegment, top: float, bottom: float
) -> float:
    segment_top = min(segment.start[1], segment.end[1])
    segment_bottom = max(segment.start[1], segment.end[1])
    return max(0.0, min(segment_bottom, bottom) - max(segment_top, top))


def _oriented_horizontal_coverage(
    segment: OrientedSegment, left: float, right: float
) -> float:
    segment_left = min(segment.start[0], segment.end[0])
    segment_right = max(segment.start[0], segment.end[0])
    return max(0.0, min(segment_right, right) - max(segment_left, left)) / max(
        1.0, right - left
    )


def _endpoint_near(value: float, segment: OrientedSegment, axis: int, tolerance: float) -> bool:
    return min(abs(segment.start[axis] - value), abs(segment.end[axis] - value)) <= tolerance


def _robust_interior_quality(
    gray: np.ndarray,
    edges: np.ndarray,
    box: tuple[int, int, int, int],
    line_height: int,
) -> tuple[float, float, float, float] | None:
    """Validate blank masks while tolerating sparse release-stamp boilerplate."""

    strict = _interior_quality(gray, edges, box, line_height)
    if strict is not None:
        mean, dark_fraction, edge_fraction = strict
        return mean, dark_fraction, edge_fraction, 1.0

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    image_h, image_w = gray.shape
    if width < line_height * 5.0 or height < line_height * 2.0:
        return None
    pad = max(2, min(line_height // 4, max(2, height // 7)))
    interior = gray[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    interior_edges = edges[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    if interior.size == 0:
        return None
    mean = float(np.mean(interior))
    dark_fraction = float(np.mean(interior < 175))
    edge_fraction = float(np.mean(interior_edges > 0))
    row_dark = np.mean(interior < 175, axis=1)
    blank_row_fraction = float(np.mean(row_dark <= 0.10))
    large = width >= image_w * 0.35 or height >= image_h * 0.10
    if not large:
        return None
    if (
        mean < 176
        or dark_fraction > 0.25
        or edge_fraction > 0.13
        or blank_row_fraction < 0.62
    ):
        return None
    return mean, dark_fraction, edge_fraction, blank_row_fraction


def _candidate_side_positions(
    vertical: list[OrientedSegment],
    observed: float,
    top: float,
    bottom: float,
    tolerance: float,
) -> list[tuple[float, float, bool]]:
    height = max(1.0, bottom - top)
    options: list[tuple[float, float, bool]] = []
    for segment in vertical:
        x_position = segment.midpoint[0]
        if abs(x_position - observed) > tolerance:
            continue
        support = _oriented_vertical_overlap(segment, top, bottom) / height
        if support >= 0.32:
            options.append((x_position, support, True))
    options.append((observed, 0.0, False))
    return sorted(options, key=lambda item: (item[1], -abs(item[0] - observed)), reverse=True)[:4]


def _lsd_geometry_rescues(
    gray: np.ndarray,
    line_height: int,
    existing: Iterable[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover warped, thin, and one-corner-clipped outline rectangles.

    Every proposal is based only on the current page. Two parallel horizontal
    borders are mandatory. Vertical strokes must terminate near those borders;
    this endpoint guard prevents tall glyphs from becoming greedy box sides.
    """

    height, width = gray.shape
    raw_horizontal, raw_vertical = _oriented_line_segments(gray)
    # Keep both granular and collinearly merged views. Granular segments retain
    # short thin boxes; merged segments bridge borders split at transparent
    # intersections in layered masks.
    horizontal = raw_horizontal + _merge_oriented_segments(
        raw_horizontal, "h", line_height, gray.shape
    )
    vertical = raw_vertical + _merge_oriented_segments(
        raw_vertical, "v", line_height, gray.shape
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 110)
    boundary_margin = max(4.0, min(height, width) * 0.018)
    side_tolerance = max(7.0, line_height * 0.65)
    corner_tolerance = max(8.0, line_height * 0.55)
    minimum_height = max(5.0, line_height * 0.32)
    minimum_width = max(18.0, line_height * 1.15)
    raw: list[BoxComponent] = []

    for top_index, first in enumerate(horizontal):
        for second in horizontal[top_index + 1:]:
            top, bottom = first, second
            top_y = (top.start[1] + top.end[1]) / 2.0
            bottom_y = (bottom.start[1] + bottom.end[1]) / 2.0
            if top_y > bottom_y:
                top, bottom = bottom, top
                top_y, bottom_y = bottom_y, top_y
            box_height = bottom_y - top_y
            if box_height < minimum_height or box_height > height * 0.58:
                continue
            if abs(top.angle_degrees - bottom.angle_degrees) > 4.5:
                continue

            top_left, top_right = top.start[0], top.end[0]
            bottom_left, bottom_right = bottom.start[0], bottom.end[0]
            overlap = max(0.0, min(top_right, bottom_right) - max(top_left, bottom_left))
            shorter = min(top_right - top_left, bottom_right - bottom_left)
            if overlap < max(10.0, shorter * 0.10):
                continue
            observed_left = min(top_left, bottom_left)
            observed_right = max(top_right, bottom_right)
            left_options = _candidate_side_positions(
                vertical, observed_left, top_y, bottom_y, side_tolerance
            )
            right_options = _candidate_side_positions(
                vertical, observed_right, top_y, bottom_y, side_tolerance
            )
            if observed_left <= boundary_margin:
                left_options.insert(0, (0.0, 1.0, True))
            if observed_right >= width - 1 - boundary_margin:
                right_options.insert(0, (float(width - 1), 1.0, True))

            for left_position, left_support, left_observed in left_options:
                for right_position, right_support, right_observed in right_options:
                    box_width = right_position - left_position
                    if box_width < minimum_width or box_width < box_height * 0.58:
                        continue
                    top_support = _oriented_horizontal_coverage(
                        top, left_position, right_position
                    )
                    bottom_support = _oriented_horizontal_coverage(
                        bottom, left_position, right_position
                    )
                    if min(top_support, bottom_support) < 0.18:
                        continue
                    if top_support + bottom_support < 0.90:
                        continue

                    left_boundary = left_position <= 0.5
                    right_boundary = right_position >= width - 1.5
                    left_effective = left_observed or left_boundary
                    right_effective = right_observed or right_boundary
                    thin = box_height < line_height * 0.92
                    if thin:
                        if (
                            min(top_support, bottom_support) < 0.62
                            or min(left_support, right_support) < 0.52
                            or not (left_observed and right_observed)
                        ):
                            continue
                    elif not (left_effective and right_effective):
                        # One occluded vertical side is permitted only for a
                        # substantial, very strongly anchored blank rectangle.
                        visible_support = max(left_support, right_support)
                        if (
                            box_width < line_height * 7.0
                            or box_height < line_height * 1.35
                            or visible_support < 0.72
                            or min(top_support, bottom_support) < 0.58
                        ):
                            continue

                    box = (
                        max(0, int(round(left_position))),
                        max(0, int(round(top_y))),
                        min(width, int(round(right_position)) + 1),
                        min(height, int(round(bottom_y)) + 1),
                    )
                    quality = _robust_interior_quality(gray, edges, box, line_height)
                    if quality is None:
                        continue

                    corner_anchors = 0
                    for y_value, h_segment in ((top_y, top), (bottom_y, bottom)):
                        for x_value, side_observed in (
                            (left_position, left_observed or left_boundary),
                            (right_position, right_observed or right_boundary),
                        ):
                            horizontal_anchor = _endpoint_near(
                                x_value, h_segment, 0, corner_tolerance
                            )
                            if x_value <= 0.5 or x_value >= width - 1.5:
                                vertical_anchor = True
                            else:
                                vertical_anchor = any(
                                    abs(segment.midpoint[0] - x_value) <= side_tolerance
                                    and _endpoint_near(
                                        y_value, segment, 1, corner_tolerance
                                    )
                                    for segment in vertical
                                )
                            if horizontal_anchor and vertical_anchor and side_observed:
                                corner_anchors += 1
                    required_anchors = 3 if thin else 2
                    if corner_anchors < required_anchors:
                        continue

                    mean, dark_fraction, edge_fraction, blank_rows = quality
                    score = (
                        top_support
                        + bottom_support
                        + left_support
                        + right_support
                        + corner_anchors * 0.35
                        + min(1.0, max(0.0, (mean - 178.0) / 60.0))
                        + blank_rows
                        - 0.35 * int(not left_effective or not right_effective)
                    )
                    raw.append(
                        BoxComponent(
                            box=box,
                            polygon=_rect_polygon(box),
                            source=(
                                "lsd_geometry_rescue:"
                                f"h={top_support:.2f}/{bottom_support:.2f}:"
                                f"v={left_support:.2f}/{right_support:.2f}:"
                                f"corners={corner_anchors}"
                            ),
                            score=float(score),
                        )
                    )

    deduplicated = _dedupe_components(raw, gray.shape)
    ordered = sorted(
        deduplicated,
        key=lambda item: (item.score, _area(item.box)),
        reverse=True,
    )
    compact: list[BoxComponent] = []
    containment_margin = max(3, round(min(height, width) * 0.006))
    for candidate in ordered:
        if any(
            _contains(prior.box, candidate.box, margin=containment_margin)
            and _area(prior.box) >= _area(candidate.box) * 1.08
            and prior.score >= candidate.score - 0.25
            for prior in compact
        ):
            continue
        compact.append(candidate)
    compact = sorted(compact[:80], key=lambda item: (item.box[1], item.box[0]))
    mask = np.zeros_like(gray)
    for candidate in compact:
        x1, y1, x2, y2 = candidate.box
        cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, 1)
    return compact, mask, {
        "lsd_horizontal_segment_count": len(horizontal),
        "lsd_vertical_segment_count": len(vertical),
        "lsd_raw_candidate_count": len(raw),
        "lsd_geometry_rescue_count": len(compact),
    }


def _lsd_corner_count(component: BoxComponent) -> int:
    marker = "corners="
    if marker not in component.source:
        return 0
    try:
        return int(component.source.rsplit(marker, 1)[1].split(":", 1)[0])
    except ValueError:
        return 0


def _apply_lsd_refinements(
    components: list[BoxComponent],
    lsd_candidates: list[BoxComponent],
    image_shape: tuple[int, int],
) -> tuple[list[BoxComponent], list[BoxComponent], int]:
    """Use four-corner LSD boxes to trim glyph-clipped strict rectangles."""

    tolerance = max(4, round(min(image_shape) * 0.009))
    used: set[int] = set()
    refined: list[BoxComponent] = []
    refinement_count = 0
    for component in components:
        if "rectilinear_zone_" not in component.source:
            refined.append(component)
            continue
        options: list[tuple[int, BoxComponent, int, float]] = []
        for index, candidate in enumerate(lsd_candidates):
            if index in used or candidate.score < 6.0 or _lsd_corner_count(candidate) < 3:
                continue
            area_ratio = _area(candidate.box) / max(1, _area(component.box))
            if not 0.55 <= area_ratio <= 1.12:
                continue
            if _intersection_over_smaller(candidate.box, component.box) < 0.82:
                continue
            aligned = sum(
                abs(left - right) <= tolerance
                for left, right in zip(candidate.box, component.box)
            )
            if aligned < 2:
                continue
            # Refinement is local boundary correction, not permission to
            # replace a supported rectangle by one overlap/intersection cell.
            # Record side agreement and area fidelity before considering the
            # LSD confidence score.
            area_fidelity = abs(
                math.log(
                    max(
                        1e-6,
                        _area(candidate.box) / max(1, _area(component.box)),
                    )
                )
            )
            options.append((index, candidate, aligned, area_fidelity))
        if not options:
            refined.append(component)
            continue
        index, candidate, aligned, _ = max(
            options,
            key=lambda item: (
                item[2],
                _lsd_corner_count(item[1]),
                -item[3],
                item[1].score,
            ),
        )
        base_width = max(1, component.box[2] - component.box[0])
        base_height = max(1, component.box[3] - component.box[1])
        width_ratio = (candidate.box[2] - candidate.box[0]) / base_width
        height_ratio = (candidate.box[3] - candidate.box[1]) / base_height
        # An overlap cell typically preserves one full dimension while cutting
        # the other nearly in half. A legitimate correction can trim a false
        # glyph side and therefore reduce total area, but it should not collapse
        # just one axis this severely without four-side agreement.
        if (
            aligned < 4
            and min(width_ratio, height_ratio) < 0.62
            and max(width_ratio, height_ratio) > 0.90
        ):
            refined.append(component)
            continue
        used.add(index)
        refinement_count += 1
        refined.append(
            BoxComponent(
                box=candidate.box,
                polygon=_rect_polygon(candidate.box),
                # Keep the measured base route. Downstream suppression must
                # still know when a refined outline began as a fully observed
                # four-sided rectangle rather than an inferred text frame.
                source=(
                    f"lsd_corner_refinement:{candidate.source}:"
                    f"base={component.source}"
                ),
                score=max(component.score, candidate.score),
            )
        )
    remaining = [
        candidate
        for index, candidate in enumerate(lsd_candidates)
        if index not in used
    ]
    return refined, remaining, refinement_count


def _candidate_coverage(
    candidate: BoxComponent, components: Iterable[BoxComponent]
) -> float:
    x1, y1, x2, y2 = candidate.box
    coverage = np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=np.uint8)
    for component in components:
        ox1, oy1, ox2, oy2 = component.box
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            coverage[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = 1
    return float(np.mean(coverage))


def _intersection_over_smaller(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    width = max(0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0, min(ly2, ry2) - max(ly1, ry1))
    return (width * height) / max(1, min(_area(left), _area(right)))


def _novel_rescues(
    candidates: Iterable[BoxComponent],
    existing: list[BoxComponent],
    image_shape: tuple[int, int],
) -> list[BoxComponent]:
    """Keep candidate-local rescues while removing duplicate union envelopes."""

    tolerance = max(3, round(min(image_shape) * 0.006))
    accepted: list[BoxComponent] = []
    for candidate in sorted(
        candidates, key=lambda item: (item.score, _area(item.box)), reverse=True
    ):
        if candidate.source.startswith("lsd_geometry_rescue") and _lsd_corner_count(candidate) < 3:
            continue
        prior = existing + accepted
        if any(_near_same_box(candidate.box, item.box, tolerance) for item in prior):
            continue
        # LSD is a local recovery channel, not permission to draw a synthetic
        # envelope around boxes already explained by the stricter line graph.
        # Overlapping/layered shapes are recovered by the contour decomposition.
        if candidate.source.startswith("lsd_geometry_rescue"):
            synthetic_envelope = any(
                _contains(candidate.box, item.box, margin=tolerance)
                and _area(candidate.box) >= _area(item.box) * 1.18
                for item in prior
            )
            heavy_overlap = any(
                _intersection_over_smaller(candidate.box, item.box) >= 0.55
                for item in prior
            )
            if synthetic_envelope or heavy_overlap:
                continue
        if "rectified_warp_fallback" in candidate.source:
            # Contour closing can wrap a real rectangle together with adjacent
            # text into a much larger envelope. Keep crossing shapes, but not a
            # candidate that mostly swallows an already-complete strict box.
            swallowed = any(
                _intersection_over_smaller(candidate.box, item.box) >= 0.75
                and _area(candidate.box) >= _area(item.box) * 1.35
                for item in prior
            )
            if swallowed:
                continue
        coverage = _candidate_coverage(candidate, prior)
        contained = [
            item
            for item in prior
            if _contains(candidate.box, item.box, margin=tolerance)
            and _area(item.box) < _area(candidate.box) * 0.92
        ]
        # Crossing rectangles often induce one tempting outer envelope. It is
        # not a physical box when two already-supported children explain most
        # of that envelope's ink geometry.
        if coverage >= 0.72 or (len(contained) >= 2 and coverage >= 0.45):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.box[1], item.box[0]))


def _suppress_lsd_border_echoes(
    components: list[BoxComponent], image_shape: tuple[int, int], line_height: int
) -> list[BoxComponent]:
    """Remove a thin LSD copy of a stronger neighbouring component border."""

    tolerance = max(3, round(min(image_shape) * 0.006))
    remove: set[int] = set()
    for index, candidate in enumerate(components):
        if not candidate.source.startswith("lsd_geometry_rescue"):
            continue
        x1, y1, x2, y2 = candidate.box
        width, height = x2 - x1, y2 - y1
        if height > line_height * 1.55:
            continue
        for other_index, other in enumerate(components):
            if other_index == index or other.source.startswith("lsd_geometry_rescue"):
                continue
            ox1, oy1, ox2, oy2 = other.box
            overlap_width = max(0, min(x2, ox2) - max(x1, ox1))
            horizontal_fraction = overlap_width / max(1, min(width, ox2 - ox1))
            vertical_overlap = max(0, min(y2, oy2) - max(y1, oy1))
            vertical_gap = max(0, max(y1, oy1) - min(y2, oy2))
            aligned_side = (
                abs(x1 - ox1) <= tolerance * 2
                or abs(x2 - ox2) <= tolerance * 2
                or (ox1 <= x1 + tolerance and ox2 >= x2 - tolerance)
            )
            near_horizontal_border = (
                min(abs(y1 - oy1), abs(y1 - oy2), abs(y2 - oy1), abs(y2 - oy2))
                <= max(tolerance * 2, round(line_height * 0.45))
            )
            if (
                horizontal_fraction >= 0.78
                and aligned_side
                and near_horizontal_border
                and (vertical_overlap > 0 or vertical_gap <= line_height * 0.25)
            ):
                remove.add(index)
                break
    return [
        component for index, component in enumerate(components) if index not in remove
    ]


def _dense_blackout_candidates(
    gray: np.ndarray, line_height: int, existing: Iterable[BoxComponent]
) -> tuple[list[BoxComponent], np.ndarray]:
    """Recover rare solid black masks, including mildly irregular ink blocks."""

    height, width = gray.shape
    raw = ((gray < 105).astype(np.uint8) * 255)
    joined = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(900, round(width * height * 0.00020))
    candidates: list[BoxComponent] = []
    existing_components = list(existing)
    tolerance = max(3, round(min(height, width) * 0.005))
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box = (x, y, x + box_width, y + box_height)
        area = box_width * box_height
        if (
            area < minimum_area
            or box_height < max(9, round(line_height * 0.62))
            or box_width < max(55, round(line_height * 2.8), round(box_height * 1.35))
        ):
            continue
        roi = joined[y:y + box_height, x:x + box_width] > 0
        gray_roi = gray[y:y + box_height, x:x + box_width]
        fill = float(np.mean(roi)) if roi.size else 0.0
        contour_fill = float(cv2.contourArea(contour) / max(1, area))
        coherent_rows = float(np.mean(np.mean(roi, axis=1) >= 0.55))
        if (
            fill < 0.72
            or contour_fill < 0.64
            or coherent_rows < 0.58
            or float(np.mean(gray_roi)) > 105
        ):
            continue
        if any(_near_same_box(box, component.box, tolerance) for component in existing_components):
            continue
        candidates.append(
            BoxComponent(
                box=box,
                polygon=_rect_polygon(box),
                source="dense_blackout_rectified",
                score=5.0 + fill + contour_fill + coherent_rows,
            )
        )
    return _dedupe_components(candidates, gray.shape), joined


def _canonicalize_components(
    components: list[BoxComponent],
    image_shape: tuple[int, int],
    line_height: int,
) -> list[BoxComponent]:
    """Replace locally clipped duplicates without collapsing true overlaps."""

    tolerance = max(3, round(min(image_shape) * 0.006))
    remove: set[int] = set()
    for outer_index, outer in enumerate(components):
        if "rectified_warp_fallback" not in outer.source:
            continue
        for inner_index, inner in enumerate(components):
            if outer_index == inner_index or inner_index in remove:
                continue
            if "rectilinear_zone_" not in inner.source:
                continue
            if not _contains(outer.box, inner.box, margin=tolerance):
                continue
            ratio = _area(outer.box) / max(1, _area(inner.box))
            aligned = sum(
                abs(left - right) <= tolerance * 2
                for left, right in zip(outer.box, inner.box)
            )
            if ratio <= 2.8 and aligned >= 2:
                remove.add(inner_index)
    retained = [
        component
        for index, component in enumerate(components)
        if index not in remove
    ]
    partition_counts = Counter(
        component.source
        for component in retained
        if component.source.startswith("rectilinear_contour_partition:")
    )
    # A contour partition is useful only as a multi-rectangle explanation of
    # a notched outline. A lone, sub-line-height fragment is usually a glyph
    # or scan artifact clipped off an otherwise valid rectangle. Genuine thin
    # masks remain available through the stricter four-corner LSD channel.
    return [
        component
        for component in retained
        if not (
            component.source.startswith("rectilinear_contour_partition:")
            and partition_counts[component.source] == 1
            and component.box[3] - component.box[1] < max(9, round(line_height * 0.65))
        )
    ]


def _dedupe_components(
    components: list[BoxComponent], image_shape: tuple[int, int]
) -> list[BoxComponent]:
    tolerance = max(2, round(min(image_shape) * 0.004))
    ordered = sorted(components, key=lambda item: (item.score, _area(item.box)), reverse=True)
    kept: list[BoxComponent] = []
    for component in ordered:
        if any(_near_same_box(component.box, prior.box, tolerance) for prior in kept):
            continue
        kept.append(component)
    return sorted(kept, key=lambda item: (item.box[1], item.box[0], -_area(item.box)))


def _positive_overlap(a: BoxComponent, b: BoxComponent) -> bool:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
    overlap_area = overlap_width * overlap_height
    minimum_area = min(_area(a.box), _area(b.box))
    if overlap_width < 3 or overlap_height < 3:
        return False
    return overlap_area / max(1, minimum_area) >= 0.02


def _substantial_physical_contact(
    a: BoxComponent,
    b: BoxComponent,
    *,
    tolerance: int,
    line_height: int,
    text_mask: np.ndarray,
) -> bool:
    """Return whether two outlines share a material edge or overlap zone.

    A transparent overlapping box can erase the other box's corner, leaving
    only a one-pixel overlap or a tiny raster gap. Such components belong to
    one semantic redaction region when they share a substantial boundary.
    Corner proximity alone is insufficient, and visible text in a real gap
    prevents a merge.
    """

    if _positive_overlap(a, b):
        return True
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    vertical_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
    horizontal_gap = max(0, max(ax1, bx1) - min(ax2, bx2))

    if vertical_gap <= tolerance:
        minimum_width = max(1, min(ax2 - ax1, bx2 - bx1))
        shared_fraction = horizontal_overlap / minimum_width
        # A one-pixel overlap can be the rasterized common edge of layered
        # boxes, but only when the narrower component is almost entirely
        # aligned with the wider one. Do not merge merely stacked masks whose
        # spans happen to cross or sit a few pixels apart.
        if (
            vertical_overlap > 0
            and horizontal_overlap >= line_height * 1.5
            and shared_fraction >= 0.80
        ):
            gap = (
                max(ax1, bx1),
                min(ay2, by2),
                min(ax2, bx2),
                max(ay1, by1),
            )
            return vertical_gap == 0 or _ink_fraction(text_mask, gap) <= 0.03

    if horizontal_gap <= tolerance:
        minimum_height = max(1, min(ay2 - ay1, by2 - by1))
        shared_fraction = vertical_overlap / minimum_height
        if vertical_overlap >= line_height * 0.55 and shared_fraction >= 0.72:
            gap = (
                min(ax2, bx2),
                max(ay1, by1),
                max(ax1, bx1),
                min(ay2, by2),
            )
            return horizontal_gap == 0 or _ink_fraction(text_mask, gap) <= 0.03
    return False


def _same_reading_row(
    first: BoxComponent,
    second: BoxComponent,
    line_pitch: int,
) -> bool:
    _, fy1, _, fy2 = first.box
    _, sy1, _, sy2 = second.box
    vertical_overlap = max(0, min(fy2, sy2) - max(fy1, sy1))
    minimum_height = max(1, min(fy2 - fy1, sy2 - sy1))
    centers_close = (
        abs((fy1 + fy2) - (sy1 + sy2)) / 2.0
        <= max(4, line_pitch * 0.42)
    )
    return vertical_overlap >= minimum_height * 0.45 or centers_close


def _text_mask_and_margins(
    gray: np.ndarray,
    line_mask: np.ndarray,
    components: list[BoxComponent],
) -> tuple[np.ndarray, int, int, int]:
    h, w = gray.shape
    text = ((gray < 170).astype(np.uint8) * 255)
    expanded_lines = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    text[expanded_lines > 0] = 0
    for component in components:
        x1, y1, x2, y2 = component.box
        text[max(0, y1 - 2):min(h, y2 + 2), max(0, x1 - 2):min(w, x2 + 2)] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(text, connectivity=8)
    xs: list[float] = []
    ys: list[float] = []
    heights: list[int] = []
    clean = np.zeros_like(text)
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area < 3 or width > w * 0.10 or height > h * 0.08:
            continue
        clean[labels == label] = 255
        xs.extend([x, x + width])
        ys.append(float(centroids[label][1]))
        heights.append(height)
    if xs:
        left = max(0, int(np.percentile(xs, 2)))
        right = min(w - 1, int(np.percentile(xs, 98)))
    else:
        left, right = 0, w - 1
    projection = np.count_nonzero(clean, axis=1)
    row_threshold = max(3, round(w * 0.004))
    bands: list[tuple[int, int]] = []
    band_start: int | None = None
    for row_index, count_on_row in enumerate(projection):
        if count_on_row >= row_threshold and band_start is None:
            band_start = row_index
        elif count_on_row < row_threshold and band_start is not None:
            if row_index - band_start >= 3:
                bands.append((band_start, row_index))
            band_start = None
    if band_start is not None and h - band_start >= 3:
        bands.append((band_start, h))
    band_heights = [end - start for start, end in bands]
    typical_height = (
        float(np.median(band_heights))
        if band_heights
        else (float(np.median(heights)) if heights else max(8.0, h / 45.0))
    )
    centers = [(start + end) / 2.0 for start, end in bands]
    center_gaps = [
        right - left
        for left, right in zip(centers, centers[1:])
        if typical_height * 0.75 <= right - left <= typical_height * 2.4
    ]
    pitch = int(round(float(np.median(center_gaps)))) if center_gaps else 0
    if pitch < typical_height:
        pitch = max(10, int(round(typical_height * 1.35)))
    return clean, left, right, pitch


def _ink_fraction(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    roi = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    return float(np.mean(roi > 0)) if roi.size else 0.0


def _group_components_reading_order(
    components: list[BoxComponent],
    gray: np.ndarray,
    line_mask: np.ndarray,
) -> list[RedactionRegion]:
    if not components:
        return []
    h, w = gray.shape
    text_mask, content_left, content_right, line_pitch = _text_mask_and_margins(
        gray, line_mask, components
    )
    tolerance = max(2, round(min(h, w) * 0.004))
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

    # A notched/occluded outline is intentionally decomposed into a compact
    # rectangular cover. Preserve the source contour identity so those cover
    # pieces remain one semantic redaction rather than unrelated regions.
    contour_groups: dict[str, list[int]] = {}
    for index, component in enumerate(components):
        if component.source.startswith("rectilinear_contour_partition:"):
            contour_groups.setdefault(component.source, []).append(index)
    for indices in contour_groups.values():
        for index in indices[1:]:
            union(indices[0], index)

    line_height = _text_line_height(gray, line_mask)

    # Real geometric overlap or a substantial shared edge is one physical
    # redaction zone. Tiny corner contact remains separate.
    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            if _substantial_physical_contact(
                components[left],
                components[right],
                tolerance=tolerance,
                line_height=line_height,
                text_mask=text_mask,
            ):
                union(left, right)

    # Reading-order continuation is directional and one-to-one. Merely stacked
    # boxes at the same margin are not merged.
    wrap_candidates: list[tuple[float, int, int]] = []
    margin_slack = max(tolerance * 3, round(w * 0.11))
    content_width = max(1, content_right - content_left)
    for left, first in enumerate(components):
        fx1, fy1, fx2, fy2 = first.box
        for right, second in enumerate(components):
            if left == right or find(left) == find(right):
                continue
            sx1, sy1, sx2, sy2 = second.box
            delta = sy1 - fy1
            # Multi-line masks can overlap vertically even though their text
            # order is unambiguous (line-end component followed by a
            # next-line-start component). The directional margin and empty-
            # text guards below carry the precision burden; do not require a
            # half-line vertical displacement here.
            if delta < max(4, line_pitch * 0.25) or delta > line_pitch * 1.75:
                continue
            after = (fx2, fy1 + 1, content_right + 1, max(fy1 + 2, fy2 - 1))
            before = (content_left, sy1 + 1, sx1, max(sy1 + 2, sy2 - 1))
            after_ink = _ink_fraction(text_mask, after)
            before_ink = _ink_fraction(text_mask, before)
            # Residual border antialiasing and scan speckle commonly occupy
            # about two percent of an otherwise empty continuation gap.
            # Actual intervening words are materially denser; keep this below
            # that regime while allowing the reviewed final-012 wrap.
            if after_ink > 0.030 or before_ink > 0.030:
                continue
            exits_line = (
                fx2 >= content_right - margin_slack
                or (
                    fx1 >= content_left + content_width * 0.34
                    and fx2 >= content_left + content_width * 0.62
                )
            )
            enters_line = (
                sx1 <= content_left + margin_slack
                or (
                    sx1 <= content_left + content_width * 0.38
                    and sx2 <= content_left + content_width * 0.72
                )
            )
            # The first component must be materially to the right of the next
            # component. This reading-order guard prevents vertically stacked
            # boxes at the same margin from being joined merely because both
            # have blank surrounding space.
            if not exits_line or not enters_line or fx1 <= sx1 + tolerance * 2:
                continue
            # Continuation follows reading order, not merely page margins. If
            # another independent redaction occurs farther right on the same
            # line, that component is the possible line exit. Likewise, the
            # incoming component must be the leftmost redaction on its line.
            if any(
                index != left
                and find(index) != find(left)
                and other.box[0] > fx1 + tolerance
                and _same_reading_row(first, other, line_pitch)
                for index, other in enumerate(components)
            ):
                continue
            if any(
                index != right
                and find(index) != find(right)
                and other.box[0] < sx1 - tolerance
                and _same_reading_row(second, other, line_pitch)
                for index, other in enumerate(components)
            ):
                continue
            cost = abs(delta - line_pitch) + abs(content_right - fx2) * 0.02 + abs(sx1 - content_left) * 0.02
            wrap_candidates.append((float(cost), left, right))

    used_exit: set[int] = set()
    used_entry: set[int] = set()
    for _, left, right in sorted(wrap_candidates):
        if left in used_exit or right in used_entry or find(left) == find(right):
            continue
        union(left, right)
        used_exit.add(left)
        used_entry.add(right)

    grouped: dict[int, list[BoxComponent]] = {}
    for index, component in enumerate(components):
        grouped.setdefault(find(index), []).append(component)
    regions = [
        RedactionRegion(
            components=tuple(sorted(values, key=lambda item: (item.box[1], item.box[0])))
        )
        for values in grouped.values()
    ]
    return sorted(
        regions,
        key=lambda region: (
            region.components[0].box[1],
            region.components[0].box[0],
        ),
    )


def detect_redaction_regions_with_artifacts(
    gray: np.ndarray,
) -> tuple[list[RedactionRegion], dict[str, np.ndarray], dict[str, Any]]:
    work_gray, inverse, rotation = core_stage._deskew(gray)
    dark = core_stage._binarize_dark(work_gray)
    enhanced, faint = core_stage._binarize_faint(work_gray)
    h_segments, v_segments, horizontal, vertical = _extract_axis_segments(dark, faint)
    line_mask = cv2.bitwise_or(horizontal, vertical)
    edges = cv2.Canny(enhanced, 30, 110)
    line_height = _text_line_height(work_gray, line_mask)
    zones = _line_zones(horizontal, vertical)
    proposals: list[RectangleProposal] = []
    for zone_id, zone in enumerate(zones, start=1):
        proposals.extend(
            _enumerate_zone_rectangles(
                work_gray,
                edges,
                h_segments,
                v_segments,
                zone,
                zone_id,
                line_height,
            )
        )
    strict_outline = _compact_maximal_cover(proposals, work_gray.shape)
    # The contour route traces the blank area actually enclosed by observed border ink. It
    # does not turn a loose contour's outer bounds into a filled rectangle,
    # which preserves concave notches around visible text.
    closed_candidates, closed_mask, closed_diagnostics = _closed_blank_contour_candidates(
        work_gray, dark, edges, line_height
    )
    closed_rescues = _novel_rescues(
        closed_candidates, strict_outline, work_gray.shape
    )
    fallback_candidates, enclosed_mask, enclosed_diagnostics = _enclosed_blank_candidates(
        work_gray, dark, faint, edges, line_height
    )
    contour_rescues = _novel_rescues(
        fallback_candidates,
        strict_outline + closed_rescues,
        work_gray.shape,
    )
    outline = strict_outline + closed_rescues + contour_rescues
    solid, solid_mask = _rectified_solid_candidates(work_gray, line_height)
    blackout, blackout_mask = _dense_blackout_candidates(
        work_gray, line_height, outline + solid
    )
    base_components = _dedupe_components(
        outline + solid + blackout, work_gray.shape
    )
    lsd_candidates, lsd_mask, lsd_diagnostics = _lsd_geometry_rescues(
        work_gray, line_height, base_components
    )
    refined_base, remaining_lsd, lsd_refinement_count = _apply_lsd_refinements(
        base_components, lsd_candidates, work_gray.shape
    )
    lsd_rescues = _novel_rescues(
        remaining_lsd, refined_base, work_gray.shape
    )
    combined_components = _dedupe_components(refined_base + lsd_rescues, work_gray.shape)
    combined_components = _suppress_lsd_border_echoes(
        combined_components, work_gray.shape, line_height
    )
    components = _canonicalize_components(
        combined_components,
        work_gray.shape,
        line_height,
    )
    regions = _group_components_reading_order(components, work_gray, line_mask)
    artifacts = {
        "deskewed_source": work_gray,
        "dark_mask": dark,
        "faint_mask": faint,
        "horizontal_lines": horizontal,
        "vertical_lines": vertical,
        "lines_mask": line_mask,
        "solid_mask": solid_mask,
        "blackout_mask": blackout_mask,
        "closed_blank_contours": closed_mask,
        "enclosed_blank_outlines": enclosed_mask,
        "lsd_geometry_rescues": lsd_mask,
        "edges": edges,
    }
    diagnostics: dict[str, Any] = {
        "coordinate_system": "deskewed",
        "estimated_rotation_degrees": rotation,
        "inverse_affine_to_original": [[float(value) for value in row] for row in inverse],
        "horizontal_segment_count": len(h_segments),
        "vertical_segment_count": len(v_segments),
        "line_zone_count": len(zones),
        "rectangle_proposal_count": len(proposals),
        "strict_outline_component_count": len(strict_outline),
        "outline_component_count": len(outline),
        "closed_blank_rescue_count": len(closed_rescues),
        "rectified_warp_candidate_count": len(fallback_candidates),
        "rectified_warp_rescue_count": len(contour_rescues),
        # Compatibility key retained for existing diagnostic consumers.
        "rectified_warp_fallback_count": len(contour_rescues),
        "solid_candidate_count": len(solid),
        "dense_blackout_candidate_count": len(blackout),
        "lsd_corner_refinement_count": lsd_refinement_count,
        "lsd_geometry_rescue_count": len(lsd_rescues),
        "component_count_before_grouping": len(components),
        "region_count": len(regions),
        "estimated_text_line_height": line_height,
        "detector_policy": (
            "Single-page geometry only: strict rectilinear backbone plus "
            "shape-preserving enclosed-blank, line-segment, and blackout rescues."
        ),
    }
    diagnostics.update(closed_diagnostics)
    diagnostics.update(enclosed_diagnostics)
    diagnostics.update(lsd_diagnostics)
    diagnostics["lsd_geometry_rescue_count"] = len(lsd_rescues)
    return regions, artifacts, diagnostics


def detect_redaction_boxes_with_artifacts(
    gray: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], dict[str, np.ndarray]]:
    regions, artifacts, _ = detect_redaction_regions_with_artifacts(gray)
    return [region.box for region in regions], artifacts


def _map_polygon_to_original(
    polygon: tuple[tuple[int, int], ...], inverse: np.ndarray, width: int, height: int
) -> list[list[int]]:
    points = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    mapped = cv2.transform(points, inverse).reshape((-1, 2))
    return [
        [
            max(0, min(width - 1, int(round(x)))),
            max(0, min(height - 1, int(round(y)))),
        ]
        for x, y in mapped
    ]


def _region_outline_polygons(
    region: RedactionRegion, image_shape: tuple[int, int]
) -> list[tuple[tuple[int, int], ...]]:
    """Trace the exterior union without inventing borders inside overlaps."""

    mask = np.zeros(image_shape, dtype=np.uint8)
    for component in region.components:
        cv2.fillPoly(mask, [np.asarray(component.polygon, dtype=np.int32)], 255)
    # Scans and deskewing can leave a one-pixel seam between outlines that
    # physically meet. Close only that seam before tracing the exterior; this
    # does not alter the retained component coordinates.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines: list[tuple[tuple[int, int], ...]] = []
    for contour in sorted(contours, key=lambda value: cv2.boundingRect(value)[:2][::-1]):
        if len(contour) < 3:
            continue
        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, max(0.75, perimeter * 0.001), True)
        polygon = tuple((int(point[0][0]), int(point[0][1])) for point in simplified)
        if len(polygon) >= 3:
            outlines.append(polygon)
    return outlines


def process_record(record: InputRecord, *, out_root: Path, save_debug_masks: bool) -> dict[str, Any]:
    original_gray = cv2.imread(str(record.rendered_image_path), cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise RuntimeError(f"Could not read image: {record.rendered_image_path}")
    regions, artifacts, diagnostics = detect_redaction_regions_with_artifacts(original_gray)
    display_gray = artifacts["deskewed_source"]
    inverse = np.asarray(diagnostics["inverse_affine_to_original"], dtype=np.float32)
    item_dir = out_root / record.item_key
    item_dir.mkdir(parents=True, exist_ok=True)

    overlay = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
    palette = [
        (0, 145, 0),
        (210, 80, 0),
        (175, 0, 175),
        (0, 125, 220),
        (150, 105, 0),
        (20, 165, 165),
    ]
    region_outlines = [
        _region_outline_polygons(region, display_gray.shape) for region in regions
    ]
    for region_index, (region, outlines) in enumerate(
        zip(regions, region_outlines), start=1
    ):
        color = palette[(region_index - 1) % len(palette)]
        for outline_index, polygon in enumerate(outlines, start=1):
            cv2.polylines(
                overlay,
                [np.asarray(polygon, dtype=np.int32)],
                True,
                color,
                2,
                cv2.LINE_8,
            )
            x1, y1, _, _ = _polygon_box(polygon)
            label = (
                f"R{region_index}"
                if len(outlines) == 1
                else f"R{region_index}.{outline_index}"
            )
            cv2.putText(
                overlay,
                label,
                (x1 + 2, min(display_gray.shape[0] - 3, max(12, y1 + 12))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    original_path = item_dir / f"{record.item_key}.original.png"
    source_path = item_dir / f"{record.item_key}.source.png"
    overlay_path = item_dir / f"{record.item_key}.redaction_boxes.png"
    metadata_path = item_dir / f"{record.item_key}.redaction_boxes.json"
    cv2.imwrite(str(original_path), original_gray)
    cv2.imwrite(str(source_path), display_gray)
    cv2.imwrite(str(overlay_path), overlay)

    debug_paths: dict[str, str] = {}
    if save_debug_masks:
        debug_dir = item_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for key, mask in artifacts.items():
            if key == "deskewed_source":
                continue
            output = debug_dir / f"{key}.png"
            cv2.imwrite(str(output), mask)
            debug_paths[key] = str(output)

    original_h, original_w = original_gray.shape
    payload = {
        "item_key": record.item_key,
        "source_path": str(record.source_path),
        "rendered_image_path": str(record.rendered_image_path),
        "source_kind": record.source_kind,
        "pair_key": record.pair_key,
        "page_no_1based": record.page_no_1based,
        "image_size_wh": [int(display_gray.shape[1]), int(display_gray.shape[0])],
        "coordinate_system": "deskewed",
        "redaction_box_count": len(regions),
        "redaction_region_count": len(regions),
        "physical_component_count": sum(len(region.components) for region in regions),
        "redaction_boxes_xyxy": [list(map(int, region.box)) for region in regions],
        "redaction_regions": [
            {
                "region_id": f"R{region_index}",
                "bounds_xyxy": list(map(int, region.box)),
                "outline_polygons_xy": [
                    [list(map(int, point)) for point in polygon]
                    for polygon in region_outlines[region_index - 1]
                ],
                "original_image_outline_polygons_xy": [
                    _map_polygon_to_original(polygon, inverse, original_w, original_h)
                    for polygon in region_outlines[region_index - 1]
                ],
                "components": [
                    {
                        "component_id": f"R{region_index}.{component_index}",
                        "bounds_xyxy": list(map(int, component.box)),
                        "polygon_xy": [list(map(int, point)) for point in component.polygon],
                        "original_image_polygon_xy": _map_polygon_to_original(
                            component.polygon, inverse, original_w, original_h
                        ),
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
            "original_source_png": str(original_path),
            "source_png": str(source_path),
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
    rows: list[dict[str, Any]] = []
    for record in core_stage._progress(records, total=len(records), desc="Detect redaction boxes"):
        try:
            rows.append(process_record(record, out_root=out_root, save_debug_masks=save_debug_masks))
        except Exception as exc:
            rows.append(
                {
                    "item_key": record.item_key,
                    "source_path": str(record.source_path),
                    "rendered_image_path": str(record.rendered_image_path),
                    "error": str(exc),
                }
            )
    manifest_json = out_root / "box_manifest.json"
    manifest_jsonl = out_root / "box_manifest.jsonl"
    summary_json = out_root / "box_summary.json"
    manifest_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "stats": stats,
        "record_count": len(records),
        "success_count": sum("redaction_box_count" in row for row in rows),
        "error_count": sum("redaction_box_count" not in row for row in rows),
        "manifest_json": str(manifest_json),
        "manifest_jsonl": str(manifest_jsonl),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
