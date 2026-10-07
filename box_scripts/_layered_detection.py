from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from . import _core_detection as core_stage
from . import _geometry_detection as geometry_stage


# Input discovery and stable data structures are independent of detector version.
PDFPair = geometry_stage.PDFPair
PairCollectionStats = geometry_stage.PairCollectionStats
InputRecord = geometry_stage.InputRecord
BoxComponent = geometry_stage.BoxComponent
RedactionRegion = geometry_stage.RedactionRegion
AxisSegment = geometry_stage.AxisSegment
RectangleProposal = geometry_stage.RectangleProposal
OrientedSegment = geometry_stage.OrientedSegment
collect_pdf_pairs_with_stats = geometry_stage.collect_pdf_pairs_with_stats
render_pdf_to_images = geometry_stage.render_pdf_to_images
iter_input_records = geometry_stage.iter_input_records

_rect_polygon = geometry_stage._rect_polygon
_polygon_box = geometry_stage._polygon_box
_area = geometry_stage._area
_contains = geometry_stage._contains
_near_same_box = geometry_stage._near_same_box
_intersection_over_smaller = geometry_stage._intersection_over_smaller
_ink_fraction = geometry_stage._ink_fraction
_text_line_height = geometry_stage._text_line_height
_region_outline_polygons = geometry_stage._region_outline_polygons
_map_polygon_to_original = geometry_stage._map_polygon_to_original


def _oriented_line_segments(
    gray: np.ndarray,
) -> tuple[list[OrientedSegment], list[OrientedSegment]]:
    """Return near-axis borders with a direction-independent orientation.

    LSD may report the same physical line in either endpoint order.
    The geometry stage ordered endpoints but retained the pre-swap angle,
    causing parallel borders to appear as 0 and 180 degrees. The layered
    stage retains normalized endpoint ordering.
    """

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
        raw_angle = math.degrees(math.atan2(dy, dx))
        axis_angle = abs(((raw_angle + 90.0) % 180.0) - 90.0)
        if axis_angle <= 5.5 and length >= max(16.0, width * 0.018):
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            horizontal.append(
                OrientedSegment("h", (x1, y1), (x2, y2), length, angle)
            )
        elif abs(axis_angle - 90.0) <= 5.5 and length >= max(7.0, height * 0.006):
            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            vertical.append(
                OrientedSegment("v", (x1, y1), (x2, y2), length, angle)
            )
    return horizontal, vertical


def _lsd_geometry_rescues(
    gray: np.ndarray,
    line_height: int,
    existing: Iterable[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover warped, thin, and drawing-layer-clipped rectangles."""

    height, width = gray.shape
    raw_horizontal, raw_vertical = _oriented_line_segments(gray)
    horizontal = raw_horizontal + geometry_stage._merge_oriented_segments(
        raw_horizontal, "h", line_height, gray.shape
    )
    vertical = raw_vertical + geometry_stage._merge_oriented_segments(
        raw_vertical, "v", line_height, gray.shape
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 110)
    boundary_margin = max(4.0, min(height, width) * 0.018)
    # A transparent overlapping box can hide the corner-adjacent portion of a
    # side. Permit one text-line of lateral displacement; endpoint and blank-
    # interior guards still prevent nearby glyph stems from becoming sides.
    side_tolerance = max(7.0, line_height * 1.45)
    corner_tolerance = max(8.0, line_height * 0.55)
    minimum_height = max(5.0, line_height * 0.32)
    minimum_width = max(18.0, line_height * 1.15)
    raw: list[BoxComponent] = []

    for top_index, first in enumerate(horizontal):
        for second in horizontal[top_index + 1 :]:
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
            overlap = max(
                0.0, min(top_right, bottom_right) - max(top_left, bottom_left)
            )
            shorter = min(top_right - top_left, bottom_right - bottom_left)
            if overlap < max(10.0, shorter * 0.10):
                continue
            observed_left = min(top_left, bottom_left)
            observed_right = max(top_right, bottom_right)
            left_options = geometry_stage._candidate_side_positions(
                vertical, observed_left, top_y, bottom_y, side_tolerance
            )
            right_options = geometry_stage._candidate_side_positions(
                vertical, observed_right, top_y, bottom_y, side_tolerance
            )
            if observed_left <= boundary_margin:
                left_options.insert(0, (0.0, 1.0, True))
            if observed_right >= width - 1 - boundary_margin:
                right_options.insert(0, (float(width - 1), 1.0, True))

            for left_position, left_support, left_observed in left_options:
                for right_position, right_support, right_observed in right_options:
                    box_width = right_position - left_position
                    if box_width < minimum_width:
                        continue
                    tall_narrow = box_width < box_height * 0.58
                    top_support = geometry_stage._oriented_horizontal_coverage(
                        top, left_position, right_position
                    )
                    bottom_support = geometry_stage._oriented_horizontal_coverage(
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
                    quality = geometry_stage._robust_interior_quality(
                        gray, edges, box, line_height
                    )
                    if quality is None:
                        continue

                    corner_anchors = 0
                    for y_value, horizontal_segment in (
                        (top_y, top),
                        (bottom_y, bottom),
                    ):
                        for x_value, side_observed in (
                            (left_position, left_observed or left_boundary),
                            (right_position, right_observed or right_boundary),
                        ):
                            horizontal_anchor = geometry_stage._endpoint_near(
                                x_value, horizontal_segment, 0, corner_tolerance
                            )
                            if x_value <= 0.5 or x_value >= width - 1.5:
                                vertical_anchor = True
                            else:
                                vertical_anchor = any(
                                    abs(segment.midpoint[0] - x_value)
                                    <= side_tolerance
                                    and geometry_stage._endpoint_near(
                                        y_value, segment, 1, corner_tolerance
                                    )
                                    for segment in vertical
                                )
                            if horizontal_anchor and vertical_anchor and side_observed:
                                corner_anchors += 1
                    required_anchors = 4 if tall_narrow else (3 if thin else 2)
                    if corner_anchors < required_anchors:
                        continue
                    if tall_narrow and (
                        min(top_support, bottom_support) < 0.62
                        or min(left_support, right_support) < 0.62
                        or not (left_observed and right_observed)
                    ):
                        continue

                    mean, _, _, blank_rows = quality
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

    deduplicated = geometry_stage._dedupe_components(raw, gray.shape)
    ordered = sorted(
        deduplicated, key=lambda item: (item.score, _area(item.box)), reverse=True
    )
    compact: list[BoxComponent] = []
    containment_margin = max(3, round(min(height, width) * 0.006))
    for candidate in ordered:
        if any(
            _contains(previous.box, candidate.box, margin=containment_margin)
            and _area(previous.box) >= _area(candidate.box) * 1.08
            and previous.score >= candidate.score - 0.25
            for previous in compact
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


def _shape_preserving_solid_candidates(
    gray: np.ndarray, line_height: int
) -> tuple[list[BoxComponent], np.ndarray]:
    """Keep observed blackout contours, including large rectilinear steps.

    The legacy solid detector deliberately required a mostly filled bounding
    rectangle. That is appropriate for ordinary bars, but it drops an L-shaped
    or stair-stepped blackout even when every retained pixel belongs to one
    coherent mask. The layered stage adds a second, stricter shape channel over the same
    page-local binary mask. It accepts only large connected shapes with long
    dark row cores and predominantly axis-aligned boundaries; it never uses a
    document identifier, OCR token, paired page, or annotation.
    """

    candidates, solid_mask = core_stage._solid_candidates(gray)
    raw_solid = ((gray < 72).astype(np.uint8) * 255)
    measured_solid = cv2.morphologyEx(
        raw_solid,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    measured_solid = _remove_narrow_mask_attachments(measured_solid, line_height)
    height, width = gray.shape
    edge_margin = max(3, round(min(height, width) * 0.018))
    retained: list[BoxComponent] = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.box
        if (
            x1 <= edge_margin
            and x2 >= width - edge_margin
            and y2 - y1 < line_height * 2.5
        ):
            continue
        measured_polygon = _adaptive_mask_polygon(
            measured_solid,
            candidate.box,
            gray.shape,
        )
        polygon = measured_polygon or candidate.polygon
        polygon_box = _polygon_box(polygon)
        polygon_area = abs(
            cv2.contourArea(np.asarray(polygon, dtype=np.int32))
        )
        envelope_area = max(1, _area(polygon_box))
        nearly_rectangular = (
            len(polygon) == 4 and polygon_area / envelope_area >= 0.88
        )
        retained.append(
            BoxComponent(
                box=polygon_box,
                polygon=(
                    _rect_polygon(polygon_box)
                    if nearly_rectangular
                    else polygon
                ),
                source=(
                    "solid_fill_rectified"
                    if nearly_rectangular
                    else "solid_fill_shape_preserved"
                ),
                score=candidate.score,
            )
        )

    # Recover coherent stepped blackouts whose concavity makes their envelope
    # too sparse for the ordinary rectangular-fill threshold. Morphological
    # opening has already removed character-width attachments, so the remaining
    # contour must still satisfy strong scale-relative area, row-core, and
    # rectilinearity requirements.
    contours, _ = cv2.findContours(
        measured_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    page_area = height * width
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        box = (x, y, x + box_width, y + box_height)
        envelope_area = box_width * box_height
        if (
            box_width < max(55, round(line_height * 2.8))
            # This channel is only for multi-line, non-rectangular blackout
            # regions. Ordinary bars and underlines belong to the stricter
            # rectangular solid detector above.
            or box_height < max(18, round(line_height * 1.80))
            or envelope_area < max(1200, round(page_area * 0.00045))
            or any(
                _near_same_box(box, component.box, max(3, round(min(gray.shape) * 0.006)))
                for component in retained
            )
        ):
            continue

        roi = measured_solid[y : y + box_height, x : x + box_width] > 0
        if not roi.size:
            continue
        fill = float(np.mean(roi))
        contour_fill = float(cv2.contourArea(contour) / max(1, envelope_area))
        row_occupancy = np.mean(roi, axis=1)
        coherent_rows = float(np.mean(row_occupancy >= 0.55))
        broad_rows = float(np.mean(row_occupancy >= 0.30))
        column_occupancy = np.mean(roi, axis=0)
        coherent_columns = float(np.mean(column_occupancy >= 0.30))
        if (
            fill < 0.42
            or contour_fill < 0.38
            or coherent_rows < 0.45
            or broad_rows < 0.72
            or coherent_columns < 0.55
            or float(np.max(row_occupancy)) < 0.80
        ):
            continue

        polygon = _adaptive_mask_polygon(measured_solid, box, gray.shape)
        if len(polygon) <= 4:
            continue
        polygon_area = abs(cv2.contourArea(np.asarray(polygon, dtype=np.int32)))
        if polygon_area / max(1, envelope_area) >= 0.88:
            continue
        axis_aligned = 0
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            if dx == 0 and dy == 0:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx))) % 90.0
            distance_to_axis = min(angle, 90.0 - angle)
            axis_aligned += int(distance_to_axis <= 12.0)
        if axis_aligned / max(1, len(polygon)) < 0.70:
            continue

        retained.append(
            BoxComponent(
                box=_polygon_box(polygon),
                polygon=polygon,
                source="solid_fill_rectilinear_step_rescue",
                score=(
                    5.0
                    + fill
                    + contour_fill
                    + coherent_rows
                    + broad_rows
                ),
            )
        )

    retained = geometry_stage._dedupe_components(retained, gray.shape)
    return retained, solid_mask


def _remove_narrow_mask_attachments(
    mask: np.ndarray, line_height: int
) -> np.ndarray:
    """Remove glyph-width attachments without closing genuine mask notches.

    Dense redactions can touch printed letters at their boundary. Those thin
    attachments should not become part of the recovered outline. The opening
    kernel is bounded and derived only from the page's measured text scale.
    """

    kernel_size = max(3, min(9, int(round(line_height * 0.15))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        ),
        iterations=1,
    )


def _adaptive_mask_polygon(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return a compact contour while retaining material concave steps.

    The approximation budget is fixed across pages. It selects the most
    detailed contour with at most 16 vertices, which is enough for several
    right-angle steps but too small to trace individual letter shapes.
    """

    x1, y1, x2, y2 = box
    roi = mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return tuple()
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return tuple()

    approximation: np.ndarray | None = None
    for ratio in (
        0.0025,
        0.0030,
        0.0035,
        0.0040,
        0.0045,
        0.0050,
        0.0055,
        0.0060,
        0.0070,
        0.0080,
        0.0100,
        0.0120,
    ):
        candidate = cv2.approxPolyDP(
            contour, max(1.0, ratio * perimeter), True
        ).reshape((-1, 2))
        if 4 <= len(candidate) <= 16:
            approximation = candidate
            break
    if approximation is None:
        return tuple()

    height, width = image_shape
    polygon: list[tuple[int, int]] = []
    for raw_x, raw_y in approximation:
        point = (
            max(0, min(width - 1, int(raw_x) + x1)),
            max(0, min(height - 1, int(raw_y) + y1)),
        )
        if not polygon or point != polygon[-1]:
            polygon.append(point)
    if len(polygon) > 2 and polygon[0] == polygon[-1]:
        polygon.pop()
    return tuple(polygon) if 4 <= len(polygon) <= 16 else tuple()


def _axis_segment_support(
    segments: Iterable[OrientedSegment],
    *,
    orientation: str,
    fixed_position: float,
    start: float,
    end: float,
    tolerance: float,
) -> float:
    """Measure union coverage along one expected axis-aligned border."""

    intervals: list[tuple[float, float]] = []
    for segment in segments:
        if segment.orientation != orientation:
            continue
        if orientation == "h":
            if abs(segment.midpoint[1] - fixed_position) > tolerance:
                continue
            left = max(start, min(segment.start[0], segment.end[0]))
            right = min(end, max(segment.start[0], segment.end[0]))
        else:
            if abs(segment.midpoint[0] - fixed_position) > tolerance:
                continue
            left = max(start, min(segment.start[1], segment.end[1]))
            right = min(end, max(segment.start[1], segment.end[1]))
        if right > left:
            intervals.append((left, right))
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1] + tolerance:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    covered = sum(right - left for left, right in merged)
    return float(min(1.0, covered / max(1.0, end - start)))


def _polygon_union_with_boxes(
    component: BoxComponent,
    boxes: Iterable[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return a measured union outline without replacing it by its envelope."""

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(component.polygon, dtype=np.int32)], 255)
    all_boxes = [component.box]
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
        all_boxes.append((x1, y1, x2, y2))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    bounds = (
        min(box[0] for box in all_boxes),
        min(box[1] for box in all_boxes),
        max(box[2] for box in all_boxes),
        max(box[3] for box in all_boxes),
    )
    return _adaptive_mask_polygon(mask, bounds, image_shape)


def _shared_edge_step_completions(
    gray: np.ndarray,
    line_height: int,
    components: list[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray, int]:
    """Complete a blank tab that shares an outer side with a known component.

    Layered redaction masks sometimes form one stepped polygon: the smaller tab
    has an observed opposite edge and outer edge, while its inner side is not a
    vertical line at all. Requiring four rectangle sides therefore loses the
    tab. This routine permits that one absent inner side only when the tab is
    contained within the component's horizontal span, shares its outer side,
    has a blank interior, and is not already explained by another component.
    """

    raw_horizontal, raw_vertical = _oriented_line_segments(gray)
    horizontal = raw_horizontal + geometry_stage._merge_oriented_segments(
        raw_horizontal, "h", line_height, gray.shape
    )
    vertical = raw_vertical + geometry_stage._merge_oriented_segments(
        raw_vertical, "v", line_height, gray.shape
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 110)
    position_tolerance = max(3.0, line_height * 0.36)
    minimum_width = max(12.0, line_height * 1.15)
    minimum_height = max(4.0, line_height * 0.30)
    maximum_height = line_height * 4.0
    output: list[BoxComponent] = []
    completion_mask = np.zeros_like(gray)
    completion_count = 0

    for component_index, component in enumerate(components):
        bx1, by1, bx2, by2 = component.box
        other_components = [
            other for index, other in enumerate(components) if index != component_index
        ]
        tab_candidates: list[BoxComponent] = []
        for segment in horizontal:
            segment_y = segment.midpoint[1]
            segment_left = min(segment.start[0], segment.end[0])
            segment_right = max(segment.start[0], segment.end[0])
            if segment_right - segment_left < minimum_width:
                continue

            if by1 - maximum_height <= segment_y <= by1 - minimum_height:
                vertical_target = by1
                tab_top = int(round(segment_y))
                tab_bottom = by1 + 1
            elif by2 + minimum_height <= segment_y <= by2 + maximum_height:
                vertical_target = by2 - 1
                tab_top = by2 - 1
                tab_bottom = int(round(segment_y)) + 1
            else:
                continue
            if tab_bottom - tab_top < minimum_height:
                continue

            side_options = (
                ("left", bx1, segment_left, segment_right),
                ("right", bx2 - 1, segment_left, segment_right),
            )
            for side_name, outer_x, left_x, right_x in side_options:
                endpoint = left_x if side_name == "left" else right_x
                if abs(endpoint - outer_x) > position_tolerance:
                    continue
                inner_x = right_x if side_name == "left" else left_x
                if not bx1 + minimum_width <= inner_x <= bx2 - minimum_width:
                    continue
                cap_left = bx1 if side_name == "left" else int(round(inner_x))
                cap_right = int(round(inner_x)) + 1 if side_name == "left" else bx2
                if cap_right - cap_left < minimum_width:
                    continue
                cap_box = (cap_left, tab_top, cap_right, tab_bottom)
                cap = BoxComponent(
                    box=cap_box,
                    polygon=_rect_polygon(cap_box),
                    source="shared_edge_step_candidate",
                    score=0.0,
                )
                if geometry_stage._candidate_coverage(cap, other_components) >= 0.40:
                    continue
                outer_support = _axis_segment_support(
                    vertical,
                    orientation="v",
                    fixed_position=float(outer_x),
                    start=float(tab_top),
                    end=float(tab_bottom),
                    tolerance=position_tolerance,
                )
                opposite_support = _axis_segment_support(
                    horizontal,
                    orientation="h",
                    fixed_position=float(segment_y),
                    start=float(cap_left),
                    end=float(cap_right),
                    tolerance=position_tolerance * 0.55,
                )
                if outer_support < 0.55 or opposite_support < 0.70:
                    continue
                quality = geometry_stage._robust_interior_quality(
                    gray, edges, cap_box, line_height
                )
                if quality is None:
                    continue
                mean, dark_fraction, edge_fraction, blank_rows = quality
                tab_candidates.append(
                    BoxComponent(
                        box=cap_box,
                        polygon=_rect_polygon(cap_box),
                        source=(
                            "shared_edge_step_candidate:"
                            f"outer={outer_support:.2f}:"
                            f"opposite={opposite_support:.2f}"
                        ),
                        score=(
                            outer_support
                            + opposite_support
                            + blank_rows
                            + max(0.0, (mean - 180.0) / 75.0)
                            - dark_fraction
                            - edge_fraction
                        ),
                    )
                )

        unique_tabs = geometry_stage._dedupe_components(tab_candidates, gray.shape)
        if not unique_tabs:
            output.append(component)
            continue
        # Prefer the most strongly supported cap when duplicate line segments
        # describe the same tab. Distinct tabs may still augment one component.
        selected: list[BoxComponent] = []
        for tab in sorted(unique_tabs, key=lambda item: item.score, reverse=True):
            if any(_near_same_box(tab.box, prior.box, int(position_tolerance)) for prior in selected):
                continue
            selected.append(tab)
        union_polygon = _polygon_union_with_boxes(
            component, [tab.box for tab in selected], gray.shape
        )
        if len(union_polygon) < 4:
            output.append(component)
            continue
        completion_count += len(selected)
        for tab in selected:
            cv2.rectangle(
                completion_mask,
                (tab.box[0], tab.box[1]),
                (tab.box[2] - 1, tab.box[3] - 1),
                255,
                1,
            )
        output.append(
            BoxComponent(
                box=_polygon_box(union_polygon),
                polygon=union_polygon,
                source=(
                    "shared_edge_step_completion:"
                    f"tabs={len(selected)}:{component.source}"
                ),
                score=component.score + max(tab.score for tab in selected) * 0.05,
            )
        )
    return output, completion_mask, completion_count


def _shape_preserving_blackout_candidates(
    gray: np.ndarray, line_height: int, existing: Iterable[BoxComponent]
) -> tuple[list[BoxComponent], np.ndarray]:
    """Preserve the measured outline for dense masks missed by solid-fill CV."""

    candidates, joined = geometry_stage._dense_blackout_candidates(gray, line_height, existing)
    measured_joined = _remove_narrow_mask_attachments(joined, line_height)
    shaped: list[BoxComponent] = []
    for candidate in candidates:
        polygon = _adaptive_mask_polygon(
            measured_joined,
            candidate.box,
            gray.shape,
        )
        polygon = polygon or candidate.polygon
        shaped.append(
            BoxComponent(
                box=_polygon_box(polygon),
                polygon=polygon,
                source="dense_blackout_shape_preserved",
                score=candidate.score,
            )
        )
    return shaped, joined


def _promote_supported_outer_rectangles(
    base: list[BoxComponent],
    candidates: list[BoxComponent],
    image_shape: tuple[int, int],
    line_height: int,
) -> tuple[list[BoxComponent], list[BoxComponent], int]:
    """Replace a clipped child when a strongly anchored outer border exists."""

    tolerance = max(4, round(min(image_shape) * 0.008))
    replacements: dict[int, tuple[int, BoxComponent]] = {}
    for candidate_index, candidate in enumerate(candidates):
        corners = geometry_stage._lsd_corner_count(candidate)
        support_match = re.search(
            r"h=([0-9.]+)/([0-9.]+):v=([0-9.]+)/([0-9.]+)",
            candidate.source,
        )
        supports = (
            tuple(float(value) for value in support_match.groups())
            if support_match
            else tuple()
        )
        # Promotion replaces an already plausible child, so it intentionally
        # has a higher burden than ordinary candidate recovery: all four
        # corners and all four sides must be independently well supported.
        if (
            corners < 4
            or len(supports) != 4
            or min(supports) < 0.75
            or candidate.score < 5.4
        ):
            continue
        for base_index, child in enumerate(base):
            if not _contains(candidate.box, child.box, margin=tolerance):
                continue
            ratio = _area(candidate.box) / max(1, _area(child.box))
            if ratio < 1.14 or ratio > 4.5:
                continue
            aligned = sum(
                abs(left - right) <= tolerance
                for left, right in zip(candidate.box, child.box)
            )
            if aligned < 3:
                continue
            current = replacements.get(base_index)
            if current is None or candidate.score > current[1].score:
                replacements[base_index] = (candidate_index, candidate)

    used: set[int] = set()
    promoted: list[BoxComponent] = []
    for index, component in enumerate(base):
        replacement = replacements.get(index)
        if replacement is None:
            promoted.append(component)
            continue
        candidate_index, candidate = replacement
        used.add(candidate_index)
        promoted.append(
            BoxComponent(
                box=candidate.box,
                polygon=candidate.polygon,
                source=f"lsd_outer_border_promotion:{candidate.source}",
                score=max(component.score, candidate.score) + 0.05,
            )
        )
    remaining = [
        candidate for index, candidate in enumerate(candidates) if index not in used
    ]
    return promoted, remaining, len(used)


def _lsd_support_values(component: BoxComponent) -> tuple[float, ...]:
    support_match = re.search(
        r"h=([0-9.]+)/([0-9.]+):v=([0-9.]+)/([0-9.]+)",
        component.source,
    )
    if support_match is None:
        return tuple()
    return tuple(float(value) for value in support_match.groups())


def _layer_occlusion_rescues(
    candidates: list[BoxComponent],
    existing: list[BoxComponent],
    image_shape: tuple[int, int],
    line_height: int,
) -> list[BoxComponent]:
    """Retain a rectangle whose two corners are hidden by a touching layer."""

    ordinary = geometry_stage._novel_rescues(candidates, existing, image_shape)
    accepted = list(ordinary)
    tolerance = max(3, round(min(image_shape) * 0.006))
    for candidate in candidates:
        if geometry_stage._lsd_corner_count(candidate) != 2:
            continue
        supports = _lsd_support_values(candidate)
        if len(supports) != 4:
            continue
        x1, y1, x2, y2 = candidate.box
        box_height = y2 - y1
        if not line_height * 0.45 <= box_height <= line_height * 4.0:
            continue
        if any(_near_same_box(candidate.box, item.box, tolerance) for item in existing + accepted):
            continue
        width = max(1, x2 - x1)
        touches_supported_layer = False
        for other in existing + ordinary:
            ox1, oy1, ox2, oy2 = other.box
            overlap_width = max(0, min(x2, ox2) - max(x1, ox1))
            overlap_height = max(0, min(y2, oy2) - max(y1, oy1))
            boundary_gap = min(abs(y2 - oy1), abs(oy2 - y1))
            strong_touching_sides = min(supports) >= 0.88
            staircase_overlap = (
                sum(value >= 0.90 for value in supports) >= 2
                and min(supports) >= 0.40
                and sum(supports) / 4.0 >= 0.70
                and _intersection_over_smaller(candidate.box, other.box) >= 0.08
                and x1 > ox1 + line_height
                and x2 > ox2 + line_height
                and y1 < oy1 - line_height * 0.45
                and y2 < oy2
            )
            # A drawing layer can erase one candidate corner rather than a
            # whole side. Retain that rectangle when three sides remain
            # materially supported, the overlap is localized to one corner,
            # and the candidate extends beyond the known box on both axes.
            # This is intentionally stricter than ordinary two-corner rescue.
            candidate_area = max(1, _area(candidate.box))
            localized_overlap = (overlap_width * overlap_height) / candidate_area
            substantially_covers_known_component = any(
                _intersection_area(candidate.box, known.box) / candidate_area >= 0.25
                or _intersection_over_smaller(candidate.box, known.box) >= 0.70
                for known in existing + ordinary
            )
            horizontal_extension = max(0, x2 - ox2, ox1 - x1)
            vertical_extension = max(0, y2 - oy2, oy1 - y1)
            corner_occlusion = (
                candidate.score >= 5.2
                and sum(value >= 0.62 for value in supports) >= 3
                and max(supports) >= 0.90
                and sum(supports) / 4.0 >= 0.68
                and 0.015 <= localized_overlap <= 0.20
                and overlap_width <= width * 0.38
                and overlap_height <= box_height * 0.58
                and horizontal_extension >= line_height * 0.40
                and vertical_extension >= line_height * 0.40
                and not substantially_covers_known_component
            )
            if staircase_overlap or corner_occlusion or (
                strong_touching_sides
                and boundary_gap <= max(tolerance, round(line_height * 0.28))
                and overlap_width / width >= 0.55
            ):
                touches_supported_layer = True
                break
        if touches_supported_layer:
            accepted.append(
                BoxComponent(
                    box=candidate.box,
                    polygon=candidate.polygon,
                    source=f"lsd_layer_occlusion_rescue:{candidate.source}",
                    score=candidate.score,
                )
            )
    return geometry_stage._dedupe_components(accepted, image_shape)


def _suppress_lsd_border_echoes(
    components: list[BoxComponent], image_shape: tuple[int, int], line_height: int
) -> list[BoxComponent]:
    """Suppress weak line echoes while preserving complete thin rectangles."""

    filtered = geometry_stage._suppress_lsd_border_echoes(components, image_shape, line_height)
    protected = []
    for component in components:
        if not component.source.startswith("lsd_geometry_rescue"):
            continue
        supports = _lsd_support_values(component)
        height = component.box[3] - component.box[1]
        if (
            geometry_stage._lsd_corner_count(component) >= 3
            and len(supports) == 4
            and min(supports) >= 0.75
            and height >= line_height * 0.45
        ):
            protected.append(component)
    return geometry_stage._dedupe_components(filtered + protected, image_shape)


def _intersection_area(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> int:
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _has_material_overlap(
    index: int, components: list[BoxComponent], line_height: int
) -> bool:
    candidate = components[index]
    for other_index, other in enumerate(components):
        if index == other_index:
            continue
        intersection = _intersection_area(candidate.box, other.box)
        if intersection >= max(
            line_height * line_height,
            min(_area(candidate.box), _area(other.box)) * 0.08,
        ):
            return True
    return False


def _reliable_lsd_rectangle(
    component: BoxComponent, line_height: int
) -> bool:
    """Return whether independent line geometry outweighs residual text ink."""

    if "lsd_" not in component.source or geometry_stage._lsd_corner_count(component) < 3:
        return False
    supports = _lsd_support_values(component)
    if len(supports) != 4:
        return False
    width = component.box[2] - component.box[0]
    height = component.box[3] - component.box[1]
    return (
        min(supports) >= 0.50
        and sum(supports) / 4.0 >= 0.80
        and width >= line_height * 1.30
        and height >= line_height * 0.60
    )


def _boundary_text_crossing(
    gray: np.ndarray,
    line_mask: np.ndarray,
    box: tuple[int, int, int, int],
    line_height: int,
) -> float:
    """Measure glyph strokes crossing a proposed horizontal frame edge."""

    x1, y1, x2, y2 = box
    radius = max(3, round(line_height * 0.32))
    raw_ink = gray < 175
    expanded_lines = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    ink = np.logical_and(raw_ink, expanded_lines == 0)
    best = 0.0
    for edge_y in (y1, y2 - 1):
        if edge_y - radius < 0 or edge_y + radius + 1 > gray.shape[0]:
            continue
        left = max(0, x1 + 2)
        right = min(gray.shape[1], x2 - 2)
        if right <= left:
            continue
        above = np.any(ink[edge_y - radius : edge_y, left:right], axis=0)
        below = np.any(ink[edge_y + 1 : edge_y + radius + 1, left:right], axis=0)
        shared = np.logical_and(above, below)

        # A release stamp can itself be absorbed into the long-line mask. A
        # second measurement erases only the candidate's narrow border strip,
        # retaining glyph strokes on both sides without counting the border.
        raw_above = np.any(
            raw_ink[edge_y - radius : max(edge_y - 2, edge_y - radius), left:right],
            axis=0,
        )
        raw_below = np.any(
            raw_ink[min(edge_y + 3, edge_y + radius) : edge_y + radius + 1, left:right],
            axis=0,
        )
        raw_shared = np.logical_and(raw_above, raw_below)
        best = max(best, float(np.mean(shared)), float(np.mean(raw_shared)))
    return best


def _suppress_text_frames_and_glyphs(
    gray: np.ndarray,
    line_mask: np.ndarray,
    components: list[BoxComponent],
    line_height: int,
) -> tuple[list[BoxComponent], dict[str, int]]:
    """Reject isolated framed text and glyph contours using page geometry."""

    height, image_width = gray.shape
    remove: set[int] = set()
    reasons: dict[str, int] = {
        "framed_text": 0,
        "marginal_text_crossing": 0,
        "small_glyph_contour": 0,
    }
    for index, component in enumerate(components):
        material_overlap = _has_material_overlap(index, components, line_height)
        x1, y1, x2, y2 = component.box
        width, box_height = x2 - x1, y2 - y1
        source = component.source
        marginal = y1 <= height * 0.16 or y2 >= height * 0.90
        reliable_lsd = _reliable_lsd_rectangle(component, line_height)
        fully_observed_axis_box = (
            (
                source.startswith("rectilinear_zone_")
                or (
                    source.startswith("lsd_corner_refinement:")
                    and ":base=rectilinear_zone_" in source
                )
            )
            and ":observed=4:virtual=0" in source
        )
        closed_support = re.search(
            r"^closed_blank_contour:support=([0-9.]+):fill=([0-9.]+)",
            source,
        )
        strong_large_closed_contour = bool(
            closed_support
            and float(closed_support.group(1)) >= 0.58
            and float(closed_support.group(2)) >= 0.85
            and width >= line_height * 6.0
            and box_height >= line_height * 2.0
        )
        if (
            source.startswith(("closed_blank_contour", "enclosed_blank_outline"))
            and width < line_height * 1.65
            and box_height < line_height * 1.75
        ):
            remove.add(index)
            reasons["small_glyph_contour"] += 1
            continue

        pad = max(2, min(5, box_height // 7))
        local_polygon = np.asarray(
            [(x - x1, y - y1) for x, y in component.polygon], dtype=np.int32
        )
        polygon_mask = np.zeros((box_height, width), dtype=np.uint8)
        if len(local_polygon) >= 3:
            cv2.fillPoly(polygon_mask, [local_polygon], 255)
        if pad > 0:
            polygon_mask = cv2.erode(
                polygon_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (pad * 2 + 1, pad * 2 + 1)
                ),
                iterations=1,
            )
        interior = gray[y1:y2, x1:x2]
        valid = polygon_mask > 0
        if interior.size and np.any(valid):
            ink = np.logical_and(interior < 175, valid)
            dark_fraction = float(np.count_nonzero(ink) / np.count_nonzero(valid))
            row_denominators = np.count_nonzero(valid, axis=1)
            row_ink = np.count_nonzero(ink, axis=1)
            row_density = np.divide(
                row_ink,
                row_denominators,
                out=np.zeros_like(row_ink, dtype=np.float64),
                where=row_denominators > 0,
            )
            active = row_density > 0.08
            longest = 0
            run = 0
            for value in active:
                if value:
                    run += 1
                    longest = max(longest, run)
                else:
                    run = 0
            occupancy = float(np.mean(active[row_denominators > 0]))
            distributed_text = (
                dark_fraction >= 0.055
                and occupancy >= 0.16
                and longest >= max(8, round(line_height * 0.25))
            )
            solid_mask = source.startswith(
                ("solid_fill", "dense_blackout")
            )
            top_text_frame = y2 <= height * 0.14
            if (
                distributed_text
                and not solid_mask
                and not (
                    (
                        reliable_lsd
                        and not top_text_frame
                        and (not marginal or width < image_width * 0.20)
                        # A high-confidence line rectangle may retain sparse
                        # scan residue. It must not, however, survive merely
                        # because it overlaps another candidate while bridging
                        # a substantial run of visible prose.
                        and not (
                            material_overlap
                            and dark_fraction >= 0.070
                            and occupancy >= 0.35
                            and geometry_stage._lsd_corner_count(component) < 4
                            and box_height <= line_height * 1.80
                            and width / max(1, box_height) >= 8.0
                            and (
                                not _lsd_support_values(component)
                                or min(_lsd_support_values(component)) < 0.75
                            )
                        )
                    )
                    # Multi-line boxes can contain released text after two
                    # masking layers overlap. Preserve only compact, strongly
                    # corner-supported shapes; thin line-like bridges remain
                    # subject to the text test above.
                    or (
                        material_overlap
                        and box_height >= line_height * 2.50
                        and width / max(1, box_height) < 8.0
                        and geometry_stage._lsd_corner_count(component) >= 3
                        and bool(_lsd_support_values(component))
                        and max(_lsd_support_values(component)) >= 0.95
                    )
                    or (material_overlap and fully_observed_axis_box)
                )
            ):
                remove.add(index)
                reasons["framed_text"] += 1
                continue

        if (
            marginal
            and box_height >= line_height * 0.90
            and not source.startswith(("solid_fill", "dense_blackout"))
            and not reliable_lsd
            and not strong_large_closed_contour
            and _boundary_text_crossing(
                gray, line_mask, component.box, line_height
            )
            >= 0.40
        ):
            remove.add(index)
            reasons["marginal_text_crossing"] += 1

    return (
        [component for index, component in enumerate(components) if index not in remove],
        reasons,
    )


def _text_mask_and_margins(
    gray: np.ndarray,
    line_mask: np.ndarray,
    components: list[BoxComponent],
) -> tuple[np.ndarray, int, int, int]:
    """Estimate body margins and line pitch from substantial text lines."""

    height, width = gray.shape
    line_height = _text_line_height(gray, line_mask)
    text = ((gray < 170).astype(np.uint8) * 255)
    expanded_lines = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    text[expanded_lines > 0] = 0
    for component in components:
        x1, y1, x2, y2 = component.box
        text[
            max(0, y1 - 2) : min(height, y2 + 2),
            max(0, x1 - 2) : min(width, x2 + 2),
        ] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text, connectivity=8
    )
    glyphs: list[tuple[int, int, int, int, int, float]] = []
    clean = np.zeros_like(text)
    for label in range(1, count):
        x, y, glyph_width, glyph_height, area = map(int, stats[label])
        if (
            area < 3
            or glyph_width > width * 0.10
            or glyph_height > height * 0.08
            or glyph_height > line_height * 2.2
        ):
            continue
        clean[labels == label] = 255
        glyphs.append(
            (x, y, glyph_width, glyph_height, area, float(centroids[label][1]))
        )

    clusters: list[list[tuple[int, int, int, int, int, float]]] = []
    cluster_tolerance = max(3.0, line_height * 0.52)
    for glyph in sorted(glyphs, key=lambda item: item[5]):
        best_index: int | None = None
        best_distance = float("inf")
        for index, cluster in enumerate(clusters):
            center = float(np.median([item[5] for item in cluster]))
            distance = abs(glyph[5] - center)
            if distance <= cluster_tolerance and distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is None:
            clusters.append([glyph])
        else:
            clusters[best_index].append(glyph)

    def horizontal_runs(
        cluster: list[tuple[int, int, int, int, int, float]],
    ) -> list[list[tuple[int, int, int, int, int, float]]]:
        ordered = sorted(cluster, key=lambda item: item[0])
        if not ordered:
            return []
        gap_limit = max(line_height * 2.0, width * 0.025)
        runs = [[ordered[0]]]
        run_right = ordered[0][0] + ordered[0][2]
        for glyph in ordered[1:]:
            if glyph[0] - run_right > gap_limit:
                runs.append([glyph])
            else:
                runs[-1].append(glyph)
            run_right = max(run_right, glyph[0] + glyph[2])
        return runs

    def marginal_short_label(
        run: list[tuple[int, int, int, int, int, float]],
    ) -> bool:
        run_left = min(item[0] for item in run)
        run_right = max(item[0] + item[2] for item in run)
        span = run_right - run_left
        return span < width * 0.12 and (
            run_right < width * 0.22 or run_left > width * 0.78
        )

    # Short alphanumeric handling marks are isolated from the main text
    # column. Split same-baseline text at large geometric gaps so a marginal
    # mark cannot widen an otherwise normal prose line to the full page.
    marginal_runs: list[list[tuple[int, int, int, int, int, float]]] = []
    all_runs: list[list[tuple[int, int, int, int, int, float]]] = []
    body_clusters: list[list[tuple[int, int, int, int, int, float]]] = []
    for cluster in clusters:
        runs = horizontal_runs(cluster)
        all_runs.extend(runs)
        run_weights = [sum(item[2] for item in run) for run in runs]
        retained: list[tuple[int, int, int, int, int, float]] = []
        for run_index, run in enumerate(runs):
            has_body_companion = any(
                other_index != run_index
                and other_weight >= max(width * 0.05, run_weights[run_index] * 1.5)
                for other_index, other_weight in enumerate(run_weights)
            )
            run_left = min(item[0] for item in run)
            run_right = max(item[0] + item[2] for item in run)
            far_edge = run_right < width * 0.12 or run_left > width * 0.88
            if marginal_short_label(run) and (has_body_companion or far_edge):
                marginal_runs.append(run)
            else:
                retained.extend(run)
        if retained:
            body_clusters.append(retained)
    substantial: list[tuple[int, int, float, list[tuple[int, int, int, int, int, float]]]] = []
    for cluster in body_clusters:
        left = min(item[0] for item in cluster)
        right = max(item[0] + item[2] for item in cluster)
        center = float(np.median([item[5] for item in cluster]))
        glyph_width_sum = sum(item[2] for item in cluster)
        if (
            len(cluster) >= 3
            and (right - left >= width * 0.12 or glyph_width_sum >= width * 0.08)
        ):
            substantial.append((left, right, center, cluster))

    if substantial:
        # Inner quartiles recover the dominant prose column while ignoring a
        # few full-width release stamps, page numbers, and indented lines.
        left = max(0, int(round(np.percentile([item[0] for item in substantial], 25))))
        right = min(
            width - 1,
            int(round(np.percentile([item[1] for item in substantial], 75))),
        )
    else:
        left, right = 0, width - 1

    centers = sorted(item[2] for item in substantial)
    gaps = [
        current - previous
        for previous, current in zip(centers, centers[1:])
        if line_height * 0.78 <= current - previous <= line_height * 2.20
    ]
    pitch = (
        int(round(float(np.median(gaps))))
        if gaps
        else int(round(line_height * 1.35))
    )
    pitch = max(
        int(round(line_height * 1.05)),
        min(int(round(line_height * 1.85)), pitch),
    )

    # Isolated short labels outside the body bounds are release markings, not
    # intervening prose. Remove them only from semantic-continuation checks.
    for run in all_runs:
        run_left = min(item[0] for item in run)
        run_right = max(item[0] + item[2] for item in run)
        span = run_right - run_left
        outside_body = run_right < left or run_left > right
        if run in marginal_runs or (outside_body and span < width * 0.12):
            for x, y, glyph_width, glyph_height, _, _ in run:
                clean[y : y + glyph_height, x : x + glyph_width] = 0
    return clean, left, right, pitch


def _deep_crossing_is_distinct(
    first: BoxComponent, second: BoxComponent, line_height: int
) -> bool:
    """Keep tall-column and broad-block masks separate despite overlap."""

    for tall, broad in ((first, second), (second, first)):
        tx1, ty1, tx2, ty2 = tall.box
        bx1, by1, bx2, by2 = broad.box
        tall_width, tall_height = tx2 - tx1, ty2 - ty1
        broad_width, broad_height = bx2 - bx1, by2 - by1
        intersection = _intersection_area(tall.box, broad.box)
        if (
            tall_height >= line_height * 5.0
            and tall_height >= tall_width * 1.8
            and broad_width >= tall_width * 3.0
            and broad_height >= line_height * 4.0
            and intersection / max(1, _area(tall.box)) >= 0.35
            and intersection / max(1, _area(broad.box)) <= 0.18
        ):
            return True
    return False


def _substantial_physical_contact(
    first: BoxComponent,
    second: BoxComponent,
    *,
    tolerance: int,
    line_height: int,
    text_mask: np.ndarray,
) -> bool:
    intersection = _intersection_area(first.box, second.box)
    if intersection > 0:
        if _deep_crossing_is_distinct(first, second, line_height):
            return False
        if _intersection_over_smaller(first.box, second.box) >= 0.02:
            return True
        # A one-pixel overlap after deskewing is an edge contact, not material
        # overlap. Let the aligned-edge tests below decide it.

    ax1, ay1, ax2, ay2 = first.box
    bx1, by1, bx2, by2 = second.box
    horizontal_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    vertical_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
    horizontal_gap = max(0, max(ax1, bx1) - min(ax2, bx2))

    if vertical_gap <= tolerance:
        minimum_width = max(1, min(ax2 - ax1, bx2 - bx1))
        shared_fraction = horizontal_overlap / minimum_width
        side_alignment = max(abs(ax1 - bx1), abs(ax2 - bx2))
        width_ratio = minimum_width / max(1, max(ax2 - ax1, bx2 - bx1))
        aligned_split = (
            side_alignment <= line_height * 0.50 and width_ratio >= 0.78
        )
        anchored_multiline = (
            min(abs(ax1 - bx1), abs(ax2 - bx2)) <= line_height * 0.50
            and max(ay2 - ay1, by2 - by1) >= line_height * 2.0
        )
        if (
            horizontal_overlap >= line_height * 1.5
            and shared_fraction >= 0.80
            and (vertical_gap == 0 or aligned_split or anchored_multiline)
        ):
            top = min(ay2, by2)
            bottom = max(ay1, by1)
            gap = (max(ax1, bx1), top, min(ax2, bx2), bottom)
            return vertical_gap == 0 or _ink_fraction(text_mask, gap) <= 0.03

    if horizontal_gap <= tolerance:
        minimum_height = max(1, min(ay2 - ay1, by2 - by1))
        shared_fraction = vertical_overlap / minimum_height
        side_alignment = max(abs(ay1 - by1), abs(ay2 - by2))
        height_ratio = minimum_height / max(1, max(ay2 - ay1, by2 - by1))
        if (
            vertical_overlap >= line_height * 0.55
            and shared_fraction >= 0.72
            and (
                horizontal_gap == 0
                or (
                    side_alignment <= line_height * 0.40
                    and height_ratio >= 0.78
                )
            )
        ):
            left = min(ax2, bx2)
            right = max(ax1, bx1)
            gap = (left, max(ay1, by1), right, min(ay2, by2))
            return horizontal_gap == 0 or _ink_fraction(text_mask, gap) <= 0.03
    return False


def _group_components_reading_order(
    components: list[BoxComponent],
    gray: np.ndarray,
    line_mask: np.ndarray,
) -> list[RedactionRegion]:
    if not components:
        return []
    height, width = gray.shape
    text_mask, content_left, content_right, line_pitch = _text_mask_and_margins(
        gray, line_mask, components
    )
    line_height = _text_line_height(gray, line_mask)
    tolerance = max(2, round(min(height, width) * 0.004))
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

    contour_groups: dict[str, list[int]] = {}
    for index, component in enumerate(components):
        if component.source.startswith("rectilinear_contour_partition:"):
            contour_groups.setdefault(component.source, []).append(index)
    for indices in contour_groups.values():
        for index in indices[1:]:
            union(indices[0], index)

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

    wrap_candidates: list[tuple[float, int, int]] = []
    margin_slack = max(tolerance * 3, round(width * 0.11))
    content_width = max(1, content_right - content_left)
    for left, first in enumerate(components):
        fx1, fy1, fx2, fy2 = first.box
        for right, second in enumerate(components):
            if left == right or find(left) == find(right):
                continue
            sx1, sy1, sx2, sy2 = second.box
            if max(fy2 - fy1, sy2 - sy1) > line_pitch * 6.0:
                # Reading-flow continuation is defined between line-scale
                # pieces. Page-height masks must share real geometry instead
                # of being attached to a nearby line by their top coordinate.
                continue
            delta = sy1 - fy1
            if delta < max(4, line_pitch * 0.30) or delta > line_pitch * 1.80:
                continue
            after = (fx2, fy1 + 1, content_right + 1, max(fy1 + 2, fy2 - 1))
            before = (content_left, sy1 + 1, sx1, max(sy1 + 2, sy2 - 1))
            if _ink_fraction(text_mask, after) > 0.030:
                continue
            if _ink_fraction(text_mask, before) > 0.030:
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
            if not exits_line or not enters_line or fx1 <= sx1 + tolerance * 2:
                continue
            if any(
                index != left
                and find(index) != find(left)
                and other.box[0] > fx1 + tolerance
                and geometry_stage._same_reading_row(first, other, line_pitch)
                for index, other in enumerate(components)
            ):
                continue
            if any(
                index != right
                and find(index) != find(right)
                and other.box[0] < sx1 - tolerance
                and geometry_stage._same_reading_row(second, other, line_pitch)
                for index, other in enumerate(components)
            ):
                continue
            cost = (
                abs(delta - line_pitch)
                + abs(content_right - fx2) * 0.02
                + abs(sx1 - content_left) * 0.02
            )
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
            components=tuple(
                sorted(values, key=lambda item: (item.box[1], item.box[0]))
            )
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
    h_segments, v_segments, horizontal, vertical = geometry_stage._extract_axis_segments(dark, faint)
    line_mask = cv2.bitwise_or(horizontal, vertical)
    edges = cv2.Canny(enhanced, 30, 110)
    line_height = _text_line_height(work_gray, line_mask)
    zones = geometry_stage._line_zones(horizontal, vertical)
    proposals: list[RectangleProposal] = []
    for zone_id, zone in enumerate(zones, start=1):
        proposals.extend(
            geometry_stage._enumerate_zone_rectangles(
                work_gray,
                edges,
                h_segments,
                v_segments,
                zone,
                zone_id,
                line_height,
            )
        )
    strict_outline = geometry_stage._compact_maximal_cover(proposals, work_gray.shape)
    closed_candidates, closed_mask, closed_diagnostics = (
        geometry_stage._closed_blank_contour_candidates(work_gray, dark, edges, line_height)
    )
    closed_rescues = geometry_stage._novel_rescues(
        closed_candidates, strict_outline, work_gray.shape
    )
    fallback_candidates, enclosed_mask, enclosed_diagnostics = (
        geometry_stage._enclosed_blank_candidates(
            work_gray, dark, faint, edges, line_height
        )
    )
    contour_rescues = geometry_stage._novel_rescues(
        fallback_candidates,
        strict_outline + closed_rescues,
        work_gray.shape,
    )
    outline = strict_outline + closed_rescues + contour_rescues
    solid, solid_mask = _shape_preserving_solid_candidates(work_gray, line_height)
    blackout, blackout_mask = _shape_preserving_blackout_candidates(
        work_gray, line_height, outline + solid
    )
    base_components = geometry_stage._dedupe_components(
        outline + solid + blackout, work_gray.shape
    )
    lsd_candidates, lsd_mask, lsd_diagnostics = _lsd_geometry_rescues(
        work_gray, line_height, base_components
    )
    refined_base, remaining_lsd, refinement_count = geometry_stage._apply_lsd_refinements(
        base_components, lsd_candidates, work_gray.shape
    )
    promoted_base, remaining_lsd, promotion_count = (
        _promote_supported_outer_rectangles(
            refined_base,
            remaining_lsd,
            work_gray.shape,
            line_height,
        )
    )
    lsd_rescues = _layer_occlusion_rescues(
        remaining_lsd,
        promoted_base,
        work_gray.shape,
        line_height,
    )
    combined = geometry_stage._dedupe_components(
        promoted_base + lsd_rescues, work_gray.shape
    )
    combined = _suppress_lsd_border_echoes(
        combined, work_gray.shape, line_height
    )
    canonical = geometry_stage._canonicalize_components(
        combined, work_gray.shape, line_height
    )
    completed, step_completion_mask, step_completion_count = (
        _shared_edge_step_completions(work_gray, line_height, canonical)
    )
    components, suppression = _suppress_text_frames_and_glyphs(
        work_gray, line_mask, completed, line_height
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
        "shared_edge_step_completions": step_completion_mask,
        "edges": edges,
    }
    diagnostics: dict[str, Any] = {
        "coordinate_system": "deskewed",
        "estimated_rotation_degrees": rotation,
        "inverse_affine_to_original": [
            [float(value) for value in row] for row in inverse
        ],
        "horizontal_segment_count": len(h_segments),
        "vertical_segment_count": len(v_segments),
        "line_zone_count": len(zones),
        "rectangle_proposal_count": len(proposals),
        "strict_outline_component_count": len(strict_outline),
        "outline_component_count": len(outline),
        "closed_blank_rescue_count": len(closed_rescues),
        "rectified_warp_candidate_count": len(fallback_candidates),
        "rectified_warp_rescue_count": len(contour_rescues),
        "rectified_warp_fallback_count": len(contour_rescues),
        "solid_candidate_count": len(solid),
        "rectilinear_solid_step_rescue_count": sum(
            component.source.startswith("solid_fill_rectilinear_step_rescue")
            for component in solid
        ),
        "dense_blackout_candidate_count": len(blackout),
        "lsd_corner_refinement_count": refinement_count,
        "lsd_outer_border_promotion_count": promotion_count,
        "lsd_geometry_rescue_count": len(lsd_rescues),
        "lsd_layer_occlusion_rescue_count": sum(
            component.source.startswith("lsd_layer_occlusion_rescue")
            for component in lsd_rescues
        ),
        "shared_edge_step_completion_count": step_completion_count,
        "framed_text_suppression_count": suppression["framed_text"],
        "marginal_text_crossing_suppression_count": suppression[
            "marginal_text_crossing"
        ],
        "small_glyph_suppression_count": suppression["small_glyph_contour"],
        "component_count_before_grouping": len(components),
        "region_count": len(regions),
        "estimated_text_line_height": line_height,
        "detector_policy": (
            "Single-page geometry only: direction-invariant border recovery, "
            "shared-edge step completion, shape-preserving rectilinear solid "
            "masks, text-bridge suppression, and reading-order semantic "
            "grouping."
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


def process_record(
    record: InputRecord,
    *,
    out_root: Path,
    save_debug_masks: bool,
    detector_fn: Any | None = None,
    detector_version: str = "layered-detection",
) -> dict[str, Any]:
    original_gray = cv2.imread(str(record.rendered_image_path), cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise RuntimeError(f"Could not read image: {record.rendered_image_path}")
    detector = detector_fn or detect_redaction_regions_with_artifacts
    regions, artifacts, diagnostics = detector(original_gray)
    display_gray = artifacts["deskewed_source"]
    inverse = np.asarray(
        diagnostics["inverse_affine_to_original"], dtype=np.float32
    )
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
    # Render the measured physical components, not only the exterior union of
    # a semantic region. An exterior-only contour can bridge an intentional
    # text notch and falsely imply that visible prose was redacted.
    region_outlines = [
        [component.polygon for component in region.components]
        for region in regions
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

    original_height, original_width = original_gray.shape
    payload = {
        "detector_version": detector_version,
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
        "label_semantics": {
            "region": (
                "R<n> identifies one inferred redaction unit in page reading order. "
                "It is not a security classification."
            ),
            "component": (
                "R<n>.<m> identifies the m-th physical box or measured shape "
                "belonging to region R<n>."
            ),
            "single_component_overlay": (
                "A one-component region is abbreviated R<n> on the PNG overlay; "
                "its JSON component identifier remains R<n>.1."
            ),
        },
        "redaction_boxes_xyxy": [list(map(int, region.box)) for region in regions],
        "redaction_regions": [
            {
                "region_id": f"R{region_index}",
                "region_order": region_index,
                "component_count": len(region.components),
                "bounds_xyxy": list(map(int, region.box)),
                "outline_polygons_xy": [
                    [list(map(int, point)) for point in polygon]
                    for polygon in region_outlines[region_index - 1]
                ],
                "original_image_outline_polygons_xy": [
                    _map_polygon_to_original(
                        polygon, inverse, original_width, original_height
                    )
                    for polygon in region_outlines[region_index - 1]
                ],
                "components": [
                    {
                        "component_id": f"R{region_index}.{component_index}",
                        "component_order": component_index,
                        "bounds_xyxy": list(map(int, component.box)),
                        "polygon_xy": [
                            list(map(int, point)) for point in component.polygon
                        ],
                        "original_image_polygon_xy": _map_polygon_to_original(
                            component.polygon,
                            inverse,
                            original_width,
                            original_height,
                        ),
                        "proposal_source": component.source,
                        "proposal_score": round(float(component.score), 6),
                    }
                    for component_index, component in enumerate(
                        region.components, start=1
                    )
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
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    record_processor: Any | None = None,
    detector_version: str = "layered-detection",
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    label_guide = out_root / "LABEL_SEMANTICS.md"
    label_guide.write_text(
        "# Detection Labels\n\n"
        "- `R1`, `R2`, ... identify inferred redaction units in page reading "
        "order. They are not secrecy classifications or confidence grades.\n"
        "- `R1.1`, `R1.2`, ... identify separate physical boxes or measured "
        "shapes grouped into `R1`.\n"
        "- A one-component unit is abbreviated `R1` on the PNG overlay; its "
        "JSON component ID remains `R1.1`.\n"
        "- The JSON retains every component polygon, proposal source, and "
        "grouped-region boundary for audit.\n",
        encoding="utf-8",
    )
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
    processor = record_processor or process_record
    for record in core_stage._progress(records, total=len(records), desc="Detect redaction boxes"):
        try:
            rows.append(
                processor(
                    record,
                    out_root=out_root,
                    save_debug_masks=save_debug_masks,
                    detector_version=detector_version,
                )
            )
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
    manifest_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
        "label_semantics_markdown": str(label_guide),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
