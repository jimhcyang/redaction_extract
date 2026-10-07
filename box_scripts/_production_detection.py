"""High-recall, answer-blind redaction geometry built on validated candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import _geometry_detection as geometry_stage
from . import _layered_detection as layered_stage
from . import _validated_detection as validated_stage


PDFPair = validated_stage.PDFPair
PairCollectionStats = validated_stage.PairCollectionStats
InputRecord = validated_stage.InputRecord
BoxComponent = validated_stage.BoxComponent
RedactionRegion = validated_stage.RedactionRegion
collect_pdf_pairs_with_stats = validated_stage.collect_pdf_pairs_with_stats
render_pdf_to_images = validated_stage.render_pdf_to_images
iter_input_records = validated_stage.iter_input_records


def _existing_union_coverage(
    candidate: BoxComponent,
    existing: list[BoxComponent],
) -> float:
    """Return the exact fraction of a candidate already covered by components."""

    if not existing:
        return 0.0
    x1, y1, x2, y2 = candidate.box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    candidate_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    union_mask = np.zeros_like(candidate_mask)
    candidate_polygon = np.asarray(
        [(x - x1, y - y1) for x, y in candidate.polygon], dtype=np.int32
    )
    cv2.fillPoly(candidate_mask, [candidate_polygon], 1)
    for component in existing:
        cx1, cy1, cx2, cy2 = component.box
        if cx2 <= x1 or cx1 >= x2 or cy2 <= y1 or cy1 >= y2:
            continue
        polygon = np.asarray(
            [(x - x1, y - y1) for x, y in component.polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(union_mask, [polygon], 1)
    pixels = int(np.count_nonzero(candidate_mask))
    if not pixels:
        return 0.0
    return float(np.count_nonzero(candidate_mask & union_mask) / pixels)


def _dedupe_page_frame_candidates(
    candidates: list[BoxComponent],
) -> list[BoxComponent]:
    """Keep one measured outline for nested proposals of the same page frame."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            -((item.box[2] - item.box[0]) * (item.box[3] - item.box[1])),
            -item.score,
        ),
    )
    accepted: list[BoxComponent] = []
    for candidate in ordered:
        x1, y1, x2, y2 = candidate.box
        candidate_area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for prior in accepted:
            px1, py1, px2, py2 = prior.box
            prior_area = max(1, (px2 - px1) * (py2 - py1))
            overlap = max(0, min(x2, px2) - max(x1, px1)) * max(
                0, min(y2, py2) - max(y1, py1)
            )
            if overlap / min(candidate_area, prior_area) >= 0.80:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.box[1], item.box[0]))


def _promote_supported_lsd_outlines(
    gray: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
    candidates: list[BoxComponent],
) -> tuple[list[BoxComponent], list[BoxComponent], np.ndarray]:
    """Promote measured outlines that the validated adjacency gate left dormant.

    Two routes are intentionally separate. Blank-body candidates need strong
    support on all four sides and at least two measured corners. Page-furniture
    candidates may contain release-stamp text, but must be shallow, broad,
    physically complete outlines confined to the top or bottom page bands.
    """

    height, width = gray.shape
    accepted: list[BoxComponent] = []
    page_frames: list[BoxComponent] = []
    debug = np.zeros_like(gray)
    for candidate in candidates:
        if not validated_stage._is_novel(candidate, existing + accepted, gray.shape):
            continue
        x1, y1, x2, y2 = candidate.box
        candidate_area = max(1, (x2 - x1) * (y2 - y1))
        covered_fraction = 0.0
        for component in existing + accepted:
            cx1, cy1, cx2, cy2 = component.box
            overlap = max(0, min(x2, cx2) - max(x1, cx1)) * max(
                0, min(y2, cy2) - max(y1, cy1)
            )
            covered_fraction = max(covered_fraction, overlap / candidate_area)
        # Intersections inside two overlapping physical boxes can form a
        # pristine-looking four-sided cell. It is not another redaction box.
        if covered_fraction >= 0.82:
            continue
        if _existing_union_coverage(candidate, existing + accepted) >= 0.35:
            continue
        supports = layered_stage._lsd_support_values(candidate)
        if len(supports) != 4:
            continue
        corners = geometry_stage._lsd_corner_count(candidate)
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue
        ink, active_rows, longest = validated_stage._visible_ink_stats(
            gray, candidate.box, line_height
        )
        support_sum = sum(supports)
        horizontal_min = min(supports[:2])
        vertical_min = min(supports[2:])

        blank_body = (
            candidate.score >= 5.78
            and corners >= 2
            and support_sum >= 3.02
            and horizontal_min >= 0.40
            and vertical_min >= 0.30
            # The mature validation routes already handle genuinely thin boxes. A
            # dormant outline no taller than one text line is usually the
            # narrow gap formed by borders of adjacent redactions.
            and box_height >= line_height * 1.05
            and ink <= 0.060
            and active_rows <= 0.20
            and longest <= max(line_height * 1.35, box_height * 0.32)
        )
        page_band = y1 <= height * 0.18 or y2 >= height * 0.90
        page_furniture = (
            page_band
            and candidate.score >= 6.20
            and corners >= 3
            and box_width >= max(width * 0.09, line_height * 4.0)
            and line_height * 0.75 <= box_height <= line_height * 5.0
            and box_width / max(1, box_height) >= 2.8
            and horizontal_min >= 0.85
            and max(supports[2:]) >= 0.50
            and support_sum >= 3.05
        )
        large_edge_blank = (
            (x1 <= width * 0.02 or x2 >= width * 0.98)
            and candidate.score >= 5.70
            and corners >= 2
            and box_width >= width * 0.45
            and box_height >= line_height * 3.0
            and box_width * box_height >= width * height * 0.035
            and support_sum >= 3.00
            and max(supports[:2]) >= 0.90
            and min(supports[2:]) >= 0.75
            and ink <= 0.040
            and active_rows <= 0.10
        )
        # Page-band release frames remain audit geometry even when their
        # blank interior also satisfies the generic body-outline rule.
        if page_furniture:
            page_frames.append(
                BoxComponent(
                    box=candidate.box,
                    polygon=candidate.polygon,
                    source=f"page_frame_candidate:{candidate.source}",
                    score=float(candidate.score),
                )
            )
            continue
        if not (blank_body or large_edge_blank):
            continue
        route = "blank" if blank_body else "large_edge_blank"
        promoted = BoxComponent(
            box=candidate.box,
            polygon=candidate.polygon,
            source=(
                f"supported_outline:{route}:corners={corners}:"
                f"support={support_sum:.2f}:ink={ink:.3f}:rows={active_rows:.3f}:"
                f"{candidate.source}"
            ),
            score=float(candidate.score) + 0.04,
        )
        accepted.append(promoted)
        cv2.polylines(
            debug,
            [np.asarray(promoted.polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )
    return accepted, page_frames, debug


def _complex_enclosed_outline_candidates(
    gray: np.ndarray,
    dark: np.ndarray,
    faint: np.ndarray,
    edges: np.ndarray,
    line_height: int,
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover large concave blank shapes bounded by measured line ink.

    The frozen detector intentionally rejects enclosures occupying less than
    half of their bounding rectangle. That protects against page-layout
    whitespace, but also removes genuine L- and step-shaped redactions. This
    branch keeps the exact free-space contour and requires a large enclosure,
    low interior ink, predominantly rectilinear edges, and measured perimeter
    support. It never fills the missing notch with a bounding rectangle.
    """

    height, width = gray.shape
    axis_open = geometry_stage._axis_open
    observed = cv2.bitwise_or(
        cv2.bitwise_or(axis_open(dark, "h"), axis_open(dark, "v")),
        cv2.bitwise_or(axis_open(faint, "h"), axis_open(faint, "v")),
    )
    # Some historical masks close with one slanted ruled edge. Preserve only
    # long straight non-axis segments here; short glyph strokes never become
    # barriers, and candidates must still pass the enclosure quality gates.
    line_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = line_detector.detect(cv2.GaussianBlur(gray, (3, 3), 0))[0]
    # One side of a genuine stepped or five-sided mask can be much shorter
    # than the other borders. Retain it as a barrier, then rely on the closed-
    # enclosure, blank-interior, and perimeter-support gates below. This is
    # safer than accepting the short segment as a box by itself.
    minimum_oblique_length = max(16.0, line_height * 0.45)
    if detected is not None:
        for raw in detected:
            x1, y1, x2, y2 = map(float, raw[0])
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < minimum_oblique_length:
                continue
            angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
            axis_distance = min(angle, abs(angle - 90.0), abs(angle - 180.0))
            if axis_distance <= 7.0:
                continue
            cv2.line(
                observed,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                255,
                2,
                cv2.LINE_AA,
            )
    observed = cv2.morphologyEx(
        observed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    radius = max(1, round(min(height, width) * 0.0018))
    barrier = cv2.dilate(
        observed,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1)
        ),
        iterations=1,
    )
    frame = max(2, radius + 1)
    barrier[:frame, :] = 255
    barrier[-frame:, :] = 255
    barrier[:, :frame] = 255
    barrier[:, -frame:] = 255
    free = np.where(barrier == 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        free, connectivity=8
    )
    candidates: list[BoxComponent] = []
    debug = np.zeros_like(gray)
    rejected_shape = 0
    rejected_text = 0
    minimum_area = max(
        round(width * height * 0.0025),
        round(line_height * line_height * 3.0),
    )
    support_image = cv2.dilate(observed, np.ones((7, 7), np.uint8))

    for label in range(1, count):
        x, y, box_width, box_height, free_area = map(int, stats[label])
        box_area = box_width * box_height
        if (
            box_width < max(70, round(line_height * 3.5))
            or box_height < max(35, round(line_height * 1.7))
            or box_area < minimum_area
            or box_area > width * height * 0.72
        ):
            continue
        free_fill = free_area / max(1, box_area)
        if not 0.16 <= free_fill < 0.92:
            continue

        local = np.where(
            labels[y : y + box_height, x : x + box_width] == label,
            255,
            0,
        ).astype(np.uint8)
        local = cv2.dilate(
            local,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            ),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour = contour + np.asarray([[[x, y]]], dtype=np.int32)
        polygon = geometry_stage._simplify_polygon(
            contour, gray.shape, line_height
        )
        if len(polygon) < 5:
            rejected_shape += 1
            continue
        polygon_box = geometry_stage._polygon_box(polygon)
        px1, py1, px2, py2 = polygon_box
        polygon_area = abs(
            float(
                cv2.contourArea(
                    np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
                )
            )
        )
        polygon_fill = polygon_area / max(
            1.0, float((px2 - px1) * (py2 - py1))
        )
        if not 0.16 <= polygon_fill < 0.92:
            rejected_shape += 1
            continue

        # Physical redaction borders are horizontal/vertical after deskewing.
        # Permit one locally slanted scan edge, but not arbitrary text shapes.
        axis_length = 0.0
        total_length = 0.0
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            dx = abs(second[0] - first[0])
            dy = abs(second[1] - first[1])
            length = float(np.hypot(dx, dy))
            total_length += length
            if min(dx, dy) <= max(4, 0.10 * max(dx, dy)):
                axis_length += length
        rectilinear_fraction = axis_length / max(1.0, total_length)
        if rectilinear_fraction < 0.45:
            rejected_shape += 1
            continue

        polygon_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(
            polygon_mask, [np.asarray(polygon, dtype=np.int32)], 255
        )
        erosion = max(3, 2 * radius + 1)
        interior = cv2.erode(
            polygon_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (erosion, erosion)),
            iterations=1,
        )
        values = gray[interior > 0]
        edge_values = edges[interior > 0]
        if not values.size:
            continue
        mean = float(np.mean(values))
        dark_fraction = float(np.mean(values < 175))
        edge_fraction = float(np.mean(edge_values > 0))
        if mean < 205 or dark_fraction > 0.055 or edge_fraction > 0.060:
            rejected_text += 1
            continue

        perimeter = cv2.subtract(
            cv2.dilate(polygon_mask, np.ones((3, 3), np.uint8)),
            cv2.erode(polygon_mask, np.ones((3, 3), np.uint8)),
        )
        perimeter_pixels = int(np.count_nonzero(perimeter))
        support = int(
            np.count_nonzero((perimeter > 0) & (support_image > 0))
        ) / max(1, perimeter_pixels)
        if support < 0.48:
            rejected_shape += 1
            continue

        candidate = BoxComponent(
            box=polygon_box,
            polygon=polygon,
            source=(
                "complex_enclosed_outline:"
                f"support={support:.2f}:fill={polygon_fill:.2f}:"
                f"rectilinear={rectilinear_fraction:.2f}"
            ),
            score=float(8.0 + support + rectilinear_fraction),
        )
        candidates.append(candidate)
        cv2.polylines(
            debug,
            [np.asarray(polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )

    candidates = geometry_stage._dedupe_components(candidates, gray.shape)
    return candidates, debug, {
        "complex_enclosed_candidate_count": len(candidates),
        "complex_enclosed_rejected_shape_count": rejected_shape,
        "complex_enclosed_rejected_text_count": rejected_text,
    }


def _cycle_basis(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Return a deterministic fundamental-cycle basis for an undirected graph."""

    cycles: list[list[int]] = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        stack = [root]
        predecessor = {root: root}
        used: dict[int, set[int]] = {root: set()}
        while stack:
            node = stack.pop()
            node_used = used[node]
            for neighbor in sorted(adjacency[node]):
                if neighbor not in used:
                    predecessor[neighbor] = node
                    stack.append(neighbor)
                    used[neighbor] = {node}
                elif neighbor not in node_used:
                    neighbor_used = used[neighbor]
                    cycle = [neighbor, node]
                    parent = predecessor[node]
                    while parent not in neighbor_used:
                        cycle.append(parent)
                        parent = predecessor[parent]
                    cycle.append(parent)
                    cycles.append(cycle)
                    used[neighbor].add(node)
            remaining.discard(node)
    return cycles


def _line_segment_cycle_candidates(
    gray: np.ndarray,
    edges: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover irregular masks whose measured border segments form a cycle.

    This route covers thin stepped masks and faint polygonal masks for which
    barrier dilation consumes the interior. It snaps nearby measured segment
    endpoints, finds graph cycles, and validates the exact polygon. Labels,
    filenames, answer text, and paired-page information are never consulted.
    """

    height, width = gray.shape
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(
        cv2.GaussianBlur(gray, (3, 3), 0)
    )[0]
    debug = np.zeros_like(gray)
    diagnostics = {
        "line_cycle_count": 0,
        "line_cycle_candidate_count": 0,
        "line_cycle_rejected_text_count": 0,
        "line_cycle_rejected_shape_count": 0,
    }
    if detected is None:
        return [], debug, diagnostics

    minimum_length = max(15.0, line_height * 0.40)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for raw in detected[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in raw)
        if math.hypot(x2 - x1, y2 - y1) >= minimum_length:
            segments.append(((x1, y1), (x2, y2)))
    if not segments:
        return [], debug, diagnostics

    points = [point for segment in segments for point in segment]
    parents = list(range(len(points)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    snap_tolerance = max(8.0, min(15.0, line_height * 0.38))
    cell_size = snap_tolerance
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(points):
        cell = (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for prior in buckets.get(
                    (cell[0] + offset_x, cell[1] + offset_y), []
                ):
                    prior_x, prior_y = points[prior]
                    if math.hypot(x - prior_x, y - prior_y) <= snap_tolerance:
                        union(index, prior)
        buckets.setdefault(cell, []).append(index)

    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    point_to_node: dict[int, int] = {}
    node_points: list[tuple[float, float]] = []
    for node_index, members in enumerate(groups.values()):
        coordinates = np.asarray([points[index] for index in members])
        median = np.median(coordinates, axis=0)
        node_points.append((float(median[0]), float(median[1])))
        for index in members:
            point_to_node[index] = node_index

    adjacency: dict[int, set[int]] = {}
    for segment_index in range(len(segments)):
        left = point_to_node[2 * segment_index]
        right = point_to_node[2 * segment_index + 1]
        if left == right:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    cycles = _cycle_basis(adjacency)
    diagnostics["line_cycle_count"] = len(cycles)
    candidates: list[BoxComponent] = []
    # The old implementation rebuilt a page-sized mask for every component
    # inside every candidate cycle. Keep only clipped boxes up front, then
    # lazily cache crop-local masks when a surviving cycle actually overlaps
    # that component. This preserves the exact pixel-overlap test without
    # taxing ordinary pages whose cycles fail earlier validation.
    existing_geometry: list[
        tuple[int, BoxComponent, tuple[int, int, int, int]]
    ] = []
    for component_index, component in enumerate(existing):
        cx1, cy1, cx2, cy2 = component.box
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(width, cx2), min(height, cy2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        existing_geometry.append(
            (component_index, component, (cx1, cy1, cx2, cy2))
        )
    existing_mask_cache: dict[int, tuple[np.ndarray, int]] = {}
    minimum_area = max(
        width * height * 0.0012,
        line_height * line_height * 3.0,
    )
    for cycle in cycles:
        # Four-sided rectangles already have a stricter, mature route. This
        # branch is intentionally limited to measured irregular outlines.
        if not 5 <= len(cycle) <= 12:
            continue
        polygon = tuple(
            (int(round(node_points[index][0])), int(round(node_points[index][1])))
            for index in cycle
        )
        contour = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        polygon_area = abs(float(cv2.contourArea(contour)))
        hull_area = abs(float(cv2.contourArea(cv2.convexHull(contour))))
        hull_fill = polygon_area / max(1.0, hull_area)
        box = geometry_stage._polygon_box(polygon)
        box_width, box_height = box[2] - box[0], box[3] - box[1]
        box_area = max(1, box_width * box_height)
        polygon_fill = polygon_area / box_area
        if (
            polygon_area < minimum_area
            or polygon_area > width * height * 0.45
            or box_width < line_height * 2.2
            or box_height < line_height * 0.90
            or not 0.12 <= polygon_fill <= 0.98
            or hull_fill < 0.45
        ):
            diagnostics["line_cycle_rejected_shape_count"] += 1
            continue

        polygon_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [contour], 255)
        erosion = max(3, int(round(line_height * 0.14)))
        interior = cv2.erode(
            polygon_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion)),
            iterations=1,
        )
        values = gray[interior > 0]
        edge_values = edges[interior > 0]
        if not values.size:
            diagnostics["line_cycle_rejected_shape_count"] += 1
            continue
        mean = float(np.mean(values))
        dark_fraction = float(np.mean(values < 175))
        edge_fraction = float(np.mean(edge_values > 0))
        if mean < 210 or dark_fraction > 0.055 or edge_fraction > 0.065:
            diagnostics["line_cycle_rejected_text_count"] += 1
            continue

        strong_existing_explanations = 0
        duplicate_existing_explanation = False
        candidate_pixels = max(1, int(np.count_nonzero(polygon_mask)))
        for component_index, component, component_box in existing_geometry:
            cx1, cy1, cx2, cy2 = component_box
            overlap_x1 = max(box[0], cx1)
            overlap_y1 = max(box[1], cy1)
            overlap_x2 = min(box[2], cx2)
            overlap_y2 = min(box[3], cy2)
            if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
                continue
            cached_mask = existing_mask_cache.get(component_index)
            if cached_mask is None:
                component_mask = np.zeros(
                    (cy2 - cy1, cx2 - cx1), dtype=np.uint8
                )
                local_polygon = np.asarray(
                    [(x - cx1, y - cy1) for x, y in component.polygon],
                    dtype=np.int32,
                )
                cv2.fillPoly(component_mask, [local_polygon], 255)
                component_pixels = int(np.count_nonzero(component_mask))
                if not component_pixels:
                    continue
                existing_mask_cache[component_index] = (
                    component_mask,
                    component_pixels,
                )
            else:
                component_mask, component_pixels = cached_mask
            candidate_crop = polygon_mask[
                overlap_y1:overlap_y2, overlap_x1:overlap_x2
            ]
            component_crop = component_mask[
                overlap_y1 - cy1 : overlap_y2 - cy1,
                overlap_x1 - cx1 : overlap_x2 - cx1,
            ]
            intersection = int(
                np.count_nonzero(
                    (candidate_crop > 0) & (component_crop > 0)
                )
            )
            if (
                intersection / candidate_pixels >= 0.02
                and intersection / component_pixels >= 0.60
            ):
                strong_existing_explanations += 1
            # Endpoint snapping can close a tiny sliver next to an otherwise
            # complete measured box. Do not emit that nearly identical cycle
            # as a second physical component.
            if (
                intersection / candidate_pixels >= 0.85
                and intersection / component_pixels >= 0.80
            ):
                duplicate_existing_explanation = True
        # Intersections among several overlapping rectangles also form closed
        # cycles. Those cycles are unions of already measured physical boxes,
        # not additional redactions. A single contained partial outline is
        # allowed because the cycle can be its measured outer completion.
        if strong_existing_explanations >= 2 or duplicate_existing_explanation:
            diagnostics["line_cycle_rejected_shape_count"] += 1
            continue

        candidate = BoxComponent(
            box=box,
            polygon=polygon,
            source=(
                "line_segment_cycle:"
                f"vertices={len(polygon)}:fill={polygon_fill:.2f}:"
                f"ink={dark_fraction:.3f}"
            ),
            score=float(9.0 + polygon_fill + max(0.0, (mean - 210.0) / 45.0)),
        )
        if _existing_union_coverage(candidate, existing) >= 0.35:
            diagnostics["line_cycle_rejected_shape_count"] += 1
            continue
        if not validated_stage._is_novel(candidate, existing + candidates, gray.shape):
            continue
        candidate_area = max(1, box_area)
        if any(
            max(0, min(box[2], component.box[2]) - max(box[0], component.box[0]))
            * max(0, min(box[3], component.box[3]) - max(box[1], component.box[1]))
            / candidate_area
            >= 0.75
            for component in existing + candidates
        ):
            continue
        candidates.append(candidate)
        cv2.polylines(debug, [contour], True, 255, 1, cv2.LINE_8)

    candidates = geometry_stage._dedupe_components(candidates, gray.shape)
    diagnostics["line_cycle_candidate_count"] = len(candidates)
    return candidates, debug, diagnostics


def _cycle_absorbs_partial_component(
    cycle: BoxComponent,
    component: BoxComponent,
    image_shape: tuple[int, int],
) -> bool:
    """Return whether a measured cycle is a strict completion of one partial."""

    cycle_mask = np.zeros(image_shape, dtype=np.uint8)
    component_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(
        cycle_mask, [np.asarray(cycle.polygon, dtype=np.int32)], 255
    )
    cv2.fillPoly(
        component_mask, [np.asarray(component.polygon, dtype=np.int32)], 255
    )
    cycle_pixels = max(1, int(np.count_nonzero(cycle_mask)))
    component_pixels = max(1, int(np.count_nonzero(component_mask)))
    intersection = int(
        np.count_nonzero((cycle_mask > 0) & (component_mask > 0))
    )
    return (
        intersection / component_pixels >= 0.90
        and intersection / cycle_pixels >= 0.30
        and cycle_pixels >= component_pixels * 1.10
    )


def _three_side_page_outline_candidates(
    gray: np.ndarray,
    edges: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray]:
    """Recover a header/footer box whose top or bottom is occluded by text.

    Release stamps sometimes overwrite one horizontal border while leaving
    two parallel sides and the opposite border measurable. Restricting this
    virtual-side rule to the page bands prevents ordinary body text frames
    from becoming candidates.
    """

    height, width = gray.shape
    raw_h, raw_v = layered_stage._oriented_line_segments(gray)
    page_horizontal = [
        segment
        for segment in raw_h
        if (
            max(segment.start[1], segment.end[1]) <= height * 0.22
            or min(segment.start[1], segment.end[1]) >= height * 0.82
        )
    ]
    page_vertical = [
        segment
        for segment in raw_v
        if (
            max(segment.start[1], segment.end[1]) <= height * 0.22
            or min(segment.start[1], segment.end[1]) >= height * 0.82
        )
    ]
    horizontal = page_horizontal + geometry_stage._merge_oriented_segments(
        page_horizontal, "h", line_height, gray.shape
    )
    vertical = page_vertical + geometry_stage._merge_oriented_segments(
        page_vertical, "v", line_height, gray.shape
    )
    page_verticals = [
        segment
        for segment in vertical
        if segment.length >= max(22.0, line_height * 0.70)
        and (
            max(segment.start[1], segment.end[1]) <= height * 0.20
            or min(segment.start[1], segment.end[1]) >= height * 0.84
        )
    ]
    support_tolerance = max(5.0, line_height * 0.30)
    candidates: list[BoxComponent] = []
    debug = np.zeros_like(gray)

    for left_index, first in enumerate(page_verticals):
        for second in page_verticals[left_index + 1 :]:
            first_x, second_x = first.midpoint[0], second.midpoint[0]
            if first_x > second_x:
                left, right = second, first
                left_x, right_x = second_x, first_x
            else:
                left, right = first, second
                left_x, right_x = first_x, second_x
            box_width = right_x - left_x
            if not line_height * 2.8 <= box_width <= width * 0.55:
                continue
            left_top, left_bottom = sorted((left.start[1], left.end[1]))
            right_top, right_bottom = sorted((right.start[1], right.end[1]))
            if (
                abs(left_top - right_top) > line_height * 0.45
                or abs(left_bottom - right_bottom) > line_height * 0.45
            ):
                continue
            top = float(np.median([left_top, right_top]))
            bottom = float(np.median([left_bottom, right_bottom]))
            box_height = bottom - top
            if not line_height * 0.75 <= box_height <= line_height * 3.2:
                continue
            top_support = layered_stage._axis_segment_support(
                horizontal,
                orientation="h",
                fixed_position=top,
                start=left_x,
                end=right_x,
                tolerance=support_tolerance,
            )
            bottom_support = layered_stage._axis_segment_support(
                horizontal,
                orientation="h",
                fixed_position=bottom,
                start=left_x,
                end=right_x,
                tolerance=support_tolerance,
            )
            observed_support = max(top_support, bottom_support)
            missing_support = min(top_support, bottom_support)
            if observed_support < 0.78 or missing_support > 0.42:
                continue
            box = (
                max(0, int(round(left_x))),
                max(0, int(round(top))),
                min(width, int(round(right_x)) + 1),
                min(height, int(round(bottom)) + 1),
            )
            quality = geometry_stage._robust_interior_quality(
                gray, edges, box, line_height
            )
            if quality is None:
                continue
            mean, dark_fraction, edge_fraction, _ = quality
            if mean < 205 or dark_fraction > 0.075 or edge_fraction > 0.075:
                continue
            candidate = BoxComponent(
                box=box,
                polygon=geometry_stage._rect_polygon(box),
                source=(
                    "three_side_page_outline:"
                    f"observed={observed_support:.2f}:missing={missing_support:.2f}:"
                    f"ink={dark_fraction:.3f}"
                ),
                score=float(7.5 + observed_support),
            )
            candidate_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            covered_fraction = 0.0
            for component in existing + candidates:
                cx1, cy1, cx2, cy2 = component.box
                overlap = max(0, min(box[2], cx2) - max(box[0], cx1)) * max(
                    0, min(box[3], cy2) - max(box[1], cy1)
                )
                covered_fraction = max(
                    covered_fraction, overlap / candidate_area
                )
            # A missing-side hypothesis is useful only for genuinely missing
            # geometry. Page stamps often leave a narrow three-sided cell
            # inside a complete redaction box; emitting both creates a false
            # extra component and can displace the better outline at dedupe.
            if covered_fraction >= 0.75:
                continue
            if not validated_stage._is_novel(
                candidate, existing + candidates, gray.shape
            ):
                continue
            candidates.append(candidate)
            cv2.polylines(
                debug,
                [np.asarray(candidate.polygon, dtype=np.int32)],
                True,
                255,
                1,
                cv2.LINE_8,
            )
    return geometry_stage._dedupe_components(candidates, gray.shape), debug


def detect_redaction_regions_with_artifacts(
    gray: np.ndarray,
) -> tuple[list[RedactionRegion], dict[str, np.ndarray], dict[str, Any]]:
    canonical, input_scale = validated_stage._canonical_detection_image(gray)
    base_components, artifacts, diagnostics, lsd_candidates = (
        validated_stage._base_detection_with_lsd_candidates(canonical)
    )
    work_gray = artifacts["deskewed_source"]
    line_mask = artifacts["lines_mask"]
    line_height = layered_stage._text_line_height(work_gray, line_mask)

    complex_candidates, complex_mask, complex_diagnostics = (
        _complex_enclosed_outline_candidates(
            work_gray,
            artifacts["dark_mask"],
            artifacts["faint_mask"],
            artifacts["edges"],
            line_height,
        )
    )
    marginal = validated_stage._marginal_open_outline_candidates(
        work_gray,
        line_height,
        base_components + complex_candidates,
    )
    blackout, blackout_mask = validated_stage._blackout_partition_candidates(
        work_gray,
        line_height,
        base_components + complex_candidates + marginal,
    )
    layered, layered_mask = validated_stage._layered_outline_candidates(
        work_gray,
        line_height,
        base_components
        + complex_candidates
        + marginal
        + blackout,
        lsd_candidates,
    )
    existing = list(base_components)
    for component in complex_candidates + marginal + blackout:
        if validated_stage._is_novel(component, existing, work_gray.shape):
            existing.append(component)
    for component in layered:
        # Complex-outline context can make the validated second-pass layered route
        # propose a thin cell wholly inside a mature base rectangle. It is a
        # border echo, not another physical redaction component.
        if _existing_union_coverage(component, existing) >= 0.80:
            continue
        if validated_stage._is_novel(component, existing, work_gray.shape):
            existing.append(component)
    promoted, supported_page_frames, promoted_mask = _promote_supported_lsd_outlines(
        work_gray, line_height, existing, lsd_candidates
    )
    for component in promoted:
        if validated_stage._is_novel(component, existing, work_gray.shape):
            existing.append(component)
    line_cycles, line_cycle_mask, line_cycle_diagnostics = (
        _line_segment_cycle_candidates(
            work_gray,
            artifacts["edges"],
            line_height,
            existing,
        )
    )
    for component in line_cycles:
        if validated_stage._is_novel(component, existing, work_gray.shape):
            existing = [
                prior
                for prior in existing
                if not _cycle_absorbs_partial_component(
                    component, prior, work_gray.shape
                )
            ]
            existing.append(component)
    page_open, page_open_mask = _three_side_page_outline_candidates(
        work_gray,
        artifacts["edges"],
        line_height,
        existing,
    )
    page_frame_candidates = _dedupe_page_frame_candidates(
        supported_page_frames + page_open
    )
    for component in supported_page_frames:
        cv2.polylines(
            page_open_mask,
            [np.asarray(component.polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )

    final_regions, content_bounds, layout_roles = validated_stage._group_and_order(
        existing, work_gray, line_mask
    )
    if input_scale != 1.0:
        inverse = np.asarray(
            diagnostics["inverse_affine_to_original"], dtype=np.float64
        )
        inverse /= input_scale
        diagnostics["inverse_affine_to_original"] = inverse.tolist()
    diagnostics.update(
        {
            "detector_release": "production-detection",
            "input_image_size_wh": [int(gray.shape[1]), int(gray.shape[0])],
            "canonical_detection_size_wh": [
                int(canonical.shape[1]),
                int(canonical.shape[0]),
            ],
            "input_to_detection_scale": round(float(input_scale), 8),
            "resolution_policy": (
                "All geometry is evaluated at the validated 200-DPI physical "
                "scale and mapped back to the input raster."
            ),
            "marginal_open_outline_rescue_count": len(marginal),
            "blackout_partition_rescue_count": len(blackout),
            "layered_outline_rescue_count": len(layered),
            "supported_outline_count": len(promoted),
            "line_segment_cycle_count": len(line_cycles),
            "page_frame_candidate_count": len(page_frame_candidates),
            "page_frame_candidates": [
                {
                    "box_xyxy": list(map(int, component.box)),
                    "polygon_xy": [
                        list(map(int, point)) for point in component.polygon
                    ],
                    "source": component.source,
                    "score": round(float(component.score), 6),
                }
                for component in page_frame_candidates
            ],
            "estimated_content_bounds_x": list(content_bounds),
            "region_layout_roles": layout_roles,
            "component_count_before_grouping": len(existing),
            "region_count": len(final_regions),
            "detector_policy": (
                "Single-page, answer-blind geometry. The production detector retains every validated "
                "component and adds only independently measured supported "
                "outlines. Page-band release frames are retained as diagnostic "
                "candidates but are not emitted as redactions; benchmark IDs "
                "and answer text are unavailable."
            ),
        }
    )
    diagnostics.update(complex_diagnostics)
    diagnostics.update(line_cycle_diagnostics)
    artifacts["shipping_blackout_partitions"] = blackout_mask
    artifacts["shipping_layered_outlines"] = layered_mask
    artifacts["complex_enclosed_outlines"] = complex_mask
    artifacts["line_segment_cycles"] = line_cycle_mask
    artifacts["three_side_page_outlines"] = page_open_mask
    artifacts["supported_outlines"] = promoted_mask
    return final_regions, artifacts, diagnostics


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
    detector_version: str = "production-detection",
) -> dict[str, Any]:
    payload = layered_stage.process_record(
        record,
        out_root=out_root,
        save_debug_masks=save_debug_masks,
        detector_fn=detect_redaction_regions_with_artifacts,
        detector_version=detector_version,
    )
    roles = payload.get("detector_diagnostics", {}).get(
        "region_layout_roles", {}
    )
    for index, region in enumerate(payload.get("redaction_regions", []), start=1):
        region["layout_role"] = roles.get(index, roles.get(str(index), "unknown"))
    metadata = Path(payload["output_files"]["metadata_json"])
    metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def run_box_pipeline(**kwargs: Any) -> dict[str, Any]:
    return layered_stage.run_box_pipeline(
        **kwargs,
        record_processor=process_record,
        detector_version="production-detection",
    )
