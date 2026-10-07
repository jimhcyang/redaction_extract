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
from . import _layered_detection as layered_stage


# Input discovery and data contracts remain stable across detector stages.
PDFPair = layered_stage.PDFPair
PairCollectionStats = layered_stage.PairCollectionStats
InputRecord = layered_stage.InputRecord
BoxComponent = layered_stage.BoxComponent
RedactionRegion = layered_stage.RedactionRegion
collect_pdf_pairs_with_stats = layered_stage.collect_pdf_pairs_with_stats
render_pdf_to_images = layered_stage.render_pdf_to_images
iter_input_records = layered_stage.iter_input_records

_rect_polygon = layered_stage._rect_polygon
_polygon_box = layered_stage._polygon_box
_area = layered_stage._area
_near_same_box = layered_stage._near_same_box


def _canonical_detection_image(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize 300-DPI page renders to the validated 200-DPI scale.

    Every geometric threshold in the detector is scale-relative, but raster
    line fragmentation still changes at 300 DPI. Pages whose long dimension is
    in the 300-DPI range are therefore downsampled by the physical 200/300
    ratio. Coordinates are mapped back to the original image in provenance.
    Native 300-DPI inference is deliberately not a second model.
    """

    height, width = gray.shape
    if max(height, width) < 2800:
        return gray, 1.0
    scale = 2.0 / 3.0
    normalized = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return normalized, scale


def _visible_ink_stats(
    gray: np.ndarray, box: tuple[int, int, int, int], line_height: int
) -> tuple[float, float, int]:
    """Summarize visible prose inside a proposed outline, excluding borders."""

    x1, y1, x2, y2 = box
    pad = max(3, int(round(line_height * 0.18)))
    roi = gray[y1 + pad : y2 - pad, x1 + pad : x2 - pad]
    if roi.size == 0:
        return 1.0, 1.0, 10**9
    ink = roi < 175
    row_density = np.mean(ink, axis=1)
    active = row_density > 0.05
    longest = run = 0
    for value in active:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return float(np.mean(ink)), float(np.mean(active)), longest


def _is_novel(
    candidate: BoxComponent,
    existing: Iterable[BoxComponent],
    image_shape: tuple[int, int],
) -> bool:
    tolerance = max(3, round(min(image_shape) * 0.006))
    return not any(
        _near_same_box(candidate.box, component.box, tolerance)
        for component in existing
    )


def _lsd_geometry_rescues_cached(
    gray: np.ndarray,
    line_height: int,
    existing: Iterable[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Run the validated layered line-segment rules with exact per-page memoization."""

    del existing  # The layered detector keeps this parameter for API symmetry but does not use it.
    height, width = gray.shape
    raw_horizontal, raw_vertical = layered_stage._oriented_line_segments(gray)
    horizontal = raw_horizontal + geometry_stage._merge_oriented_segments(
        raw_horizontal, "h", line_height, gray.shape
    )
    vertical = raw_vertical + geometry_stage._merge_oriented_segments(
        raw_vertical, "v", line_height, gray.shape
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 110)
    boundary_margin = max(4.0, min(height, width) * 0.018)
    side_tolerance = max(7.0, line_height * 1.45)
    corner_tolerance = max(8.0, line_height * 0.55)
    minimum_height = max(5.0, line_height * 0.32)
    minimum_width = max(18.0, line_height * 1.15)
    raw: list[BoxComponent] = []
    side_cache: dict[
        tuple[float, float, float],
        tuple[tuple[float, float, bool], ...],
    ] = {}
    quality_cache: dict[
        tuple[int, int, int, int],
        tuple[float, float, float, float] | None,
    ] = {}
    vertical_anchor_cache: dict[tuple[float, float], bool] = {}

    def side_positions(
        observed: float, top: float, bottom: float
    ) -> list[tuple[float, float, bool]]:
        key = (observed, top, bottom)
        if key not in side_cache:
            side_cache[key] = tuple(
                geometry_stage._candidate_side_positions(
                    vertical, observed, top, bottom, side_tolerance
                )
            )
        return list(side_cache[key])

    def interior_quality(
        box: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float] | None:
        if box not in quality_cache:
            quality_cache[box] = geometry_stage._robust_interior_quality(
                gray, edges, box, line_height
            )
        return quality_cache[box]

    def vertical_anchor(x_value: float, y_value: float) -> bool:
        key = (x_value, y_value)
        if key not in vertical_anchor_cache:
            vertical_anchor_cache[key] = any(
                abs(segment.midpoint[0] - x_value) <= side_tolerance
                and geometry_stage._endpoint_near(
                    y_value, segment, 1, corner_tolerance
                )
                for segment in vertical
            )
        return vertical_anchor_cache[key]

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
            left_options = side_positions(observed_left, top_y, bottom_y)
            right_options = side_positions(observed_right, top_y, bottom_y)
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
                    quality = interior_quality(box)
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
                                has_vertical_anchor = True
                            else:
                                has_vertical_anchor = vertical_anchor(
                                    x_value, y_value
                                )
                            if (
                                horizontal_anchor
                                and has_vertical_anchor
                                and side_observed
                            ):
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
            layered_stage._contains(previous.box, candidate.box, margin=containment_margin)
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


def _marginal_open_outline_candidates(
    gray: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
) -> list[BoxComponent]:
    """Recover header/footer boxes whose release stamp masks one border.

    The fallback contour detector is used only in the top/bottom page bands.
    A candidate must be broad, mostly blank, have a measured three-sided
    outline, and avoid sustained body prose. This handles release boilerplate
    without treating arbitrary page headings as masks.
    """

    height, width = gray.shape
    # A crop edge is arbitrary and cannot stand in for a missing physical page
    # border. Restrict this rescue to full-page renders.
    if min(height, width) < 1000:
        return []
    dark = core_stage._binarize_dark(gray)
    enhanced, faint = core_stage._binarize_faint(gray)
    edges = cv2.Canny(enhanced, 30, 110)
    fallback = geometry_stage._rectified_outline_fallback(
        gray, dark, faint, edges, line_height
    )
    accepted: list[BoxComponent] = []
    for candidate in fallback:
        x1, y1, x2, y2 = candidate.box
        box_width, box_height = x2 - x1, y2 - y1
        marginal = y1 <= height * 0.19 or y2 >= height * 0.82
        if (
            not marginal
            or box_width < max(width * 0.24, line_height * 7.0)
            # Release stamps frequently form small rectangular frames of
            # their own. The missing-border rescue is only defensible for a
            # substantial redacted area, not a stamp-sized header/footer.
            or box_height < max(line_height * 4.0, height * 0.065)
            or box_height > height * 0.42
            or _area(candidate.box) > width * height * 0.30
            or not _is_novel(candidate, existing + accepted, gray.shape)
        ):
            continue
        candidate_area = max(1, _area(candidate.box))
        if any(
            layered_stage._intersection_area(candidate.box, component.box) / candidate_area
            >= 0.30
            for component in existing
        ):
            continue
        dark_fraction, active_rows, longest = _visible_ink_stats(
            gray, candidate.box, line_height
        )
        if (
            dark_fraction > 0.055
            or active_rows > 0.13
            or longest > max(line_height * 1.55, box_height * 0.34)
        ):
            continue
        accepted.append(
            BoxComponent(
                box=candidate.box,
                polygon=candidate.polygon,
                source=(
                    "shipping_marginal_open_outline:"
                    f"ink={dark_fraction:.3f}:rows={active_rows:.3f}"
                ),
                score=float(candidate.score) + 0.20,
            )
        )
    return accepted


def _filled_run_rectangles(
    component_mask: np.ndarray,
    offset: tuple[int, int],
    line_height: int,
) -> list[tuple[int, int, int, int]]:
    """Decompose one connected blackout into stable horizontal run bands."""

    minimum_width = max(24, int(round(line_height * 1.25)))
    endpoint_tolerance = max(4, int(round(line_height * 0.42)))
    minimum_height = max(7, int(round(line_height * 0.34)))
    rows: list[tuple[int, int, int]] = []
    close_width = max(3, int(round(line_height * 0.22)))
    closed = cv2.morphologyEx(
        component_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1)),
        iterations=1,
    )
    for y in range(closed.shape[0]):
        values = closed[y] > 0
        padded = np.pad(values.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            if end - start >= minimum_width:
                rows.append((y, int(start), int(end)))

    tracks: list[list[tuple[int, int, int]]] = []
    active: list[int] = []
    previous_y = -2
    for y in sorted({row[0] for row in rows}):
        current = [row for row in rows if row[0] == y]
        if y != previous_y + 1:
            active = []
        available = set(active)
        next_active: list[int] = []
        for row in current:
            _, start, end = row
            best: tuple[float, int] | None = None
            for track_index in available:
                _, prior_start, prior_end = tracks[track_index][-1]
                overlap = max(0, min(end, prior_end) - max(start, prior_start))
                shorter = max(1, min(end - start, prior_end - prior_start))
                endpoint_delta = abs(start - prior_start) + abs(end - prior_end)
                if (
                    overlap / shorter >= 0.72
                    and abs(start - prior_start) <= endpoint_tolerance
                    and abs(end - prior_end) <= endpoint_tolerance
                ):
                    score = endpoint_delta - overlap * 0.01
                    if best is None or score < best[0]:
                        best = (score, track_index)
            if best is None:
                tracks.append([row])
                next_active.append(len(tracks) - 1)
            else:
                track_index = best[1]
                tracks[track_index].append(row)
                available.remove(track_index)
                next_active.append(track_index)
        active = next_active
        previous_y = y

    offset_x, offset_y = offset
    boxes: list[tuple[int, int, int, int]] = []
    for track in tracks:
        if len(track) < minimum_height:
            continue
        starts = [row[1] for row in track]
        ends = [row[2] for row in track]
        x1 = int(np.percentile(starts, 15))
        x2 = int(np.percentile(ends, 85))
        y1, y2 = track[0][0], track[-1][0] + 1
        if x2 - x1 < minimum_width:
            continue
        boxes.append((offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2))
    return boxes


def _blackout_partition_candidates(
    gray: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray]:
    """Recover and decompose dense C-, step-, and hammer-shaped blackouts."""

    height, width = gray.shape
    raw = np.where(gray < 108, 255, 0).astype(np.uint8)
    cleaned = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )
    # A modest two-dimensional opening removes character strokes while
    # preserving line-height blackout masks.
    kernel = max(3, int(round(line_height * 0.14)))
    if kernel % 2 == 0:
        kernel += 1
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel)),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    candidates: list[BoxComponent] = []
    debug = np.zeros_like(gray)
    page_area = height * width
    for label in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[label])
        if (
            area < max(700, page_area * 0.00016)
            or box_width < line_height * 1.25
            # Solid-ink recovery is for blackout masks, not glyph strokes or
            # rules. Thin ordinary redactions remain the outline detector's
            # responsibility.
            or box_height < line_height * 0.65
        ):
            continue
        local = np.where(
            labels[y : y + box_height, x : x + box_width] == label, 255, 0
        ).astype(np.uint8)
        boxes = _filled_run_rectangles(local, (x, y), line_height)
        if not boxes:
            continue
        source = f"rectilinear_contour_partition:shipping_blackout_{label}"
        for box in boxes:
            bx1, by1, bx2, by2 = box
            if by2 - by1 < line_height * 0.65:
                continue
            edge_margin = max(3, int(round(min(height, width) * 0.018)))
            if (
                bx1 <= edge_margin
                and bx2 >= width - edge_margin
                and by2 - by1 < line_height * 2.5
            ):
                continue
            roi = raw[by1:by2, bx1:bx2]
            fill = float(np.mean(roi > 0)) if roi.size else 0.0
            if fill < 0.62:
                continue
            component = BoxComponent(
                box=box,
                polygon=_rect_polygon(box),
                source=source,
                score=6.0 + fill,
            )
            if not _is_novel(component, existing + candidates, gray.shape):
                continue
            candidates.append(component)
            cv2.rectangle(debug, (bx1, by1), (bx2 - 1, by2 - 1), 255, 1)
    return candidates, debug


def _layered_outline_candidates(
    gray: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
    lsd_candidates: list[BoxComponent] | None = None,
) -> tuple[list[BoxComponent], np.ndarray]:
    """Retain strongly supported outer layers that the layered detector treated as envelopes."""

    # Adjacency is meaningful only with a complete page coordinate frame.
    # Cropped examples can place an ordinary outline against an artificial
    # crop edge and cannot establish body-text versus marginal reading flow.
    if min(gray.shape) < 1000:
        return [], np.zeros_like(gray)
    candidates = lsd_candidates
    if candidates is None:
        candidates, _, _ = layered_stage._lsd_geometry_rescues(
            gray, line_height, existing
        )
    accepted: list[BoxComponent] = []
    debug = np.zeros_like(gray)
    for candidate in candidates:
        if not _is_novel(candidate, existing + accepted, gray.shape):
            continue
        supports = layered_stage._lsd_support_values(candidate)
        corners = geometry_stage._lsd_corner_count(candidate)
        if len(supports) != 4:
            continue
        x1, y1, x2, y2 = candidate.box
        box_width, box_height = x2 - x1, y2 - y1
        dark_fraction, active_rows, longest = _visible_ink_stats(
            gray, candidate.box, line_height
        )
        # The layered compact cover intentionally removes many mathematically valid
        # rectangles assembled from unrelated sides. Re-introduce only a
        # complete rectangle that is immediately adjacent to another physical
        # component in a line-wrap arrangement. Broad two-corner envelopes are
        # never accepted here, even when they happen to cover benchmark text.
        adjacent = False
        for component in existing:
            ox1, oy1, ox2, oy2 = component.box
            vertical_gap = max(0, max(y1, oy1) - min(y2, oy2))
            overlap_width = max(0, min(x2, ox2) - max(x1, ox1))
            positive_overlap = layered_stage._intersection_area(candidate.box, component.box)
            shared_side = min(abs(x1 - ox1), abs(x2 - ox2)) <= line_height * 0.55
            if (
                positive_overlap == 0
                and vertical_gap <= line_height * 0.42
                and overlap_width >= min(box_width, ox2 - ox1) * 0.55
                and shared_side
            ):
                adjacent = True
                break
        complete_adjacent_outline = (
            corners >= 4
            and candidate.score >= 6.65
            and min(supports) >= 0.55
            and adjacent
            and active_rows <= 0.20
        )
        if not complete_adjacent_outline:
            continue
        # A long run of ordinary prose indicates an exterior envelope rather
        # than a physical mask, even if two distant borders happen to align.
        if (
            longest > max(line_height * 2.25, box_height * 0.42)
            and dark_fraction > 0.070
        ):
            continue
        accepted.append(
            BoxComponent(
                box=candidate.box,
                polygon=candidate.polygon,
                source=(
                    "shipping_layered_outline:"
                    f"corners={corners}:ink={dark_fraction:.3f}:"
                    f"rows={active_rows:.3f}:{candidate.source}"
                ),
                score=float(candidate.score) + 0.08,
            )
        )
        cv2.polylines(
            debug,
            [np.asarray(candidate.polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )
    return accepted, debug


def _component_layout_role(
    component: BoxComponent, content_left: int, content_right: int, line_height: int
) -> str:
    x1, _, x2, _ = component.box
    center = (x1 + x2) / 2.0
    slack = line_height * 0.75
    if x2 < content_left - slack or x1 > content_right + slack:
        return "margin"
    if content_left - slack <= center <= content_right + slack:
        return "main_text"
    return "boundary"


def _text_mask_and_margins_fast(
    gray: np.ndarray,
    line_mask: np.ndarray,
    components: list[BoxComponent],
) -> tuple[np.ndarray, int, int, int]:
    """Compute the validated layered text layout without image-wide per-label scans."""

    height, width = gray.shape
    line_height = layered_stage._text_line_height(gray, line_mask)
    text = (gray < 170).astype(np.uint8) * 255
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
        local_labels = labels[y : y + glyph_height, x : x + glyph_width]
        local_clean = clean[y : y + glyph_height, x : x + glyph_width]
        local_clean[local_labels == label] = 255
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
                and other_weight
                >= max(width * 0.05, run_weights[run_index] * 1.5)
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

    substantial: list[
        tuple[
            int,
            int,
            float,
            list[tuple[int, int, int, int, int, float]],
        ]
    ] = []
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
        left = max(
            0,
            int(round(np.percentile([item[0] for item in substantial], 25))),
        )
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

    for run in all_runs:
        run_left = min(item[0] for item in run)
        run_right = max(item[0] + item[2] for item in run)
        span = run_right - run_left
        outside_body = run_right < left or run_left > right
        if run in marginal_runs or (outside_body and span < width * 0.12):
            for x, y, glyph_width, glyph_height, _, _ in run:
                clean[y : y + glyph_height, x : x + glyph_width] = 0
    return clean, left, right, pitch


def _group_components_with_layout(
    components: list[BoxComponent],
    gray: np.ndarray,
    *,
    text_mask: np.ndarray,
    content_left: int,
    content_right: int,
    line_pitch: int,
    line_height: int,
) -> list[RedactionRegion]:
    """Apply the validated layered grouping rules to one precomputed text layout."""

    if not components:
        return []
    height, width = gray.shape
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
            if layered_stage._substantial_physical_contact(
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
                continue
            delta = sy1 - fy1
            if delta < max(4, line_pitch * 0.30) or delta > line_pitch * 1.80:
                continue
            after = (
                fx2,
                fy1 + 1,
                content_right + 1,
                max(fy1 + 2, fy2 - 1),
            )
            before = (
                content_left,
                sy1 + 1,
                sx1,
                max(sy1 + 2, sy2 - 1),
            )
            if layered_stage._ink_fraction(text_mask, after) > 0.030:
                continue
            if layered_stage._ink_fraction(text_mask, before) > 0.030:
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


def _group_and_order(
    components: list[BoxComponent], gray: np.ndarray, line_mask: np.ndarray
) -> tuple[list[RedactionRegion], tuple[int, int], dict[int, str]]:
    text_mask, content_left, content_right, line_pitch = (
        _text_mask_and_margins_fast(gray, line_mask, components)
    )
    line_height = layered_stage._text_line_height(gray, line_mask)
    regions = _group_components_with_layout(
        components,
        gray,
        text_mask=text_mask,
        content_left=content_left,
        content_right=content_right,
        line_pitch=line_pitch,
        line_height=line_height,
    )
    roles: dict[int, str] = {}
    ordered: list[tuple[int, int, int, RedactionRegion, str]] = []
    for index, region in enumerate(regions):
        component_roles = [
            _component_layout_role(
                component, content_left, content_right, line_height
            )
            for component in region.components
        ]
        role = (
            "main_text"
            if "main_text" in component_roles
            else ("boundary" if "boundary" in component_roles else "margin")
        )
        role_order = {"main_text": 0, "boundary": 1, "margin": 2}[role]
        ordered.append(
            (
                role_order,
                min(component.box[1] for component in region.components),
                min(component.box[0] for component in region.components),
                region,
                role,
            )
        )
    ordered.sort(key=lambda item: item[:3])
    final = [item[3] for item in ordered]
    roles = {index: item[4] for index, item in enumerate(ordered, start=1)}
    return final, (content_left, content_right), roles


def _base_detection_with_lsd_candidates(
    gray: np.ndarray,
) -> tuple[
    list[BoxComponent],
    dict[str, np.ndarray],
    dict[str, Any],
    list[BoxComponent],
]:
    """Run the validated layered decision path while retaining its LSD candidates.

    The layered detector normally discards the candidate objects after refinement and exposes
    only their debug mask. The shipping layer needs those same candidates for
    one conservative adjacent-layer check. Keeping them here avoids repeating
    the most expensive line-segment search without changing the layered base logic.
    """

    work_gray, inverse, rotation = core_stage._deskew(gray)
    dark = core_stage._binarize_dark(work_gray)
    enhanced, faint = core_stage._binarize_faint(work_gray)
    h_segments, v_segments, horizontal, vertical = geometry_stage._extract_axis_segments(
        dark, faint
    )
    line_mask = cv2.bitwise_or(horizontal, vertical)
    edges = cv2.Canny(enhanced, 30, 110)
    line_height = layered_stage._text_line_height(work_gray, line_mask)
    zones = geometry_stage._line_zones(horizontal, vertical)
    proposals: list[layered_stage.RectangleProposal] = []
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
        geometry_stage._closed_blank_contour_candidates(
            work_gray, dark, edges, line_height
        )
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
    solid, solid_mask = layered_stage._shape_preserving_solid_candidates(
        work_gray, line_height
    )
    blackout, blackout_mask = layered_stage._shape_preserving_blackout_candidates(
        work_gray, line_height, outline + solid
    )
    base_components = geometry_stage._dedupe_components(
        outline + solid + blackout, work_gray.shape
    )
    lsd_candidates, lsd_mask, lsd_diagnostics = _lsd_geometry_rescues_cached(
        work_gray, line_height, base_components
    )
    refined_base, remaining_lsd, refinement_count = geometry_stage._apply_lsd_refinements(
        base_components, lsd_candidates, work_gray.shape
    )
    promoted_base, remaining_lsd, promotion_count = (
        layered_stage._promote_supported_outer_rectangles(
            refined_base,
            remaining_lsd,
            work_gray.shape,
            line_height,
        )
    )
    lsd_rescues = layered_stage._layer_occlusion_rescues(
        remaining_lsd,
        promoted_base,
        work_gray.shape,
        line_height,
    )
    combined = geometry_stage._dedupe_components(
        promoted_base + lsd_rescues, work_gray.shape
    )
    combined = layered_stage._suppress_lsd_border_echoes(
        combined, work_gray.shape, line_height
    )
    canonical = geometry_stage._canonicalize_components(
        combined, work_gray.shape, line_height
    )
    completed, step_completion_mask, step_completion_count = (
        layered_stage._shared_edge_step_completions(work_gray, line_height, canonical)
    )
    components, suppression = layered_stage._suppress_text_frames_and_glyphs(
        work_gray, line_mask, completed, line_height
    )
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
        "base_component_count": len(components),
        "estimated_text_line_height": line_height,
        "detector_policy": (
            "Single-page geometry only: direction-invariant border recovery, "
            "shared-edge step completion, shape-preserving rectilinear solid "
            "masks, text-bridge suppression, and reading-order semantic grouping."
        ),
    }
    diagnostics.update(closed_diagnostics)
    diagnostics.update(enclosed_diagnostics)
    diagnostics.update(lsd_diagnostics)
    diagnostics["lsd_geometry_rescue_count"] = len(lsd_rescues)
    return components, artifacts, diagnostics, lsd_candidates


def detect_redaction_regions_with_artifacts(
    gray: np.ndarray,
) -> tuple[list[RedactionRegion], dict[str, np.ndarray], dict[str, Any]]:
    canonical, input_scale = _canonical_detection_image(gray)
    base_components, artifacts, diagnostics, lsd_candidates = (
        _base_detection_with_lsd_candidates(canonical)
    )
    work_gray = artifacts["deskewed_source"]
    line_mask = artifacts["lines_mask"]
    line_height = layered_stage._text_line_height(work_gray, line_mask)
    marginal = _marginal_open_outline_candidates(
        work_gray, line_height, base_components
    )
    blackout, blackout_mask = _blackout_partition_candidates(
        work_gray, line_height, base_components + marginal
    )
    layered, layered_mask = _layered_outline_candidates(
        work_gray,
        line_height,
        base_components + marginal + blackout,
        lsd_candidates,
    )
    # Layered detections are the validated base. A rescue may add genuinely new
    # geometry, but must never replace a near-identical base component merely
    # because its proposal score is numerically higher.
    combined = list(base_components)
    for component in marginal + blackout + layered:
        if _is_novel(component, combined, work_gray.shape):
            combined.append(component)
    final_regions, content_bounds, layout_roles = _group_and_order(
        combined, work_gray, line_mask
    )

    if input_scale != 1.0:
        inverse = np.asarray(
            diagnostics["inverse_affine_to_original"], dtype=np.float64
        )
        inverse /= input_scale
        diagnostics["inverse_affine_to_original"] = inverse.tolist()
    diagnostics.update(
        {
            "detector_release": "validated-detection",
            "input_image_size_wh": [int(gray.shape[1]), int(gray.shape[0])],
            "canonical_detection_size_wh": [
                int(canonical.shape[1]),
                int(canonical.shape[0]),
            ],
            "input_to_detection_scale": round(float(input_scale), 8),
            "resolution_policy": (
                "Pages in the native 300-DPI size range are analyzed at the "
                "validated 200-DPI physical scale; provenance polygons map "
                "back to the original raster."
            ),
            "marginal_open_outline_rescue_count": len(marginal),
            "blackout_partition_rescue_count": len(blackout),
            "layered_outline_rescue_count": len(layered),
            "estimated_content_bounds_x": list(content_bounds),
            "region_layout_roles": layout_roles,
            "component_count_before_grouping": len(combined),
            "region_count": len(final_regions),
            "detector_policy": (
                "Single-page, answer-blind geometry at a canonical physical "
                "scale with release-stamp-tolerant marginal closure, dense "
                "blackout decomposition, layered-outline recovery, and "
                "body-versus-margin reading structure."
            ),
        }
    )
    artifacts["shipping_blackout_partitions"] = blackout_mask
    artifacts["shipping_layered_outlines"] = layered_mask
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
    detector_version: str = "validated-detection",
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
        detector_version="validated-detection",
    )
