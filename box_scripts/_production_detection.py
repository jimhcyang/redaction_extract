"""High-recall, answer-blind redaction geometry built on validated candidates."""

from __future__ import annotations

import json
import math
import re
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


def _replace_layered_blackout_union(
    gray_shape: tuple[int, int],
    base: list[BoxComponent],
    blackout: list[BoxComponent],
) -> tuple[list[BoxComponent], int]:
    """Drop one concave solid envelope when overlapping rectangles explain it."""

    if len(blackout) < 2:
        return base, 0
    retained: list[BoxComponent] = []
    removed = 0
    for component in base:
        if not component.source.startswith(("solid_fill", "dense_blackout")):
            retained.append(component)
            continue
        if len(component.polygon) <= 4:
            retained.append(component)
            continue
        component_mask = np.zeros(gray_shape, dtype=np.uint8)
        cv2.fillPoly(
            component_mask,
            [np.asarray(component.polygon, dtype=np.int32)],
            1,
        )
        component_pixels = int(np.count_nonzero(component_mask))
        members: list[BoxComponent] = []
        for candidate in blackout:
            candidate_mask = np.zeros(gray_shape, dtype=np.uint8)
            cv2.fillPoly(
                candidate_mask,
                [np.asarray(candidate.polygon, dtype=np.int32)],
                1,
            )
            candidate_pixels = int(np.count_nonzero(candidate_mask))
            overlap = int(np.count_nonzero(candidate_mask & component_mask))
            if candidate_pixels and overlap / candidate_pixels >= 0.96:
                members.append(candidate)
        overlapping_members = any(
            layered_stage._intersection_area(left.box, right.box)
            / max(1, min(geometry_stage._area(left.box), geometry_stage._area(right.box)))
            >= 0.08
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        )
        if len(members) < 2 or not overlapping_members:
            retained.append(component)
            continue
        union = np.zeros(gray_shape, dtype=np.uint8)
        for candidate in members:
            cv2.fillPoly(
                union,
                [np.asarray(candidate.polygon, dtype=np.int32)],
                1,
            )
        union_pixels = int(np.count_nonzero(union))
        overlap = int(np.count_nonzero(union & component_mask))
        if (
            component_pixels
            and union_pixels
            and overlap / component_pixels >= 0.96
            and overlap / union_pixels >= 0.96
        ):
            removed += 1
            continue
        retained.append(component)
    return retained, removed


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
        candidate_is_detail = ":nested_detail:" in candidate.source
        duplicate = False
        for prior in accepted:
            # A small independently ruled detail nested inside a broad release
            # frame is a separate physical outline, not a duplicate rendering
            # of that frame.  Details still deduplicate against other details.
            if candidate_is_detail != (":nested_detail:" in prior.source):
                continue
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
        # A partially released mask can retain both horizontal rails and one
        # complete wall while the opposite wall is replaced by a diagonal or
        # release stamp.  This is deliberately stricter than ``blank_body``:
        # exactly one vertical side must be absent, the remaining three sides
        # must be strongly measured, and the interior must be nearly empty.
        one_vertical_side_occluded = (
            candidate.score >= 4.90
            and corners >= 2
            and horizontal_min >= 0.80
            and max(supports[2:]) >= 0.88
            and min(supports[2:]) <= 0.12
            and support_sum >= 2.65
            and box_width >= line_height * 5.0
            and line_height * 1.05 <= box_height <= line_height * 5.0
            and ink <= 0.018
            and active_rows <= 0.08
            and longest <= max(5, int(round(line_height * 0.35)))
            and y1 >= height * 0.08
            and y2 <= height * 0.92
            and x1 >= width * 0.04
            and x2 <= width * 0.96
        )
        independently_complete_thin_body = (
            candidate.score >= 6.80
            and corners == 4
            and min(supports) >= 0.68
            and support_sum >= 3.35
            and box_width >= line_height * 6.0
            and line_height * 0.82 <= box_height < line_height * 1.05
            and ink <= 0.012
            and active_rows <= 0.04
            and longest <= 2
            and y1 >= height * 0.08
            and y2 <= height * 0.92
            and x1 >= width * 0.04
            and x2 <= width * 0.96
        )
        page_band = y1 <= height * 0.18 or y2 >= height * 0.90
        complete_page_furniture = (
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
        # Release stamps are frequently composited over the original page and
        # erase one side of an otherwise measured header/footer rectangle.
        # Keep these as physical-outline diagnostics, never answer-bearing
        # body redactions.  The extreme-band and aspect-ratio gates prevent an
        # ordinary underlined heading from entering this route.
        occluded_page_furniture = (
            (y1 <= height * 0.075 or y2 >= height * 0.94)
            and candidate.score >= 5.50
            and corners >= 2
            and box_width >= max(width * 0.10, line_height * 5.0)
            and line_height * 1.45 <= box_height <= line_height * 5.0
            and box_width / max(1, box_height) >= 2.8
            and support_sum >= 2.55
            and max(supports[:2]) >= 0.65
            and min(supports[2:]) >= 0.45
            and ink <= 0.14
            and active_rows <= 0.38
            and longest <= max(22, int(round(line_height * 0.80)))
        )
        page_furniture = complete_page_furniture or occluded_page_furniture

        # A small independently ruled rectangle can be nested inside a broad
        # release-frame outline.  It remains diagnostic page furniture, but it
        # is a distinct physical component and should not disappear merely
        # because it is too small for the body-redaction route.
        nested_page_detail = any(
            max(0, min(x2, frame.box[2]) - max(x1, frame.box[0]))
            * max(0, min(y2, frame.box[3]) - max(y1, frame.box[1]))
            / candidate_area
            >= 0.94
            for frame in page_frames
        ) and (
            candidate.score >= 6.80
            and corners == 4
            and min(supports) >= 0.80
            and box_width >= line_height * 0.90
            and box_height >= line_height * 0.65
            and ink <= 0.025
            and active_rows <= 0.08
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
        if page_furniture or nested_page_detail:
            route = "nested_detail" if nested_page_detail else "outline"
            page_frames.append(
                BoxComponent(
                    box=candidate.box,
                    polygon=candidate.polygon,
                    source=f"page_frame_candidate:{route}:{candidate.source}",
                    score=float(candidate.score),
                )
            )
            continue
        if not (
            blank_body
            or one_vertical_side_occluded
            or independently_complete_thin_body
            or large_edge_blank
        ):
            continue
        route = (
            "blank"
            if blank_body
            else (
                "one_vertical_side_occluded"
                if one_vertical_side_occluded
                else (
                    "independently_complete_thin_body"
                    if independently_complete_thin_body
                    else "large_edge_blank"
                )
            )
        )
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


def _nested_measured_outline_candidates(
    gray: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
    candidates: list[BoxComponent],
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover independently ruled rectangles inside a larger blank mask.

    Transparent drawing layers can place a small physical redaction over a
    broad mask. The broad mask is valid, but ordinary novelty checks then hide
    the smaller rectangle. This route accepts only four-corner LSD geometry
    with strong support on every side, a blank interior, and at most one side
    reused from the enclosing component. Two collinear proposals split by a
    thick raster seam are merged before the independence check.
    """

    side_tolerance = max(3, round(min(gray.shape) * 0.006))
    seam_tolerance = max(3, int(round(line_height * 0.18)))
    measured: list[tuple[BoxComponent, BoxComponent]] = []
    considered = 0
    rejected_support = 0
    rejected_text = 0
    rejected_shared_sides = 0

    def enclosing_component(candidate: BoxComponent) -> BoxComponent | None:
        x1, y1, x2, y2 = candidate.box
        candidate_area = max(1, (x2 - x1) * (y2 - y1))
        options: list[BoxComponent] = []
        for component in existing:
            cx1, cy1, cx2, cy2 = component.box
            overlap = max(0, min(x2, cx2) - max(x1, cx1)) * max(
                0, min(y2, cy2) - max(y1, cy1)
            )
            component_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if (
                overlap / candidate_area >= 0.94
                and component_area >= candidate_area * 1.18
            ):
                options.append(component)
        return min(options, key=lambda item: geometry_stage._area(item.box), default=None)

    def aligned_side_count(
        box: tuple[int, int, int, int],
        enclosing: tuple[int, int, int, int],
    ) -> int:
        return sum(
            abs(left - right) <= side_tolerance
            for left, right in zip(box, enclosing)
        )

    for candidate in candidates:
        enclosing = enclosing_component(candidate)
        if enclosing is None:
            continue
        considered += 1
        supports = layered_stage._lsd_support_values(candidate)
        corners = geometry_stage._lsd_corner_count(candidate)
        x1, y1, x2, y2 = candidate.box
        box_width, box_height = x2 - x1, y2 - y1
        if (
            len(supports) != 4
            or corners != 4
            or candidate.score < 6.65
            or min(supports) < 0.70
            or sum(supports) < 3.40
            or box_width < line_height * 1.15
            or not line_height * 0.65 <= box_height <= line_height * 4.5
        ):
            rejected_support += 1
            continue
        ink, active_rows, longest = validated_stage._visible_ink_stats(
            gray, candidate.box, line_height
        )
        if (
            ink > 0.060
            or active_rows > 0.25
            or longest > max(10, int(round(line_height * 0.65)))
        ):
            rejected_text += 1
            continue
        measured.append((candidate, enclosing))

    # A thick internal raster seam can split one physical outline into two
    # high-confidence rectangles. Merge only overlapping proposals with both
    # outer sides aligned; merely adjacent physical boxes remain separate.
    merged: list[tuple[BoxComponent, BoxComponent, int]] = []
    consumed: set[int] = set()
    for index, (candidate, enclosing) in enumerate(measured):
        if index in consumed:
            continue
        box = candidate.box
        member_count = 1
        changed = True
        while changed:
            changed = False
            x1, y1, x2, y2 = box
            for other_index, (other, other_enclosing) in enumerate(measured):
                if other_index == index or other_index in consumed:
                    continue
                if other_enclosing is not enclosing:
                    continue
                ox1, oy1, ox2, oy2 = other.box
                same_vertical_sides = (
                    abs(x1 - ox1) <= side_tolerance
                    and abs(x2 - ox2) <= side_tolerance
                    and min(y2, oy2) - max(y1, oy1) >= 1
                    and min(y2, oy2) - max(y1, oy1) <= seam_tolerance
                )
                same_horizontal_sides = (
                    abs(y1 - oy1) <= side_tolerance
                    and abs(y2 - oy2) <= side_tolerance
                    and min(x2, ox2) - max(x1, ox1) >= 1
                    and min(x2, ox2) - max(x1, ox1) <= seam_tolerance
                )
                if not (same_vertical_sides or same_horizontal_sides):
                    continue
                box = (
                    min(x1, ox1),
                    min(y1, oy1),
                    max(x2, ox2),
                    max(y2, oy2),
                )
                consumed.add(other_index)
                member_count += 1
                changed = True
                break
        consumed.add(index)
        merged.append(
            (
                BoxComponent(
                    box=box,
                    polygon=geometry_stage._rect_polygon(box),
                    source=(
                        "nested_measured_outline:"
                        f"members={member_count}:seed={candidate.source}"
                    ),
                    score=float(candidate.score) + 0.03,
                ),
                enclosing,
                member_count,
            )
        )

    accepted: list[BoxComponent] = []
    shared_side_groups: dict[int, list[tuple[BoxComponent, BoxComponent, int]]] = {}
    debug = np.zeros_like(gray)
    merged_member_count = 0
    partition_replacement_count = 0
    step_replacement_count = 0
    for candidate, enclosing, member_count in merged:
        if aligned_side_count(candidate.box, enclosing.box) >= 2:
            if enclosing.source.startswith("shared_edge_step_completion:"):
                candidate = BoxComponent(
                    box=candidate.box,
                    polygon=candidate.polygon,
                    source=(
                        "nested_measured_step_replacement:"
                        f"{candidate.source}"
                    ),
                    score=float(candidate.score),
                )
                step_replacement_count += 1
            else:
                shared_side_groups.setdefault(id(enclosing), []).append(
                    (candidate, enclosing, member_count)
                )
                continue
        if not validated_stage._is_novel(
            candidate, existing + accepted, gray.shape
        ):
            continue
        accepted.append(candidate)
        merged_member_count += max(0, member_count - 1)
        cv2.polylines(
            debug,
            [np.asarray(candidate.polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )

    # A single enclosing proposal can be the envelope of two adjacent boxes
    # when their shared seam is independently measured.  Replace the envelope
    # only when two or more candidates collectively tile nearly all of it and
    # reach all four outer sides.  Small subdivisions therefore remain
    # rejected, while a genuine side-by-side pair retains both components.
    for group in shared_side_groups.values():
        enclosing = group[0][1]
        ex1, ey1, ex2, ey2 = enclosing.box
        enclosure_width = max(1, ex2 - ex1)
        enclosure_height = max(1, ey2 - ey1)
        local_union = np.zeros((enclosure_height, enclosure_width), dtype=np.uint8)
        boxes = [row[0].box for row in group]
        for candidate, _, _ in group:
            polygon = np.asarray(
                [(x - ex1, y - ey1) for x, y in candidate.polygon],
                dtype=np.int32,
            )
            cv2.fillPoly(local_union, [polygon], 1)
        bx1 = min(box[0] for box in boxes)
        by1 = min(box[1] for box in boxes)
        bx2 = max(box[2] for box in boxes)
        by2 = max(box[3] for box in boxes)
        enclosure_area = enclosure_width * enclosure_height
        union_coverage = float(np.count_nonzero(local_union) / enclosure_area)
        bounding_coverage = (
            max(0, bx2 - bx1) * max(0, by2 - by1) / enclosure_area
        )
        reaches_outer_sides = all(
            delta <= side_tolerance * 1.5
            for delta in (
                abs(bx1 - ex1),
                abs(by1 - ey1),
                abs(bx2 - ex2),
                abs(by2 - ey2),
            )
        )
        # Internal raster seams are common inside ordinary one-line boxes.
        # Treat measured subdivisions as distinct physical components only
        # when the enclosing mask is a broad, multi-token span and every
        # proposed partition is substantial relative to the text scale.  This
        # preserves genuine side-by-side redactions without splitting compact
        # boxes merely because a scan seam happens to be independently ruled.
        broad_partition = (
            enclosure_width >= max(gray.shape[1] * 0.25, line_height * 10.0)
            and enclosure_height >= line_height * 1.25
            and all(
                box[2] - box[0] >= line_height * 3.0
                for box in boxes
            )
        )
        side_by_side_partition = all(
            (box[3] - box[1]) / enclosure_height >= 0.72
            and (box[2] - box[0]) / enclosure_width <= 0.90
            for box in boxes
        )
        if (
            len(group) >= 2
            and union_coverage >= 0.78
            and bounding_coverage >= 0.88
            and reaches_outer_sides
            and broad_partition
            and side_by_side_partition
        ):
            for candidate, _, member_count in group:
                replacement = BoxComponent(
                    box=candidate.box,
                    polygon=candidate.polygon,
                    source=f"nested_measured_partition:{candidate.source}",
                    score=float(candidate.score),
                )
                # The enclosing component is intentionally being replaced, so
                # it must not veto one of its own complementary partitions.
                if validated_stage._is_novel(
                    replacement, accepted, gray.shape
                ):
                    accepted.append(replacement)
                    partition_replacement_count += 1
                    merged_member_count += max(0, member_count - 1)
                    cv2.polylines(
                        debug,
                        [np.asarray(replacement.polygon, dtype=np.int32)],
                        True,
                        255,
                        1,
                        cv2.LINE_8,
                    )
            continue
        rejected_shared_sides += len(group)

    return accepted, debug, {
        "nested_outline_considered_count": considered,
        "nested_outline_measured_count": len(measured),
        "nested_outline_merged_member_count": merged_member_count,
        "nested_outline_component_count": len(accepted),
        "nested_outline_rejected_support_count": rejected_support,
        "nested_outline_rejected_text_count": rejected_text,
        "nested_outline_rejected_shared_sides_count": rejected_shared_sides,
        "nested_outline_partition_replacement_count": partition_replacement_count,
        "nested_outline_step_replacement_count": step_replacement_count,
    }


def _remove_partitioned_envelopes(
    existing: list[BoxComponent],
    nested: list[BoxComponent],
) -> tuple[list[BoxComponent], int]:
    """Remove an envelope only when measured partition components replace it."""

    partitions = [
        component
        for component in nested
        if component.source.startswith("nested_measured_partition:")
    ]
    if len(partitions) < 2:
        return existing, 0
    retained: list[BoxComponent] = []
    removed = 0
    for enclosing in existing:
        ex1, ey1, ex2, ey2 = enclosing.box
        enclosure_area = max(1, (ex2 - ex1) * (ey2 - ey1))
        members: list[BoxComponent] = []
        for candidate in partitions:
            x1, y1, x2, y2 = candidate.box
            candidate_area = max(1, (x2 - x1) * (y2 - y1))
            overlap = max(0, min(x2, ex2) - max(x1, ex1)) * max(
                0, min(y2, ey2) - max(y1, ey1)
            )
            if overlap / candidate_area >= 0.94:
                members.append(candidate)
        if len(members) < 2:
            retained.append(enclosing)
            continue
        local_union = np.zeros((max(1, ey2 - ey1), max(1, ex2 - ex1)), dtype=np.uint8)
        for candidate in members:
            polygon = np.asarray(
                [(x - ex1, y - ey1) for x, y in candidate.polygon],
                dtype=np.int32,
            )
            cv2.fillPoly(local_union, [polygon], 1)
        if float(np.count_nonzero(local_union) / enclosure_area) >= 0.78:
            removed += 1
            continue
        retained.append(enclosing)
    return retained, removed


def _remove_text_crossing_step_envelopes(
    gray: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
    nested: list[BoxComponent],
) -> tuple[list[BoxComponent], int]:
    """Prefer measured blank slabs over a synthetic step crossing prose."""

    replacements = [
        component
        for component in nested
        if component.source.startswith("nested_measured_step_replacement:")
    ]
    if not replacements:
        return existing, 0
    retained: list[BoxComponent] = []
    removed = 0
    erosion = max(3, int(round(line_height * 0.18)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion, erosion))
    for enclosing in existing:
        if not enclosing.source.startswith("shared_edge_step_completion:"):
            retained.append(enclosing)
            continue
        enclosing_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(
            enclosing_mask,
            [np.asarray(enclosing.polygon, dtype=np.int32)],
            255,
        )
        replacement_mask = np.zeros_like(gray, dtype=np.uint8)
        member_count = 0
        for candidate in replacements:
            candidate_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.fillPoly(
                candidate_mask,
                [np.asarray(candidate.polygon, dtype=np.int32)],
                255,
            )
            candidate_pixels = int(np.count_nonzero(candidate_mask))
            overlap = int(
                np.count_nonzero((candidate_mask > 0) & (enclosing_mask > 0))
            )
            if candidate_pixels and overlap / candidate_pixels >= 0.90:
                replacement_mask = cv2.bitwise_or(
                    replacement_mask, candidate_mask
                )
                member_count += 1
        if not member_count:
            retained.append(enclosing)
            continue
        residual = cv2.bitwise_and(
            enclosing_mask,
            cv2.bitwise_not(replacement_mask),
        )
        residual_core = cv2.erode(residual, kernel, iterations=1)
        residual_pixels = int(np.count_nonzero(residual_core))
        if residual_pixels < line_height * line_height * 0.75:
            retained.append(enclosing)
            continue
        residual_ink = (gray < 175) & (residual_core > 0)
        ink_fraction = float(np.count_nonzero(residual_ink) / residual_pixels)
        active_rows = int(np.count_nonzero(np.any(residual_ink, axis=1)))
        if (
            ink_fraction >= 0.07
            and active_rows >= max(8, int(round(line_height * 0.55)))
        ):
            removed += 1
            continue
        retained.append(enclosing)
    return retained, removed


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


def _horizontal_line_runs(
    segments: list[geometry_stage.OrientedSegment],
    line_height: int,
) -> list[tuple[float, float, float]]:
    """Merge collinear LSD fragments into stable horizontal rails."""

    # A border can drift by roughly one third of a text line on a skewed scan.
    # Text baselines remain about one full line apart, so this joins the former
    # without collapsing adjacent prose rules.
    position_tolerance = max(4.0, line_height * 0.42)
    gap_tolerance = max(5.0, line_height * 0.65)
    pending = sorted(
        (
            (segment.midpoint[1], segment.start[0], segment.end[0])
            for segment in segments
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    runs: list[tuple[float, float, float]] = []
    while pending:
        seed_y, seed_start, seed_end = pending.pop(0)
        group = [(seed_y, seed_start, seed_end)]
        changed = True
        while changed:
            changed = False
            center = float(np.median([item[0] for item in group]))
            left = min(item[1] for item in group)
            right = max(item[2] for item in group)
            keep: list[tuple[float, float, float]] = []
            for item in pending:
                if (
                    abs(item[0] - center) <= position_tolerance
                    and item[1] <= right + gap_tolerance
                    and item[2] >= left - gap_tolerance
                ):
                    group.append(item)
                    changed = True
                else:
                    keep.append(item)
            pending = keep
        runs.append(
            (
                float(np.median([item[0] for item in group])),
                min(item[1] for item in group),
                max(item[2] for item in group),
            )
        )
    return sorted(runs)


def _interval_union_fraction(
    intervals: list[tuple[float, float]],
    start: float,
    end: float,
    *,
    join_gap: float,
) -> float:
    """Return measured interval coverage without counting overlaps twice."""

    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in intervals
        if min(end, right) > max(start, left)
    )
    if not clipped or end <= start:
        return 0.0
    covered = 0.0
    run_left, run_right = clipped[0]
    for left, right in clipped[1:]:
        if left <= run_right + join_gap:
            run_right = max(run_right, right)
            continue
        covered += run_right - run_left
        run_left, run_right = left, right
    covered += run_right - run_left
    return float(covered / max(1.0, end - start))


def _fragmented_rail_outline_candidates(
    gray: np.ndarray,
    edges: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
    *,
    oriented_segments: tuple[
        list[geometry_stage.OrientedSegment],
        list[geometry_stage.OrientedSegment],
    ]
    | None = None,
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover blank rectangles whose horizontal rails are scan-fragmented.

    Two aligned, independently measured vertical walls define the candidate
    height.  Horizontal LSD fragments must collectively cover both the top and
    bottom rails, and the enclosed body must remain blank.  This route never
    infers a wall from page borders and is confined to the document body.
    """

    height, width = gray.shape
    horizontal, vertical = oriented_segments or layered_stage._oriented_line_segments(
        gray
    )
    vertical_walls: list[tuple[float, float, float]] = []
    for segment in vertical:
        x = float(segment.midpoint[0])
        top = float(min(segment.start[1], segment.end[1]))
        bottom = float(max(segment.start[1], segment.end[1]))
        if (
            bottom - top >= line_height * 0.75
            # Fragmented-rail recovery is a body-box rescue. Page-frame walls
            # near the scan edge create convincing but spurious rail pairs and
            # are handled separately by the page-furniture diagnostics.
            and width * 0.12 < x < width * 0.965
        ):
            vertical_walls.append((x, top, bottom))

    candidates: list[BoxComponent] = []
    considered = 0
    rejected_rails = 0
    rejected_text = 0
    for left_index, first_wall in enumerate(vertical_walls):
        for second_wall in vertical_walls[left_index + 1 :]:
            left, right = sorted((first_wall, second_wall))
            x1, x2 = left[0], right[0]
            box_width = x2 - x1
            if not line_height * 4.0 <= box_width <= width * 0.90:
                continue
            if (
                abs(left[1] - right[1]) > line_height * 0.45
                or abs(left[2] - right[2]) > line_height * 0.45
            ):
                continue
            y1 = (left[1] + right[1]) / 2.0
            y2 = (left[2] + right[2]) / 2.0
            box_height = y2 - y1
            if not line_height * 0.75 <= box_height <= line_height * 6.0:
                continue
            if y1 < height * 0.07 or y2 > height * 0.93:
                continue
            considered += 1

            top_intervals: list[tuple[float, float]] = []
            bottom_intervals: list[tuple[float, float]] = []
            for segment in horizontal:
                position = float(segment.midpoint[1])
                interval = (
                    float(min(segment.start[0], segment.end[0])),
                    float(max(segment.start[0], segment.end[0])),
                )
                if abs(position - y1) <= line_height * 0.42:
                    top_intervals.append(interval)
                if abs(position - y2) <= line_height * 0.42:
                    bottom_intervals.append(interval)
            join_gap = max(3.0, line_height * 0.16)
            top_coverage = _interval_union_fraction(
                top_intervals, x1, x2, join_gap=join_gap
            )
            bottom_coverage = _interval_union_fraction(
                bottom_intervals, x1, x2, join_gap=join_gap
            )
            if (
                min(top_coverage, bottom_coverage) < 0.40
                or top_coverage + bottom_coverage < 1.20
            ):
                rejected_rails += 1
                continue

            box = (
                int(round(x1)),
                int(round(y1)),
                int(round(x2)) + 1,
                int(round(y2)) + 1,
            )
            quality = geometry_stage._robust_interior_quality(
                gray, edges, box, line_height
            )
            ink, active_rows, longest = validated_stage._visible_ink_stats(
                gray, box, line_height
            )
            if (
                quality is None
                or ink > 0.035
                or active_rows > 0.14
                or longest > max(5, int(round(line_height * 0.50)))
            ):
                rejected_text += 1
                continue
            candidate = BoxComponent(
                box=box,
                polygon=geometry_stage._rect_polygon(box),
                source=(
                    "fragmented_rail_outline:"
                    f"top={top_coverage:.2f}:bottom={bottom_coverage:.2f}"
                ),
                score=float(7.2 + top_coverage + bottom_coverage),
            )
            if _existing_union_coverage(candidate, existing + candidates) >= 0.65:
                continue
            if not validated_stage._is_novel(
                candidate, existing + candidates, gray.shape
            ):
                continue
            candidates.append(candidate)

    candidates = geometry_stage._dedupe_components(candidates, gray.shape)
    debug = np.zeros_like(gray)
    for candidate in candidates:
        cv2.polylines(
            debug,
            [np.asarray(candidate.polygon, dtype=np.int32)],
            True,
            255,
            1,
            cv2.LINE_8,
        )
    return candidates, debug, {
        "fragmented_rail_pair_count": considered,
        "fragmented_rail_component_count": len(candidates),
        "fragmented_rail_rejected_support_count": rejected_rails,
        "fragmented_rail_rejected_text_count": rejected_text,
    }


def _occluded_step_outline_candidates(
    gray: np.ndarray,
    edges: np.ndarray,
    line_height: int,
    existing: list[BoxComponent],
    lsd_candidates: list[BoxComponent],
    horizontal_segments: list[geometry_stage.OrientedSegment] | None = None,
) -> tuple[list[BoxComponent], np.ndarray, dict[str, int]]:
    """Recover a stepped blank mask whose long side is obscured by prose.

    A partially released mask can leave two blank slabs sharing one observed
    side while release boilerplate or newly visible text removes the opposite
    corner. Recovery requires a validated three-sided seed, a longer aligned
    outer rail, a divider rail, blank retained slabs, and sustained visible ink
    only in the excluded notch. The notch is never filled.
    """

    height, width = gray.shape
    horizontal = horizontal_segments
    if horizontal is None:
        horizontal, _ = layered_stage._oriented_line_segments(gray)
    rails = _horizontal_line_runs(horizontal, line_height)
    accepted: list[BoxComponent] = []
    debug = np.zeros_like(gray)
    considered = 0
    rejected_text = 0
    rejected_shape = 0

    for seed in lsd_candidates:
        support_match = re.search(
            r"h=([0-9.]+)/([0-9.]+):v=([0-9.]+)/([0-9.]+):corners=(\d+)",
            seed.source,
        )
        if not support_match:
            continue
        top_support, bottom_support, left_support, right_support = (
            float(value) for value in support_match.groups()[:4]
        )
        corner_count = int(support_match.group(5))
        x1, y1, x2, y2 = seed.box
        box_width, box_height = x2 - x1, y2 - y1
        missing_right = left_support >= 0.85 and right_support <= 0.12
        missing_left = right_support >= 0.85 and left_support <= 0.12
        if (
            min(top_support, bottom_support) < 0.80
            or not (missing_left ^ missing_right)
            or corner_count < 2
            or box_width < line_height * 7.0
            or box_height < line_height * 1.80
            or box_height > line_height * 4.5
        ):
            continue
        considered += 1

        near_top = [
            rail
            for rail in rails
            if abs(rail[0] - y1) <= line_height * 0.40
        ]
        near_bottom = [
            rail
            for rail in rails
            if abs(rail[0] - (y2 - 1)) <= line_height * 0.40
        ]

        def extended_rail(
            candidates: list[tuple[float, float, float]],
        ) -> tuple[float, float, float] | None:
            if missing_right:
                options = [
                    rail
                    for rail in candidates
                    if rail[1] <= x1 + line_height
                    and rail[2] >= x2 + line_height * 2.0
                ]
                return max(options, key=lambda item: item[2], default=None)
            options = [
                rail
                for rail in candidates
                if rail[2] >= x2 - line_height
                and rail[1] <= x1 - line_height * 2.0
            ]
            return min(options, key=lambda item: item[1], default=None)

        top_outer = extended_rail(near_top)
        bottom_outer = extended_rail(near_bottom)
        if bool(top_outer) == bool(bottom_outer):
            rejected_shape += 1
            continue
        outer = top_outer or bottom_outer
        assert outer is not None
        outer_on_top = top_outer is not None
        long_left = int(round(outer[1])) if missing_left else x1
        long_right = int(round(outer[2])) + 1 if missing_right else x2
        if not (0 <= long_left < long_right <= width):
            rejected_shape += 1
            continue

        divider_options: list[tuple[float, float, float]] = []
        for rail in rails:
            rail_y, rail_left, rail_right = rail
            if not y1 + line_height * 0.70 <= rail_y <= y2 - line_height * 0.35:
                continue
            if missing_right:
                aligned = (
                    rail_left <= x2 + line_height * 2.0
                    and rail_right >= long_right - line_height * 1.5
                )
            else:
                aligned = (
                    rail_right >= x1 - line_height * 2.0
                    and rail_left <= long_left + line_height * 1.5
                )
            if aligned:
                divider_options.append(rail)
        if not divider_options:
            rejected_shape += 1
            continue
        midpoint = (y1 + y2) / 2.0
        divider = min(divider_options, key=lambda item: abs(item[0] - midpoint))
        split = int(round(divider[0]))
        if min(split - y1, y2 - split) < line_height * 0.55:
            rejected_shape += 1
            continue

        if outer_on_top:
            long_box = (long_left, y1, long_right, split + 1)
            short_box = (x1, split, x2, y2)
            notch = (
                x2 if missing_right else long_left,
                split,
                long_right if missing_right else x1,
                y2,
            )
        else:
            short_box = (x1, y1, x2, split + 1)
            long_box = (long_left, split, long_right, y2)
            notch = (
                x2 if missing_right else long_left,
                y1,
                long_right if missing_right else x1,
                split + 1,
            )
        if notch[2] <= notch[0] or notch[3] <= notch[1]:
            rejected_shape += 1
            continue

        long_quality = geometry_stage._robust_interior_quality(
            gray, edges, long_box, line_height
        )
        short_quality = geometry_stage._robust_interior_quality(
            gray, edges, short_box, line_height
        )
        if long_quality is None or short_quality is None:
            rejected_text += 1
            continue
        long_ink, long_rows, long_run = validated_stage._visible_ink_stats(
            gray, long_box, line_height
        )
        short_ink, short_rows, short_run = validated_stage._visible_ink_stats(
            gray, short_box, line_height
        )
        notch_ink, notch_rows, notch_run = validated_stage._visible_ink_stats(
            gray, notch, line_height
        )
        retained_blank = (
            max(long_ink, short_ink) <= 0.060
            and max(long_rows, short_rows) <= 0.30
            and max(long_run, short_run) <= max(9, int(round(line_height * 0.60)))
        )
        excluded_is_text = (
            notch_ink >= 0.075
            and notch_rows >= 0.30
            and notch_run >= max(8, int(round(line_height * 0.45)))
        )
        if not retained_blank or not excluded_is_text:
            rejected_text += 1
            continue

        pair = [
            BoxComponent(
                box=long_box,
                polygon=geometry_stage._rect_polygon(long_box),
                source=(
                    "occluded_step_outline:long:"
                    f"notch_ink={notch_ink:.3f}:seed={seed.source}"
                ),
                score=float(seed.score) + 0.60,
            ),
            BoxComponent(
                box=short_box,
                polygon=geometry_stage._rect_polygon(short_box),
                source=(
                    "occluded_step_outline:short:"
                    f"notch_ink={notch_ink:.3f}:seed={seed.source}"
                ),
                score=float(seed.score) + 0.55,
            ),
        ]
        if any(
            not validated_stage._is_novel(
                component, existing + accepted, gray.shape
            )
            for component in pair
        ):
            continue
        accepted.extend(pair)
        for component in pair:
            cv2.polylines(
                debug,
                [np.asarray(component.polygon, dtype=np.int32)],
                True,
                255,
                1,
                cv2.LINE_8,
            )

    return accepted, debug, {
        "occluded_step_seed_count": considered,
        "occluded_step_component_count": len(accepted),
        "occluded_step_rejected_shape_count": rejected_shape,
        "occluded_step_rejected_text_count": rejected_text,
    }


def _suppress_marginal_solid_specks(
    components: list[BoxComponent],
    content_bounds: tuple[int, int],
    line_height: int,
    image_shape: tuple[int, int] | None = None,
) -> tuple[list[BoxComponent], list[BoxComponent], bool]:
    """Drop tiny solid scan marks outside prose or at an extreme page edge."""

    solid = [
        component
        for component in components
        if component.source.startswith(("solid_fill", "dense_blackout"))
    ]
    dense_regime = any(
        component.box[2] - component.box[0] >= line_height * 5.0
        and component.box[3] - component.box[1] >= line_height * 1.4
        and geometry_stage._area(component.box) >= line_height * line_height * 8.0
        for component in solid
    )
    content_left, content_right = content_bounds
    body_margin = max(5, int(round(line_height * 0.75)))
    retained: list[BoxComponent] = []
    suppressed: list[BoxComponent] = []
    page_height = image_shape[0] if image_shape is not None else None
    for component in components:
        x1, y1, x2, y2 = component.box
        box_width, box_height = x2 - x1, y2 - y1
        tiny_solid = (
            component.source.startswith(("solid_fill", "dense_blackout"))
            and box_width <= line_height * 0.95
            and box_height <= line_height * 0.70
            and geometry_stage._area(component.box) <= line_height * line_height * 0.70
        )
        outside_body = (
            x2 < content_left - body_margin or x1 > content_right + body_margin
        )
        polygon_area = abs(
            float(
                cv2.contourArea(
                    np.asarray(component.polygon, dtype=np.int32).reshape(
                        (-1, 1, 2)
                    )
                )
            )
        )
        fill_ratio = polygon_area / max(1, box_width * box_height)
        extreme_page_band = bool(
            page_height is not None
            and (y2 <= page_height * 0.055 or y1 >= page_height * 0.945)
        )
        irregular_edge_cap = (
            component.source.startswith(("solid_fill", "dense_blackout"))
            and extreme_page_band
            and box_width <= line_height * 1.80
            and box_height <= line_height * 1.15
            and geometry_stage._area(component.box)
            <= line_height * line_height * 2.10
            and len(component.polygon) >= 6
            and fill_ratio <= 0.88
        )
        solid_source = component.source.startswith(
            ("solid_fill", "dense_blackout")
        )
        area_in_line_squares = geometry_stage._area(component.box) / max(
            1.0, float(line_height * line_height)
        )
        material_solid = (
            box_width >= line_height * 2.5
            and area_in_line_squares >= 3.0
        ) or (
            box_height >= line_height * 1.4
            and area_in_line_squares >= 3.0
        )
        nonmaterial_solid = solid_source and not material_solid
        if irregular_edge_cap or nonmaterial_solid or (
            tiny_solid and outside_body and not dense_regime
        ):
            suppressed.append(component)
            continue
        retained.append(component)
    return retained, suppressed, dense_regime


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
    base_components, layered_blackout_envelope_count = (
        _replace_layered_blackout_union(
            work_gray.shape,
            base_components,
            blackout,
        )
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
    nested_outlines, nested_outline_mask, nested_outline_diagnostics = (
        _nested_measured_outline_candidates(
            work_gray,
            line_height,
            existing,
            lsd_candidates,
        )
    )
    existing, partitioned_envelope_count = _remove_partitioned_envelopes(
        existing,
        nested_outlines,
    )
    existing, text_crossing_step_envelope_count = (
        _remove_text_crossing_step_envelopes(
            work_gray,
            line_height,
            existing,
            nested_outlines,
        )
    )
    for component in nested_outlines:
        if validated_stage._is_novel(component, existing, work_gray.shape):
            existing.append(component)
    oriented_segments = layered_stage._oriented_line_segments(work_gray)
    occluded_steps, occluded_step_mask, occluded_step_diagnostics = (
        _occluded_step_outline_candidates(
            work_gray,
            artifacts["edges"],
            line_height,
            existing,
            lsd_candidates,
            horizontal_segments=oriented_segments[0],
        )
    )
    for component in occluded_steps:
        if validated_stage._is_novel(component, existing, work_gray.shape):
            existing.append(component)
    fragmented_rails, fragmented_rail_mask, fragmented_rail_diagnostics = (
        _fragmented_rail_outline_candidates(
            work_gray,
            artifacts["edges"],
            line_height,
            existing,
            oriented_segments=oriented_segments,
        )
    )
    for component in fragmented_rails:
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
    existing, suppressed_solid_artifacts, dense_ink_regime = (
        _suppress_marginal_solid_specks(
            existing,
            content_bounds,
            line_height,
            work_gray.shape,
        )
    )
    if suppressed_solid_artifacts:
        final_regions, content_bounds, layout_roles = (
            validated_stage._group_and_order(existing, work_gray, line_mask)
        )
    suppressed_solid_mask = np.zeros_like(work_gray)
    for component in suppressed_solid_artifacts:
        cv2.fillPoly(
            suppressed_solid_mask,
            [np.asarray(component.polygon, dtype=np.int32)],
            255,
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
            "blackout_replaced_envelope_count": layered_blackout_envelope_count,
            "layered_outline_rescue_count": len(layered),
            "supported_outline_count": len(promoted),
            "occluded_step_outline_component_count": len(occluded_steps),
            "fragmented_rail_outline_component_count": len(fragmented_rails),
            "marginal_solid_suppression_count": len(suppressed_solid_artifacts),
            "dense_ink_redaction_regime": dense_ink_regime,
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
    diagnostics.update(nested_outline_diagnostics)
    diagnostics["nested_outline_removed_envelope_count"] = (
        partitioned_envelope_count
    )
    diagnostics["nested_outline_removed_text_step_count"] = (
        text_crossing_step_envelope_count
    )
    diagnostics.update(occluded_step_diagnostics)
    diagnostics.update(fragmented_rail_diagnostics)
    diagnostics.update(line_cycle_diagnostics)
    artifacts["shipping_blackout_partitions"] = blackout_mask
    artifacts["shipping_layered_outlines"] = layered_mask
    artifacts["complex_enclosed_outlines"] = complex_mask
    artifacts["line_segment_cycles"] = line_cycle_mask
    artifacts["three_side_page_outlines"] = page_open_mask
    artifacts["supported_outlines"] = promoted_mask
    artifacts["nested_measured_outlines"] = nested_outline_mask
    artifacts["occluded_step_outlines"] = occluded_step_mask
    artifacts["fragmented_rail_outlines"] = fragmented_rail_mask
    artifacts["suppressed_solid_artifacts"] = suppressed_solid_mask
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
