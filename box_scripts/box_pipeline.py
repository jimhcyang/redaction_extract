"""Text-safe geometry refinements over the production detector."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from . import _geometry_detection as geometry_stage
from . import _layered_detection as layered_stage
from . import _production_detection as production_stage
from . import _validated_detection as validated_stage


PDFPair = production_stage.PDFPair
PairCollectionStats = production_stage.PairCollectionStats
InputRecord = production_stage.InputRecord
BoxComponent = production_stage.BoxComponent
RedactionRegion = production_stage.RedactionRegion
collect_pdf_pairs_with_stats = production_stage.collect_pdf_pairs_with_stats
render_pdf_to_images = production_stage.render_pdf_to_images
iter_input_records = production_stage.iter_input_records

DETECTOR_POLICY = (
    "Single-page, answer-blind geometry. Release 3.2 preserves the "
    "production detector and its page-measured safeguards: visible prose "
    "cannot serve as a synthetic step interior; an incomplete blank rectangle "
    "may expand only to its seed-connected closed white-space contour; and a "
    "compact, independently bounded blank continuation is not discarded inside "
    "a larger text-crossing envelope. OCR, paired releases, filenames, answers, "
    "and benchmark labels are unavailable to detection."
)


def _component_mask(
    component: BoxComponent, image_shape: tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(component.polygon, dtype=np.int32)], 255)
    return mask


def _component_intersection_fraction(
    candidate: BoxComponent,
    others: Iterable[BoxComponent],
    image_shape: tuple[int, int],
) -> float:
    candidate_mask = _component_mask(candidate, image_shape)
    candidate_pixels = int(np.count_nonzero(candidate_mask))
    if not candidate_pixels:
        return 0.0
    other_mask = np.zeros(image_shape, dtype=np.uint8)
    for component in others:
        cv2.fillPoly(
            other_mask,
            [np.asarray(component.polygon, dtype=np.int32)],
            255,
        )
    return float(
        np.count_nonzero((candidate_mask > 0) & (other_mask > 0))
        / candidate_pixels
    )


def _step_slabs(component: BoxComponent) -> list[tuple[int, int, int, int]]:
    """Decompose an orthogonal step outline into horizontal filled slabs."""

    x1, y1, x2, y2 = component.box
    if x2 <= x1 or y2 <= y1:
        return []
    local = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    polygon = np.asarray(
        [(x - x1, y - y1) for x, y in component.polygon], dtype=np.int32
    )
    cv2.fillPoly(local, [polygon], 255)

    row_spans: list[tuple[int, int] | None] = []
    for row in local:
        occupied = np.flatnonzero(row)
        row_spans.append(
            (int(occupied[0]), int(occupied[-1]) + 1)
            if occupied.size
            else None
        )

    slabs: list[tuple[int, int, int, int]] = []
    start = 0
    current = row_spans[0] if row_spans else None
    for index in range(1, len(row_spans) + 1):
        following = row_spans[index] if index < len(row_spans) else None
        same = bool(
            current is not None
            and following is not None
            and abs(current[0] - following[0]) <= 2
            and abs(current[1] - following[1]) <= 2
        )
        if same:
            continue
        if current is not None:
            slabs.append(
                (x1 + current[0], y1 + start, x1 + current[1], y1 + index)
            )
        start = index
        current = following

    # Polygon rasterization creates one-pixel transition rows at shared edges.
    # Absorb them into the adjacent material slab instead of emitting slivers.
    minimum_height = 3
    cleaned: list[tuple[int, int, int, int]] = []
    for slab in slabs:
        if slab[3] - slab[1] >= minimum_height:
            cleaned.append(slab)
            continue
        if cleaned:
            prior = cleaned[-1]
            cleaned[-1] = (prior[0], prior[1], prior[2], slab[3])
    return cleaned


def _text_stats_in_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    line_height: int,
) -> tuple[float, float, int]:
    return validated_stage._visible_ink_stats(gray, box, line_height)


def _split_text_crossing_steps(
    gray: np.ndarray,
    components: list[BoxComponent],
    line_height: int,
) -> tuple[list[BoxComponent], int, int]:
    """Undo a synthetic step completion when its added slab crosses prose.

    The upstream completion is useful for genuinely blank concave masks. A
    false completion has the same outline topology but one horizontal slab
    contains a sustained text row. In that case, retain only independently
    blank slabs; no filename, page pairing, OCR, or benchmark label is used.
    """

    output: list[BoxComponent] = []
    refined_components = 0
    rejected_text_slabs = 0
    for component in components:
        if not component.source.startswith("shared_edge_step_completion:"):
            output.append(component)
            continue
        slabs = _step_slabs(component)
        if len(slabs) < 2:
            output.append(component)
            continue

        measurements = [
            (*_text_stats_in_box(gray, slab, line_height), slab)
            for slab in slabs
        ]
        text_crossing = [
            row
            for row in measurements
            if row[3][3] - row[3][1]
            >= max(6, int(round(line_height * 0.65)))
            and (
                row[0] > 0.080
                or row[1] > 0.28
                or row[2] > max(12, int(round(line_height * 0.70)))
            )
        ]
        blank = [
            row
            for row in measurements
            if row[0] <= 0.060
            and row[1] <= 0.20
            and row[2] <= max(10, int(round(line_height * 0.55)))
        ]
        if not text_crossing or not blank:
            output.append(component)
            continue

        replacements: list[BoxComponent] = []
        for ink, active_rows, longest, box in blank:
            replacement = BoxComponent(
                box=box,
                polygon=geometry_stage._rect_polygon(box),
                source=(
                    "text_safe_step_split:"
                    f"ink={ink:.3f}:rows={active_rows:.3f}:run={longest}:"
                    f"{component.source}"
                ),
                score=float(component.score),
            )
            other_components = [
                prior for prior in components if prior is not component
            ] + replacements
            if _component_intersection_fraction(
                replacement, other_components, gray.shape
            ) >= 0.82:
                continue
            replacements.append(replacement)
        if not replacements:
            output.append(component)
            continue
        output.extend(replacements)
        refined_components += 1
        rejected_text_slabs += len(text_crossing)
    return output, refined_components, rejected_text_slabs


def _seeded_blank_outline(
    gray: np.ndarray,
    component: BoxComponent,
    line_height: int,
) -> BoxComponent | None:
    """Expand an incomplete rectangular shortcut to its closed blank contour."""

    if not component.source.startswith("supported_outline:"):
        return None
    corner_match = re.search(r"corners=(\d+)", component.source)
    if not corner_match or int(corner_match.group(1)) >= 4:
        return None

    x1, y1, x2, y2 = component.box
    height, width = gray.shape
    if x2 <= x1 or y2 <= y1:
        return None
    pad = max(3, int(round(line_height * 0.18)))
    # A light threshold treats both ruled borders and visible glyphs as walls.
    # Unlike a global whitespace contour, selection requires the result to
    # contain most of the already validated blank rectangle. Choosing by
    # overlap, rather than one center pixel, is robust to a clipped corner that
    # exposes a few interior pixels to the page background.
    free = (gray >= 235).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        free, connectivity=8
    )
    inset_labels = labels[
        max(0, y1 + pad) : min(height, y2 - pad),
        max(0, x1 + pad) : min(width, x2 - pad),
    ]
    if inset_labels.size == 0:
        return None
    frequencies = np.bincount(inset_labels.ravel(), minlength=count)
    frequencies[0] = 0
    label = int(np.argmax(frequencies))
    if label <= 0 or label >= count:
        return None
    bx, by, box_width, box_height, area = map(int, stats[label])
    if (
        bx <= 0
        or by <= 0
        or bx + box_width >= width
        or by + box_height >= height
    ):
        return None

    candidate_area = max(1, (x2 - x1) * (y2 - y1))
    if not candidate_area * 1.08 <= area <= candidate_area * 2.75:
        return None
    expansion_x = max(x1 - bx, bx + box_width - x2, 0)
    expansion_y = max(y1 - by, by + box_height - y2, 0)
    if (
        expansion_x > max(line_height * 2.5, (x2 - x1) * 0.20)
        or expansion_y > max(line_height * 2.5, (y2 - y1) * 0.65)
    ):
        return None

    label_mask = np.where(labels == label, 255, 0).astype(np.uint8)
    original_mask = _component_mask(component, gray.shape)
    original_pixels = int(np.count_nonzero(original_mask))
    retained = int(
        np.count_nonzero((label_mask > 0) & (original_mask > 0))
    ) / max(1, original_pixels)
    if retained < 0.72:
        return None

    contours, _ = cv2.findContours(
        label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(2.0, line_height * 0.08)
    approximation = cv2.approxPolyDP(contour, epsilon, True)
    polygon = tuple(
        (int(point[0][0]), int(point[0][1])) for point in approximation
    )
    if not 5 <= len(polygon) <= 16:
        return None
    polygon_box = geometry_stage._polygon_box(polygon)
    polygon_area = abs(
        float(
            cv2.contourArea(
                np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            )
        )
    )
    fill = polygon_area / max(
        1.0,
        float(
            (polygon_box[2] - polygon_box[0])
            * (polygon_box[3] - polygon_box[1])
        ),
    )
    if not 0.42 <= fill <= 0.92:
        return None

    barrier = gray < 235
    perimeter = cv2.subtract(
        cv2.dilate(label_mask, np.ones((3, 3), np.uint8)),
        label_mask,
    )
    perimeter_pixels = int(np.count_nonzero(perimeter))
    support = float(
        np.count_nonzero((perimeter > 0) & barrier)
        / max(1, perimeter_pixels)
    )
    if support < 0.70:
        return None
    return BoxComponent(
        box=polygon_box,
        polygon=polygon,
        source=(
            "seeded_blank_outline:"
            f"support={support:.2f}:fill={fill:.2f}:retained={retained:.2f}:"
            f"{component.source}"
        ),
        score=float(component.score) + 0.02,
    )


def _refine_incomplete_outlines(
    gray: np.ndarray,
    components: list[BoxComponent],
    line_height: int,
) -> tuple[list[BoxComponent], int]:
    output: list[BoxComponent] = []
    refinement_count = 0
    for component in components:
        refined = _seeded_blank_outline(gray, component, line_height)
        if refined is None:
            output.append(component)
            continue
        others = [prior for prior in components if prior is not component]
        if _component_intersection_fraction(
            refined, others, gray.shape
        ) >= 0.35:
            output.append(component)
            continue
        output.append(refined)
        refinement_count += 1
    return output, refinement_count


def _right_column_bridge_components(
    components: list[BoxComponent], line_height: int
) -> list[BoxComponent]:
    """Find tall side boxes that only bridge horizontal reading-flow boxes."""

    tolerance = max(3, int(round(line_height * 0.16)))
    bridges: list[BoxComponent] = []
    for candidate in components:
        x1, y1, x2, y2 = candidate.box
        width = x2 - x1
        height = y2 - y1
        if height < max(line_height * 4, width * 1.35):
            continue
        contacts: list[BoxComponent] = []
        for other in components:
            if other is candidate:
                continue
            ox1, oy1, ox2, oy2 = other.box
            vertical_overlap = max(0, min(y2, oy2) - max(y1, oy1))
            if vertical_overlap < max(3, min(line_height, oy2 - oy1) * 0.65):
                continue
            # A true side-column bridge meets only the candidate's left edge;
            # it does not materially cover the horizontal component itself.
            if abs(ox2 - x1) <= tolerance and ox1 < x1:
                contacts.append(other)
        if len(contacts) < 2:
            continue
        if not any(
            item.source.startswith("text_safe_step_split:")
            for item in contacts
        ):
            continue
        bridges.append(candidate)
    return bridges


def _merge_step_adjacent_regions(
    regions: list[RedactionRegion], line_height: int
) -> tuple[list[RedactionRegion], int]:
    """Reconnect blank slabs separated only by a rasterization seam."""

    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seam = max(2, int(round(line_height * 0.10)))
    alignment = max(4, int(round(line_height * 0.25)))
    merge_count = 0
    for left_index, left_region in enumerate(regions):
        for right_index in range(left_index + 1, len(regions)):
            right_region = regions[right_index]
            should_merge = False
            for left in left_region.components:
                for right in right_region.components:
                    if not (
                        left.source.startswith("text_safe_step_split:")
                        or right.source.startswith("text_safe_step_split:")
                    ):
                        continue
                    first, second = sorted(
                        (left, right), key=lambda item: item.box[1]
                    )
                    fx1, fy1, fx2, fy2 = first.box
                    sx1, sy1, sx2, sy2 = second.box
                    gap = sy1 - fy2
                    overlap = max(0, min(fx2, sx2) - max(fx1, sx1))
                    minimum_width = max(1, min(fx2 - fx1, sx2 - sx1))
                    edge_aligned = (
                        abs(fx1 - sx1) <= alignment
                        or abs(fx2 - sx2) <= alignment
                    )
                    if (
                        -seam <= gap <= seam
                        and overlap / minimum_width >= 0.80
                        and edge_aligned
                    ):
                        should_merge = True
                        break
                if should_merge:
                    break
            if should_merge and find(left_index) != find(right_index):
                union(left_index, right_index)
                merge_count += 1

    grouped: dict[int, list[BoxComponent]] = {}
    for index, region in enumerate(regions):
        grouped.setdefault(find(index), []).extend(region.components)
    merged = [
        RedactionRegion(
            components=tuple(
                sorted(values, key=lambda item: (item.box[1], item.box[0]))
            )
        )
        for values in grouped.values()
    ]
    merged.sort(
        key=lambda region: (
            region.components[0].box[1], region.components[0].box[0]
        )
    )
    return merged, merge_count


def _group_with_text_safe_layout(
    components: list[BoxComponent],
    gray: np.ndarray,
    line_mask: np.ndarray,
    line_height: int,
) -> tuple[
    list[RedactionRegion], tuple[int, int], dict[int, str], int, int
]:
    """Keep side-column contacts from becoming transitive reading bridges."""

    bridges = _right_column_bridge_components(components, line_height)
    bridge_ids = {id(component) for component in bridges}
    body = [component for component in components if id(component) not in bridge_ids]
    body_regions, content_bounds, _ = validated_stage._group_and_order(
        body, gray, line_mask
    )
    body_regions, seam_merges = _merge_step_adjacent_regions(
        body_regions, line_height
    )
    side_regions = [RedactionRegion(components=(component,)) for component in bridges]
    regions = body_regions + side_regions
    content_left, content_right = content_bounds
    ordered: list[tuple[int, int, int, RedactionRegion, str]] = []
    for region in regions:
        component_roles = [
            validated_stage._component_layout_role(
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
    return (
        [item[3] for item in ordered],
        content_bounds,
        {index: item[4] for index, item in enumerate(ordered, start=1)},
        len(bridges),
        seam_merges,
    )


def detect_redaction_regions_with_artifacts(
    gray: np.ndarray,
) -> tuple[list[RedactionRegion], dict[str, np.ndarray], dict[str, Any]]:
    regions, artifacts, diagnostics = production_stage.detect_redaction_regions_with_artifacts(
        gray
    )
    work_gray = artifacts["deskewed_source"]
    line_mask = artifacts["lines_mask"]
    line_height = layered_stage._text_line_height(work_gray, line_mask)
    components = [
        component for region in regions for component in region.components
    ]
    components, step_refinements, text_slabs = _split_text_crossing_steps(
        work_gray, components, line_height
    )
    components, outline_refinements = _refine_incomplete_outlines(
        work_gray, components, line_height
    )
    if not step_refinements and not outline_refinements:
        diagnostics.update(
            {
                "detector_release": "3.2.0",
                "text_safe_step_refinement_count": 0,
                "rejected_text_slab_count": 0,
                "seeded_blank_outline_refinement_count": 0,
                "side_column_bridge_split_count": 0,
                "step_seam_merge_count": 0,
                "detector_policy": DETECTOR_POLICY,
            }
        )
        return regions, artifacts, diagnostics
    components = geometry_stage._dedupe_components(components, work_gray.shape)
    (
        final_regions,
        content_bounds,
        layout_roles,
        side_bridge_splits,
        seam_merges,
    ) = _group_with_text_safe_layout(
        components, work_gray, line_mask, line_height
    )
    diagnostics.update(
        {
            "detector_release": "3.2.0",
            "text_safe_step_refinement_count": step_refinements,
            "rejected_text_slab_count": text_slabs,
            "seeded_blank_outline_refinement_count": outline_refinements,
            "side_column_bridge_split_count": side_bridge_splits,
            "step_seam_merge_count": seam_merges,
            "estimated_content_bounds_x": list(content_bounds),
            "region_layout_roles": layout_roles,
            "component_count_before_grouping": len(components),
            "region_count": len(final_regions),
            "detector_policy": DETECTOR_POLICY,
        }
    )
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
    detector_version: str = "3.2.0",
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
        detector_version="3.2.0",
    )
