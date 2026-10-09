from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from box_scripts import _geometry_detection as geometry_stage
from box_scripts import _production_detection as production_stage
from box_scripts import _validated_detection as validated_stage
from box_scripts import box_pipeline
from box_scripts.box_pipeline import detect_redaction_regions_with_artifacts
from box_scripts.v4_audit import (
    FULL_COVERAGE_THRESHOLD,
    _assignment_status_from_values,
    _boxes_overlap,
    _draw_label,
)


def test_full_assignment_requires_eighty_percent_coverage_by_default() -> None:
    assert FULL_COVERAGE_THRESHOLD == pytest.approx(0.80)
    assert (
        _assignment_status_from_values("PASS", 0.799999, True)
        == "ASSIGNED_PARTIAL"
    )
    assert _assignment_status_from_values("PASS", 0.80, True) == "ASSIGNED_FULL"


def test_assignment_threshold_is_configurable_without_hiding_other_failures() -> None:
    assert (
        _assignment_status_from_values(
            "PASS", 0.85, True, full_coverage_threshold=0.80
        )
        == "ASSIGNED_FULL"
    )
    assert (
        _assignment_status_from_values("PARTIAL_TEXT_MATCH", 0.20, True)
        == "TARGET_LOCALIZATION_UNCERTAIN"
    )
    assert (
        _assignment_status_from_values("PASS", 1.0, False)
        == "REGISTRATION_FAILED"
    )


def test_overlay_labels_are_compact_and_collision_free() -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    occupied: list[tuple[int, int, int, int]] = []
    bounds = (100, 100, 300, 180)

    first = _draw_label(
        image,
        "R3.1",
        bounds,
        (40, 60, 200),
        line_height=32,
        occupied=occupied,
    )
    second = _draw_label(
        image,
        "R3.2",
        bounds,
        (40, 140, 60),
        line_height=32,
        occupied=occupied,
    )

    assert 20 <= first[3] - first[1] <= 27
    assert 20 <= second[3] - second[1] <= 27
    assert not _boxes_overlap(first, second, gap=0)


def test_overlay_label_anchor_stays_inside_irregular_component() -> None:
    image = np.full((320, 520, 3), 255, dtype=np.uint8)
    contour = np.asarray(
        ((100, 100), (420, 100), (420, 230), (250, 230), (250, 165), (100, 165)),
        dtype=np.int32,
    )
    occupied: list[tuple[int, int, int, int]] = []

    label = _draw_label(
        image,
        "R2.1",
        (100, 100, 421, 231),
        (30, 120, 220),
        line_height=32,
        occupied=occupied,
        contour=contour,
    )

    center = ((label[0] + label[2]) // 2, (label[1] + label[3]) // 2)
    assert cv2.pointPolygonTest(contour, center, False) >= 0


def test_compound_blank_outline_is_a_compact_rectangle_cover() -> None:
    page = np.full((900, 1200), 255, dtype=np.uint8)
    component = production_stage.BoxComponent(
        box=(220, 300, 980, 620),
        polygon=(
            (220, 390),
            (430, 390),
            (430, 300),
            (980, 300),
            (980, 620),
            (220, 620),
        ),
        source="closed_blank_contour:support=0.82:fill=0.86",
        score=10.0,
    )

    parts, count = box_pipeline._decompose_compound_blank_outlines(
        page, [component], 30
    )

    assert count == 1
    assert len(parts) == 2
    assert all(part.source.startswith("compound_blank_partition:") for part in parts)
    assert sum(part.box[1] == 300 for part in parts) == 1
    assert sum(part.box[0] == 220 for part in parts) == 1


def test_fragmented_rails_recover_one_blank_body_box() -> None:
    page = np.full((1000, 900), 255, dtype=np.uint8)
    box = (180, 400, 760, 510)
    cv2.line(page, (180, 400), (180, 509), 0, 3)
    cv2.line(page, (759, 400), (759, 509), 0, 3)
    for left, right in ((180, 330), (350, 510), (540, 760)):
        cv2.line(page, (left, 400), (right, 400), 0, 3)
    for left, right in ((180, 410), (435, 610), (630, 760)):
        cv2.line(page, (left, 509), (right, 509), 0, 3)

    def segment(
        orientation: str,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> geometry_stage.OrientedSegment:
        return geometry_stage.OrientedSegment(
            orientation=orientation,
            start=start,
            end=end,
            length=float(np.hypot(end[0] - start[0], end[1] - start[1])),
            angle_degrees=0.0 if orientation == "h" else 90.0,
        )

    horizontal = [
        segment("h", (left, y), (right, y))
        for y, spans in (
            (400.0, ((180.0, 330.0), (350.0, 510.0), (540.0, 760.0))),
            (509.0, ((180.0, 410.0), (435.0, 610.0), (630.0, 760.0))),
        )
        for left, right in spans
    ]
    vertical = [
        segment("v", (180.0, 400.0), (180.0, 509.0)),
        segment("v", (759.0, 400.0), (759.0, 509.0)),
    ]
    candidates, _, diagnostics = production_stage._fragmented_rail_outline_candidates(
        page,
        cv2.Canny(page, 30, 110),
        30,
        [],
        oriented_segments=(horizontal, vertical),
    )

    assert diagnostics["fragmented_rail_component_count"] == 1
    assert len(candidates) == 1
    assert max(abs(candidates[0].box[index] - box[index]) for index in range(4)) <= 2


def test_grouping_splits_indented_paragraph_and_joins_aligned_stack() -> None:
    page = np.full((900, 1400), 255, dtype=np.uint8)

    def component(box: tuple[int, int, int, int]) -> production_stage.BoxComponent:
        return production_stage.BoxComponent(
            box=box,
            polygon=geometry_stage._rect_polygon(box),
            source="synthetic_outline",
            score=7.0,
        )

    paragraph_end = component((300, 300, 810, 340))
    indented_start = component((390, 334, 1160, 375))
    stack_top = component((300, 500, 800, 540))
    stack_lateral = component((800, 500, 1120, 540))
    stack_bottom = component((304, 543, 660, 575))
    regions = validated_stage._group_components_with_layout(
        [
            paragraph_end,
            indented_start,
            stack_top,
            stack_lateral,
            stack_bottom,
        ],
        page,
        text_mask=np.zeros_like(page),
        content_left=285,
        content_right=1175,
        line_pitch=38,
        line_height=30,
    )

    memberships = [{item.box for item in region.components} for region in regions]
    assert {paragraph_end.box} in memberships
    assert {indented_start.box} in memberships
    assert {stack_top.box, stack_lateral.box, stack_bottom.box} in memberships


def test_text_crossing_envelope_does_not_hide_blank_continuation() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, text in enumerate(
        (
            "INTELLIGENCE SUMMARY",
            "A surrounding line of ordinary visible text",
            "another ordinary sentence supplies page scale",
        )
    ):
        cv2.putText(
            page,
            text,
            (220, 250 + index * 65),
            font,
            1.15,
            0,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(page, (700, 1300), (1440, 1440), 0, 3)
    cv2.rectangle(page, (700, 1440), (1015, 1480), 0, 3)
    cv2.rectangle(page, (1290, 1440), (1440, 1480), 0, 3)
    cv2.putText(
        page,
        "the Kuomintang'",
        (1025, 1474),
        font,
        0.82,
        0,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(page, (620, 1485), (1390, 1525), 0, 3)
    cv2.putText(
        page,
        "is multiplying its propagandists and intelligence agents",
        (700, 1575),
        font,
        0.75,
        0,
        2,
        cv2.LINE_AA,
    )

    regions, _, _ = detect_redaction_regions_with_artifacts(page)
    continuation_regions = [
        region
        for region in regions
        if any(
            component.box[1] >= 1480
            and component.box[0] <= 630
            and component.box[2] >= 1380
            for component in region.components
        )
    ]

    assert len(continuation_regions) == 1
    assert len(continuation_regions[0].components) == 2
    assert min(component.box[0] for component in continuation_regions[0].components) <= 630
    assert max(component.box[0] for component in continuation_regions[0].components) >= 1280
    assert continuation_regions[0].box[3] <= 1535

    preceding_regions = [
        region
        for region in regions
        if any(
            component.box[1] <= 1310
            and component.box[0] >= 690
            and component.box[2] >= 1430
            for component in region.components
        )
    ]
    assert len(preceding_regions) == 1
    assert len(preceding_regions[0].components) == 2


def test_overlapping_four_sided_boxes_survive_refinement_and_text_suppression() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, text in enumerate(
        (
            "CURRENT INTELLIGENCE BULLETIN",
            "was unusually friendly and gave the impression",
            "that Moscow wishes to avoid new controversies",
            "with the Austrian Government.",
        )
    ):
        cv2.putText(
            page,
            text,
            (230, 180 + index * 52),
            font,
            0.82,
            0,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(page, "Comment:", (760, 407), font, 0.8, 0, 2, cv2.LINE_AA)
    cv2.rectangle(page, (965, 340), (1617, 411), 0, 3)
    cv2.rectangle(page, (269, 382), (1617, 427), 0, 3)
    cv2.rectangle(page, (269, 433), (470, 468), 0, 3)
    cv2.putText(
        page,
        "since Soviet tactics in Austria have",
        (480, 463),
        font,
        0.76,
        0,
        2,
        cv2.LINE_AA,
    )

    regions, _, _ = detect_redaction_regions_with_artifacts(page)
    target = [
        region
        for region in regions
        if region.box[0] <= 275
        and region.box[1] <= 350
        and region.box[2] >= 1610
        and region.box[3] >= 460
    ]
    assert len(target) == 1
    assert len(target[0].components) == 3


def test_occluded_step_recovery_keeps_blank_slabs_and_excludes_text_notch() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(page, (730, 1080), (1500, 1080), 0, 3)
    cv2.line(page, (730, 1080), (730, 1180), 0, 3)
    cv2.line(page, (1100, 1140), (1500, 1140), 0, 3)
    cv2.line(page, (730, 1180), (1100, 1180), 0, 3)
    cv2.putText(
        page,
        "3.3(h)(2)",
        (1450, 1105),
        font,
        0.65,
        0,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        page,
        "revealed words continue here",
        (1120, 1172),
        font,
        0.65,
        0,
        2,
        cv2.LINE_AA,
    )
    seed_box = (730, 1080, 1101, 1181)
    seed = production_stage.BoxComponent(
        box=seed_box,
        polygon=geometry_stage._rect_polygon(seed_box),
        source="lsd_geometry_rescue:h=0.98/0.95:v=0.99/0.00:corners=2",
        score=5.2,
    )

    components, _, diagnostics = (
        production_stage._occluded_step_outline_candidates(
            page,
            cv2.Canny(page, 30, 110),
            31,
            [],
            [seed],
        )
    )

    assert diagnostics["occluded_step_component_count"] == 2
    assert len(components) == 2
    long_component = max(components, key=lambda item: item.box[2])
    short_component = min(components, key=lambda item: item.box[2])
    assert long_component.box[2] >= 1495
    assert short_component.box[2] <= 1110
    assert long_component.box[3] <= short_component.box[1] + 2


def test_tiny_solid_margin_specks_require_a_dense_page_regime() -> None:
    ordinary_box = (700, 900, 1400, 980)
    speck_box = (55, 650, 75, 662)
    components = [
        production_stage.BoxComponent(
            box=ordinary_box,
            polygon=geometry_stage._rect_polygon(ordinary_box),
            source="rectilinear_zone_1:observed=4:virtual=0",
            score=12.0,
        ),
        production_stage.BoxComponent(
            box=speck_box,
            polygon=geometry_stage._rect_polygon(speck_box),
            source="solid_fill_shape_preserved",
            score=5.5,
        ),
    ]

    retained, suppressed, dense_regime = (
        production_stage._suppress_marginal_solid_specks(
            components,
            (240, 1450),
            31,
        )
    )

    assert not dense_regime
    assert suppressed == [components[1]]
    assert retained == [components[0]]


def test_medium_solid_box_does_not_enable_page_wide_spill_logic() -> None:
    medium_box = (700, 900, 810, 930)
    speck_box = (55, 650, 75, 662)
    components = [
        production_stage.BoxComponent(
            box=medium_box,
            polygon=geometry_stage._rect_polygon(medium_box),
            source="solid_fill_shape_preserved",
            score=7.0,
        ),
        production_stage.BoxComponent(
            box=speck_box,
            polygon=geometry_stage._rect_polygon(speck_box),
            source="solid_fill_shape_preserved",
            score=5.5,
        ),
    ]

    retained, suppressed, dense_regime = (
        production_stage._suppress_marginal_solid_specks(
            components,
            (240, 1450),
            31,
        )
    )

    assert not dense_regime
    assert suppressed == [components[1]]
    assert retained == [components[0]]


def test_nested_measured_outline_requires_independent_sides() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    enclosing_box = (200, 500, 1500, 1100)
    enclosing = production_stage.BoxComponent(
        box=enclosing_box,
        polygon=geometry_stage._rect_polygon(enclosing_box),
        source="rectilinear_zone_1:observed=4:virtual=0",
        score=8.0,
    )
    independent_box = (600, 700, 1050, 750)
    independent = production_stage.BoxComponent(
        box=independent_box,
        polygon=geometry_stage._rect_polygon(independent_box),
        source="lsd_geometry_rescue:h=1.00/0.99:v=0.98/0.99:corners=4",
        score=7.3,
    )
    subdivision_box = (200, 500, 700, 570)
    subdivision = production_stage.BoxComponent(
        box=subdivision_box,
        polygon=geometry_stage._rect_polygon(subdivision_box),
        source="lsd_geometry_rescue:h=1.00/1.00:v=1.00/1.00:corners=4",
        score=7.4,
    )

    accepted, _, diagnostics = (
        production_stage._nested_measured_outline_candidates(
            page,
            32,
            [enclosing],
            [independent, subdivision],
        )
    )

    assert [component.box for component in accepted] == [independent_box]
    assert diagnostics["nested_outline_component_count"] == 1
    assert diagnostics["nested_outline_rejected_shared_sides_count"] == 1


def test_nested_measured_outline_merges_only_overlapping_seam_halves() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    enclosing_box = (200, 500, 1500, 1100)
    enclosing = production_stage.BoxComponent(
        box=enclosing_box,
        polygon=geometry_stage._rect_polygon(enclosing_box),
        source="rectilinear_zone_1:observed=4:virtual=0",
        score=8.0,
    )

    def candidate(box: tuple[int, int, int, int]) -> production_stage.BoxComponent:
        return production_stage.BoxComponent(
            box=box,
            polygon=geometry_stage._rect_polygon(box),
            source="lsd_geometry_rescue:h=1.00/0.99:v=0.98/0.99:corners=4",
            score=7.3,
        )

    accepted, _, diagnostics = (
        production_stage._nested_measured_outline_candidates(
            page,
            32,
            [enclosing],
            [candidate((500, 700, 1100, 730)), candidate((500, 727, 1100, 760))],
        )
    )

    assert [component.box for component in accepted] == [(500, 700, 1100, 760)]
    assert diagnostics["nested_outline_merged_member_count"] == 1


def test_complementary_nested_rectangles_replace_one_envelope() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    enclosing = production_stage.BoxComponent(
        box=(300, 800, 1500, 860),
        polygon=geometry_stage._rect_polygon((300, 800, 1500, 860)),
        source="rectilinear_zone_1:observed=4:virtual=0",
        score=8.0,
    )

    def candidate(box: tuple[int, int, int, int]) -> production_stage.BoxComponent:
        return production_stage.BoxComponent(
            box=box,
            polygon=geometry_stage._rect_polygon(box),
            source="lsd_geometry_rescue:h=0.99/0.99:v=0.98/0.98:corners=4",
            score=7.3,
        )

    accepted, _, diagnostics = production_stage._nested_measured_outline_candidates(
        page,
        31,
        [enclosing],
        [candidate((300, 802, 650, 858)), candidate((652, 802, 1500, 858))],
    )
    retained, removed = production_stage._remove_partitioned_envelopes(
        [enclosing], accepted
    )

    assert len(accepted) == 2
    assert all(
        component.source.startswith("nested_measured_partition:")
        for component in accepted
    )
    assert diagnostics["nested_outline_partition_replacement_count"] == 2
    assert retained == []
    assert removed == 1


def test_stacked_internal_seams_do_not_partition_one_outline() -> None:
    page = np.full((1200, 800), 255, dtype=np.uint8)
    enclosing = production_stage.BoxComponent(
        box=(210, 400, 610, 470),
        polygon=geometry_stage._rect_polygon((210, 400, 610, 470)),
        source="rectilinear_zone_1:observed=4:virtual=0",
        score=8.0,
    )

    def candidate(box: tuple[int, int, int, int]) -> production_stage.BoxComponent:
        return production_stage.BoxComponent(
            box=box,
            polygon=geometry_stage._rect_polygon(box),
            source="lsd_geometry_rescue:h=0.99/0.99:v=0.98/0.98:corners=4",
            score=7.3,
        )

    accepted, _, diagnostics = production_stage._nested_measured_outline_candidates(
        page,
        24,
        [enclosing],
        [candidate((210, 402, 610, 435)), candidate((210, 438, 610, 468))],
    )

    assert accepted == []
    assert diagnostics["nested_outline_partition_replacement_count"] == 0
    assert diagnostics["nested_outline_rejected_shared_sides_count"] == 2


def test_occluded_page_frame_and_nested_detail_remain_diagnostic() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    outer = production_stage.BoxComponent(
        box=(420, 45, 980, 135),
        polygon=geometry_stage._rect_polygon((420, 45, 980, 135)),
        source="lsd_geometry_rescue:h=0.42/0.72:v=0.95/0.82:corners=2",
        score=5.7,
    )
    detail = production_stage.BoxComponent(
        box=(700, 92, 736, 124),
        polygon=geometry_stage._rect_polygon((700, 92, 736, 124)),
        source="lsd_geometry_rescue:h=0.92/0.91:v=0.96/0.95:corners=4",
        score=7.1,
    )

    body, frames, _ = production_stage._promote_supported_lsd_outlines(
        page, 31, [], [outer, detail]
    )
    deduped = production_stage._dedupe_page_frame_candidates(frames)

    assert body == []
    assert len(deduped) == 2
    assert any(":nested_detail:" in component.source for component in deduped)


def test_text_crossing_step_is_replaced_by_measured_blank_slabs() -> None:
    page = np.full((2200, 1700), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(page, "visible prose crosses this synthetic leg", (520, 955), font, 0.8, 0, 2)
    cv2.putText(page, "and continues on another row", (520, 990), font, 0.8, 0, 2)
    step = production_stage.BoxComponent(
        box=(500, 850, 1300, 1050),
        polygon=((500, 850), (1300, 850), (1300, 900), (760, 900), (760, 1050), (500, 1050)),
        source="shared_edge_step_completion:tabs=1:test",
        score=8.0,
    )
    top = production_stage.BoxComponent(
        box=(500, 850, 1300, 900),
        polygon=geometry_stage._rect_polygon((500, 850, 1300, 900)),
        source="nested_measured_step_replacement:test",
        score=7.0,
    )

    retained, removed = production_stage._remove_text_crossing_step_envelopes(
        page, 31, [step], [top]
    )

    assert retained == []
    assert removed == 1


def test_overlapping_blackout_union_recovers_maximal_rectangles() -> None:
    mask = np.zeros((500, 1200), dtype=np.uint8)
    cv2.rectangle(mask, (50, 40), (400, 280), 255, -1)
    cv2.rectangle(mask, (160, 190), (1100, 460), 255, -1)

    boxes = validated_stage._filled_run_rectangles(mask, (0, 0), 31)

    assert boxes == [(50, 40, 401, 281), (160, 190, 1101, 461)]


def test_irregular_solid_cap_at_page_edge_is_suppressed() -> None:
    cap_box = (800, 45, 848, 74)
    cap = production_stage.BoxComponent(
        box=cap_box,
        polygon=(
            (800, 45),
            (848, 45),
            (846, 62),
            (838, 72),
            (823, 73),
            (808, 68),
            (801, 57),
        ),
        source="solid_fill_shape_preserved",
        score=5.8,
    )
    broad_header_box = (500, 80, 900, 150)
    broad_header = production_stage.BoxComponent(
        box=broad_header_box,
        polygon=geometry_stage._rect_polygon(broad_header_box),
        source="solid_fill_shape_preserved",
        score=7.0,
    )

    retained, suppressed, dense_regime = (
        production_stage._suppress_marginal_solid_specks(
            [cap, broad_header],
            (240, 1450),
            32,
            (2200, 1700),
        )
    )

    assert dense_regime
    assert suppressed == [cap]
    assert retained == [broad_header]


def test_blackout_partitioning_requires_a_material_page_regime() -> None:
    ordinary_page = np.full((2200, 1700), 255, dtype=np.uint8)
    cv2.rectangle(ordinary_page, (200, 500), (280, 520), 0, -1)
    ordinary, _ = validated_stage._blackout_partition_candidates(
        ordinary_page,
        31,
        [],
    )

    spill_page = np.full((2200, 1700), 255, dtype=np.uint8)
    cv2.rectangle(spill_page, (200, 500), (1200, 650), 0, -1)
    spill, _ = validated_stage._blackout_partition_candidates(
        spill_page,
        31,
        [],
    )

    assert ordinary == []
    assert spill


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LATER_OCCLUDED_PDF = (
    REPOSITORY_ROOT / "data/docs/unredacted_pdfs/cib_02739300.pdf"
)
EARLIER_MARGIN_PDF = (
    REPOSITORY_ROOT
    / "data/docs/redacted_pdfs/CIA-RDP79T00975A000100360001-0.pdf"
)
LATER_MARGIN_PDF = REPOSITORY_ROOT / "data/docs/unredacted_pdfs/cib_02733106.pdf"


def _render_pdf_page(path: Path, page_number: int) -> np.ndarray:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open(path)
    pixmap = document[page_number - 1].get_pixmap(
        matrix=pymupdf.Matrix(200 / 72, 200 / 72),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width
    )


@pytest.mark.skipif(not LATER_OCCLUDED_PDF.exists(), reason="local PDFs absent")
def test_real_boilerplate_occluded_step_is_recovered() -> None:
    page = _render_pdf_page(LATER_OCCLUDED_PDF, 3)
    regions, _, diagnostics = detect_redaction_regions_with_artifacts(page)
    components = [component for region in regions for component in region.components]
    recovered = [
        component
        for component in components
        if component.source.startswith("occluded_step_outline:")
    ]

    assert diagnostics["occluded_step_component_count"] == 2
    assert len(recovered) == 2
    assert min(component.box[0] for component in recovered) <= 750
    assert max(component.box[2] for component in recovered) >= 1500


@pytest.mark.skipif(not EARLIER_MARGIN_PDF.exists(), reason="local PDFs absent")
def test_real_margin_specks_are_suppressed() -> None:
    page = _render_pdf_page(EARLIER_MARGIN_PDF, 5)
    regions, _, diagnostics = detect_redaction_regions_with_artifacts(page)
    components = [component for region in regions for component in region.components]

    assert diagnostics["marginal_solid_suppression_count"] == 4
    assert all(
        not (
            component.source.startswith(("solid_fill", "dense_blackout"))
            and component.box[2] < 100
        )
        for component in components
    )


@pytest.mark.skipif(not LATER_MARGIN_PDF.exists(), reason="local PDFs absent")
def test_real_later_release_margin_specks_are_suppressed() -> None:
    page = _render_pdf_page(LATER_MARGIN_PDF, 5)
    regions, _, diagnostics = detect_redaction_regions_with_artifacts(page)
    components = [component for region in regions for component in region.components]

    assert diagnostics["marginal_solid_suppression_count"] == 3
    assert all(
        not (
            component.source.startswith(("solid_fill", "dense_blackout"))
            and component.box[2] < 100
        )
        for component in components
    )
