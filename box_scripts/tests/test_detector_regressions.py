from __future__ import annotations

import cv2
import numpy as np

from box_scripts.box_pipeline import detect_redaction_regions_with_artifacts
from box_scripts.v4_audit import _boxes_overlap, _draw_label


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
