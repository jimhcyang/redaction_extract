"""Redaction Box CV public API."""

from .box_pipeline import (
    collect_pdf_pairs_with_stats,
    detect_redaction_boxes_with_artifacts,
    detect_redaction_regions_with_artifacts,
    process_record,
    run_box_pipeline,
)

__version__ = "3.6.0"
__all__ = [
    "collect_pdf_pairs_with_stats",
    "detect_redaction_boxes_with_artifacts",
    "detect_redaction_regions_with_artifacts",
    "process_record",
    "run_box_pipeline",
]
