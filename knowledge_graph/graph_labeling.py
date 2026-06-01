#!/usr/bin/env python3
"""
Generic graph-labeling helpers for Stage B and Stage C.

These are intentionally dataset-agnostic fallback utilities. They do not
hardcode corpus-specific topic taxonomies.
"""

from __future__ import annotations

import re
from typing import List

DISPLAY_LABEL_SEPARATOR = " / "
TOP_LABELS_SEPARATOR = " | "


def clean_label_text(value: object) -> str:
    text = str(value if value is not None else "")
    text = re.sub(r"\s+", " ", text.replace("_", " ")).strip()
    return text.strip("|;/")


def split_label_list(value: object) -> List[str]:
    text = str(value if value is not None else "")
    parts = re.split(r"\s+\|\s+|;", text)
    return [clean_label_text(part) for part in parts if clean_label_text(part)]


def _has_any(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


def make_theme_family(top_labels: object, fallback: str = "") -> str:
    text = str(top_labels).upper()
    fallback = clean_label_text(fallback)

    if _has_any(text, ["LABOR", "STRIKE", "WORKERS", "UNION", "PEASANT"]):
        return "Labor unrest and internal political pressure"
    if _has_any(text, ["OIL", "PETROLEUM", "REFINERY", "PIPELINE"]):
        return "Oil, energy infrastructure, and economic leverage"
    if _has_any(text, ["NATO", "UNITED NATIONS", "UNGA", "CENTO", "SEATO", "OAS"]):
        return "International diplomatic and security coordination"
    if _has_any(text, ["ARMY", "MILITARY", "SECURITY", "DEFENSE", "COUP", "REBELS"]):
        return "Military posture, insurgency, and regime stability"
    if _has_any(text, ["ELECTION", "PARLIAMENT", "CABINET", "PRESIDENT", "PREMIER", "GOVERNMENT"]):
        return "Government leadership, cabinet politics, and state decision-making"
    if _has_any(text, ["EMBASSY", "AMBASSADOR", "CONSUL", "STATE DEPARTMENT", "FOREIGN MINISTER"]):
        return "Diplomatic reporting and foreign policy activity"
    if _has_any(text, ["SOVIET", "USSR", "MOSCOW", "KREMLIN", "COMMUNIST"]):
        return "Cold War bloc politics and strategic positioning"

    parts = split_label_list(top_labels)
    if len(parts) >= 2:
        return f"{parts[0]} and related regional affairs"[:90]
    if parts:
        return f"{parts[0]}-centered political affairs"[:90]
    return fallback[:90] if fallback else "Regional political and diplomatic affairs"


def make_topic_label(top_labels: object, fallback: str = "") -> str:
    parts = split_label_list(top_labels)
    if len(parts) >= 2:
        return f"{parts[0]} / {parts[1]}"[:90]
    if parts:
        return parts[0][:90]
    return clean_label_text(fallback)[:90]
