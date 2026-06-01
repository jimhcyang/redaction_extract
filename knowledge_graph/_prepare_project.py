#!/usr/bin/env python3
"""
Prepare a knowledge-graph project folder from this repo's CIA unredacted corpus.

The output project folder is compatible with Notebook A's expected input schema:
    doc_id, id, subject, date, body

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PAIR_METADATA_COLUMNS = [
    "collection",
    "record_id",
    "date_display",
    "source_pdf_url",
    "date_compact",
    "cia_document_id",
    "document_url",
    "meta_path",
    "title",
    "year",
    "paired_flag",
]

def safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def normalize_date(*candidates: object) -> str:
    for candidate in candidates:
        text = safe_str(candidate).strip()
        if not text:
            continue
        parsed = pd.to_datetime(text, errors="coerce")
        if not pd.isna(parsed):
            return parsed.strftime("%Y-%m-%d")
    return safe_str(candidates[0]).strip() if candidates else ""


def clean_transcription_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\[\s*PAGE\s+\d+\s*\]", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pair_metadata(meta_path: Path) -> Dict[str, str]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    csv_row = payload.get("csv_row") or []

    row_map: Dict[str, str] = {}
    for idx, name in enumerate(PAIR_METADATA_COLUMNS):
        row_map[name] = safe_str(csv_row[idx]) if idx < len(csv_row) else ""

    row_map["pair_key"] = safe_str(payload.get("pair_key")) or meta_path.parent.parent.name
    row_map["model"] = safe_str(payload.get("model"))
    return row_map


def build_document_record(doc_dir: Path) -> Optional[Dict[str, str]]:
    meta_path = doc_dir / "document" / "pair_metadata.json"
    text_path = doc_dir / "document" / "unredacted.transcription.txt"

    if not meta_path.exists() or not text_path.exists():
        return None

    metadata = parse_pair_metadata(meta_path)
    body = clean_transcription_text(text_path.read_text(encoding="utf-8", errors="ignore"))
    if not body:
        return None

    pair_key = metadata["pair_key"] or doc_dir.name
    cia_document_id = metadata["cia_document_id"] or pair_key.split("_", 1)[-1]
    subject = metadata["title"] or metadata["collection"] or cia_document_id
    date_value = normalize_date(metadata["date_compact"], metadata["date_display"])

    return {
        "doc_id": pair_key,
        "id": cia_document_id,
        "subject": subject,
        "date": date_value,
        "body": body,
        "collection": metadata["collection"],
        "record_id": metadata["record_id"],
        "date_display": metadata["date_display"],
        "date_compact": metadata["date_compact"],
        "document_title": metadata["title"],
        "year": metadata["year"],
        "document_url": metadata["document_url"],
        "source_pdf_url": metadata["source_pdf_url"],
        "pair_key": pair_key,
        "pair_model": metadata["model"],
        "transcription_path": str(text_path),
        "metadata_path": str(meta_path),
        "tag": "",
        "concepts": "",
    }


def collect_document_records(corpus_root: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    for doc_dir in sorted(corpus_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        record = build_document_record(doc_dir)
        if record is None:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    return records


def build_dataframe(records: List[Dict[str, str]]) -> pd.DataFrame:
    if not records:
        raise ValueError("No unredacted transcription records were found.")

    df = pd.DataFrame(records)
    df["date_sort"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date_sort", "doc_id"], na_position="last").drop(columns=["date_sort"])

    ordered_columns = [
        "doc_id",
        "id",
        "subject",
        "date",
        "body",
        "collection",
        "record_id",
        "date_display",
        "date_compact",
        "document_title",
        "year",
        "document_url",
        "source_pdf_url",
        "pair_key",
        "pair_model",
        "transcription_path",
        "metadata_path",
        "tag",
        "concepts",
    ]
    return df[ordered_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Matryoshka project directory from the CIA unredacted corpus."
    )
    parser.add_argument(
        "--corpus-root",
        default="postprocessed",
        help="Root folder containing per-document cleaned corpus directories. Use postprocessed by default.",
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Output project directory to create or refresh.",
    )
    parser.add_argument(
        "--csv-name",
        default="input_documents.csv",
        help="Name of the Notebook A input CSV written inside the project directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a small test run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_root = Path(args.corpus_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()

    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus root not found: {corpus_root}")

    project_dir.mkdir(parents=True, exist_ok=True)
    output_csv = project_dir / args.csv_name

    records = collect_document_records(corpus_root, limit=args.limit)
    df = build_dataframe(records)
    df.to_csv(output_csv, index=False)

    summary = {
        "corpus_root": str(corpus_root),
        "project_dir": str(project_dir),
        "input_csv": str(output_csv),
        "documents_written": int(len(df)),
    }
    (project_dir / "input_build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
