from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from box_scripts.box_pipeline import run_box_pipeline


def main() -> None:
    default_images_dir = ROOT / "images"
    parser = argparse.ArgumentParser(description="Detect redaction boxes and emit overlay PNG + metadata JSON.")
    parser.add_argument("--input", default=None, help=f"Image/PDF file or directory of images/PDFs. Defaults to {default_images_dir} when present.")
    parser.add_argument("--docs_root", default=None, help="Docs root containing cibcia.csv and PDF folders")
    parser.add_argument("--out", default="box_outputs", help="Output folder")
    parser.add_argument("--source_kind", choices=["redacted", "unredacted", "both"], default="redacted", help="Which side of docs_root pairs to process")
    parser.add_argument("--csv_name", default="cibcia.csv")
    parser.add_argument("--redacted_dir_name", default="redacted_pdfs")
    parser.add_argument("--unredacted_dir_name", default="unredacted_pdfs")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on number of input files or valid pairs")
    parser.add_argument("--max_pages", type=int, default=None, help="Optional cap on pages per PDF")
    parser.add_argument("--save_debug_masks", action="store_true", help="Also save dark/lines/contours/edges masks")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    docs_root = Path(args.docs_root) if args.docs_root else None
    if input_path is None and docs_root is None and default_images_dir.exists():
        input_path = default_images_dir
    if input_path is None and docs_root is None:
        raise SystemExit("Provide either --input or --docs_root.")

    summary = run_box_pipeline(
        out_root=Path(args.out),
        input_path=input_path,
        docs_root=docs_root,
        source_kind=args.source_kind,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        dpi=args.dpi,
        max_files=args.max_files,
        max_pages=args.max_pages,
        save_debug_masks=bool(args.save_debug_masks),
    )
    print(f"[BOXES] records={summary['record_count']} success={summary['success_count']} errors={summary['error_count']}")
    print(f"[BOXES] summary={Path(args.out) / 'box_summary.json'}")


if __name__ == "__main__":
    main()
