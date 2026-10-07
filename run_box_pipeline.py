#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from box_scripts.box_pipeline import run_box_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect physical redaction geometry on images or PDFs."
    )
    parser.add_argument("--input")
    parser.add_argument("--docs-root")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--source-kind", choices=["redacted", "unredacted", "both"],
        default="redacted",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--save-debug-masks", action="store_true")
    args = parser.parse_args()
    summary = run_box_pipeline(
        out_root=Path(args.out),
        input_path=Path(args.input) if args.input else None,
        docs_root=Path(args.docs_root) if args.docs_root else None,
        source_kind=args.source_kind,
        dpi=args.dpi,
        max_files=args.max_files,
        max_pages=args.max_pages,
        save_debug_masks=args.save_debug_masks,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
