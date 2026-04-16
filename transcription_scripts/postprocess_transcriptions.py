from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transcription_scripts.openai_pdf_pipeline import postprocess_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Split, align, and diff OpenAI PDF transcriptions.")
    parser.add_argument("--docs_root", default="docs", help="Folder containing cibcia.csv and PDF directories")
    parser.add_argument("--out_root", default="transcriptions", help="Output transcription root")
    parser.add_argument("--csv_name", default="cibcia.csv")
    parser.add_argument("--redacted_dir_name", default="redacted_pdfs")
    parser.add_argument("--unredacted_dir_name", default="unredacted_pdfs")
    parser.add_argument("--pair_key", default=None, help="Run exactly one pair key")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional cap on selected valid pairs")
    parser.add_argument("--skip_existing", action="store_true", help="Skip pairs that already have difference outputs")
    args = parser.parse_args()

    postprocess_pairs(
        docs_root=Path(args.docs_root),
        out_root=Path(args.out_root),
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        pair_key=args.pair_key,
        max_pairs=args.max_pairs,
        skip_existing=bool(args.skip_existing),
    )


if __name__ == "__main__":
    main()
