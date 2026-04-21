from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transcription_scripts.openai_pdf_pipeline import TRANSCRIPTION_PROMPT, transcribe_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe valid redacted/unredacted PDF pairs with OpenAI.")
    parser.add_argument("--docs_root", default="docs", help="Folder containing cibcia.csv and PDF directories")
    parser.add_argument("--out_root", default="transcriptions", help="Output transcription root")
    parser.add_argument("--csv_name", default="cibcia.csv")
    parser.add_argument("--redacted_dir_name", default="redacted_pdfs")
    parser.add_argument("--unredacted_dir_name", default="unredacted_pdfs")
    parser.add_argument("--pair_key", default=None, help="Run exactly one pair key")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional cap on selected valid pairs")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model name")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing non-empty transcription outputs")
    parser.add_argument("--keep_uploaded_files", action="store_true", help="Do not delete uploaded OpenAI file objects after use")
    parser.add_argument("--prompt_file", default=None, help="Optional text file overriding the default transcription prompt")
    parser.add_argument("--max_retries", type=int, default=5, help="API retry count per PDF")
    args = parser.parse_args()

    prompt = TRANSCRIPTION_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    transcribe_pairs(
        docs_root=Path(args.docs_root),
        out_root=Path(args.out_root),
        model=args.model,
        prompt=prompt,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        pair_key=args.pair_key,
        max_pairs=args.max_pairs,
        skip_existing=bool(args.skip_existing),
        delete_uploaded_files=not bool(args.keep_uploaded_files),
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
