# transcription_scripts

Self-contained OpenAI-based document transcription and redacted-vs-unredacted diff workflow.

This package only needs:

- `docs/`
- `transcription_scripts/`
- an `OPENAI_API_KEY`

It does not depend on the old `core/` code anymore.

## What it does

1. Reads `docs/cibcia.csv`
2. Keeps only valid 1:1 pairs where both PDFs exist
3. Sends each PDF to OpenAI with the archival transcription prompt
4. Stores full-document transcriptions under:
   - `transcriptions/<pair_key>/document/`
5. Splits transcriptions back into pages using `[PAGE XX]`
6. Aligns redacted and unredacted page streams globally
7. Builds aligned text, bracketed difference output, and chunk reports under:
   - `transcriptions/<pair_key>/pages/`
   - `transcriptions/<pair_key>/difference/`

## Install

```bash
pip install -r transcription_scripts/requirements.txt
```

## Required environment

```bash
export OPENAI_API_KEY=...
```

## Main commands

Single pair, end to end:

```bash
python transcription_scripts/run_openai_docs_pipeline.py \
  --docs_root docs \
  --out_root transcriptions \
  --pair_key 260604_CIA-RDP79T00975A003900030001-5 \
  --model gpt-5 \
  --skip_existing
```

Transcription only:

```bash
python transcription_scripts/openai_transcribe_pdfs.py \
  --docs_root docs \
  --out_root transcriptions \
  --model gpt-5 \
  --skip_existing
```

Postprocess only:

```bash
python transcription_scripts/postprocess_transcriptions.py \
  --docs_root docs \
  --out_root transcriptions \
  --skip_existing
```

## Notes

- Internet access is required because PDFs are uploaded to OpenAI.
- No GPU is required.
- The OpenAI workflow is separate from `box_scripts/`, which handles only redaction-box detection.
