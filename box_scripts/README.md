# box_scripts

Self-contained redaction-box detection workflow.

This folder is independent of the OCR/diff pipeline. It reproduces the redaction-box detector that produced artifacts such as `05_redaction_boxes.png`, and packages just the parts needed to run box detection on images or PDFs.

## What it outputs

For each processed item it writes a subfolder under the chosen output directory containing:

- `<item_key>.source.png`
  - the source raster image used for detection
- `<item_key>.redaction_boxes.png`
  - the box overlay image corresponding to the old `05_redaction_boxes.png`
- `<item_key>.redaction_boxes.json`
  - metadata including image size, source path, and `redaction_boxes_xyxy`
- `debug/*.png` if `--save_debug_masks` is used
  - `dark_mask.png`, `lines_mask.png`, `contours_mask.png`, `edges.png`

It also writes:

- `box_manifest.json`
- `box_manifest.jsonl`
- `box_summary.json`

## Supported inputs

1. `--input <file-or-dir>`
- single image
- single PDF
- directory of images and/or PDFs

2. `--docs_root <dir>`
- expects `cibcia.csv`, `redacted_pdfs/`, `unredacted_pdfs/`
- validates pairs using the CSV mapping
- processes `redacted`, `unredacted`, or `both` via `--source_kind`

## Install

```bash
pip install -r box_scripts/requirements.txt
```

## Examples

Single PNG:

```bash
python box_scripts/run_box_pipeline.py \
  --input 0_redacted.png \
  --out box_outputs_single \
  --save_debug_masks
```

Directory of PNGs/PDFs:

```bash
python box_scripts/run_box_pipeline.py \
  --input example_batch \
  --out box_outputs_batch
```

Docs root using valid redacted PDFs from `cibcia.csv`:

```bash
python box_scripts/run_box_pipeline.py \
  --docs_root docs_example \
  --source_kind redacted \
  --out box_outputs_docs \
  --max_files 10 \
  --max_pages 2
```

## Notes

- The primary research artifact is `<item_key>.redaction_boxes.png` plus `<item_key>.redaction_boxes.json`.
- PDF support uses `pypdfium2` to rasterize pages before detection.
- This package does not require OpenAI or any OCR model.
