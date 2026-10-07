# Redaction Box Scripts 3.1

This repository contains the final classical redaction-box geometry detector,
its four manually labeled evaluation collections, the frozen Astra comparison
inputs and results, a self-contained HTML review, and 8
representative paired CIB documents. Open `results/index.html` first.

## What the detector sees

The detector processes one raster page independently. It sees grayscale pixels,
line segments, contours, blank interiors, dense ink, page margins, and locally
measured geometric support. It does **not** see OCR text, document IDs, filenames
as semantic features, the paired release, Astra answers, or manual labels. It
outputs physical polygons and groups line-wrapped components into reading-order
redaction units; it does not infer hidden words.

The production operating point is 200 DPI. PDF inputs are rendered at that
scale. Raster inputs should be 200-DPI-equivalent for benchmark-comparable
results.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_box_pipeline.py \
  --input manual_gold/curated/images/example_000_earlier.original.png \
  --out outputs/example \
  --dpi 200
```

Run the included paired-PDF sample:

```bash
python run_box_pipeline.py \
  --docs-root data/example_pdfs \
  --source-kind both \
  --out outputs/example_pairs \
  --dpi 200
```

Each page emits the rendered source, labeled overlay, and a JSON record with
source-coordinate polygons, reading-unit labels `R<n>`, physical-component
labels `R<n>.<m>`, proposal provenance, and diagnostics.

## Final implementation

`box_scripts/box_pipeline.py` is the public 3.1 detector. The underscore-named
modules are required internal layers of this same final implementation, not
selectable older releases.

Public functions:

- `detect_redaction_regions_with_artifacts(gray)` returns grouped regions,
  diagnostic masks, and decision metadata for one grayscale page;
- `detect_redaction_boxes_with_artifacts(gray)` provides the compatibility
  box view plus artifacts;
- `process_record(...)` processes one page record and writes source, overlay,
  and JSON outputs;
- `run_box_pipeline(...)` processes an image, PDF, directory, or paired corpus;
- `collect_pdf_pairs_with_stats(...)` validates and enumerates paired CIB PDFs.

1. Render at a canonical physical scale and deskew while retaining mappings to
   source coordinates.
2. Propose ordinary boxes from dark/faint masks, contours, horizontal and
   vertical segments, corners, blank interiors, and dense blackouts.
3. Recover supported page-edge, overlapping, clipped, layered, stepped, or
   concave geometry using measured borders and compact rectangular covers.
4. Reject glyph-sized strokes, prose-crossing envelopes, unsupported virtual
   borders, page furniture, duplicate boxes, and intersection-only cells.
5. Split a synthesized stepped outline when a full text-line-height
   slab contains sustained visible prose; independently blank slabs survive.
6. Allow an incomplete supported rectangle to expand only to the closed
   white-space component containing most of its already validated seed, with
   strict area, support, retention, fill, and expansion limits.
7. Group surviving physical components only when natural line-wrap geometry
   supports one continuous redacted reading unit.

Rule precedence is conservative: completion routes can recover geometry but
cannot bypass evidence, prose-suppression, deduplication, or grouping checks.
No rule contains benchmark IDs, document-specific coordinates, or expected
answers.

## Evaluation

| Collection | Pages | Gold components | Component F1 @ IoU 0.50 | Purpose |
|---|---:|---:|---:|---|
| Curated | 118 | 336 | 97.60% | Diverse screenshots and prior failure modes |
| Dense gold | 40 | 185 | 100.00% | Exhaustively labeled box-heavy pages |
| Final-task pilot | 40 | 200 | 99.50% | Production-style paired pages |
| Hard cases | 18 | 119 | 97.54% | Prior Astra disagreements and final refinement target |

Hard-case physical recall is
100.00% at IoU 0.50 and
98.32% at IoU 0.75;
precision at IoU 0.50 is
95.20%.
These collections guided rule development and serve as regression suites, not
an untouched estimate of performance on an unrelated corpus.

## Astra comparison

Astra and Box CV perform different tasks. Astra saw paired releases and saved
semantic answers plus visible context. Box CV sees one page and emits geometry.
The local comparison adapter then locates each frozen Astra answer in the later
PDF text layer, registers the later scan to the earlier scan with SIFT/RANSAC,
projects answer-word centers into the earlier page, and measures their coverage
by Box CV polygons. No new LLM or OCR service is called at this stage.

Across 1,592 frozen benchmark targets, release 3.1 records:

- **1583** full assignments;
- **6** partial assignments;
- **2** unique changed-region assignments;
- **1** localization-uncertain assignment.

`ASSIGNED_FULL` means at least 80% of localized answer-word centers fall inside
detected geometry. This is strong evidence that Box CV found the physical hidden
region associated with Astra's independently saved answer. It is not a claim
that Box CV recovered the words itself or that the two methods are semantically
equivalent. `results/index.html` exposes every manual page and every non-full
production target so the aggregate can be audited visually.

## Repository layout

```text
box_scripts/              Final detector and required private implementation layers
run_box_pipeline.py       Command-line entry point
manual_gold/              Images, annotations, detector overlays, and metrics
data/benchmark/           Frozen local benchmark/Astra artifacts
data/example_pdfs/        16 PDFs from 8 representative document pairs
results/index.html        Standalone review browser
results/data/             Sanitized production table and release summaries
RELEASE_MANIFEST.json     Scope, counts, and code provenance
SHA256SUMS.txt             Integrity hashes
```

The full 6,018-PDF corpus is intentionally omitted because it is roughly 2.3 GB.
The included PDFs reproduce the hard-case examples. To run the full corpus,
provide the original `cibcia.csv`, `redacted_pdfs/`, and `unredacted_pdfs/`
under a separate `--docs-root`.

## Limits

- Geometry detection does not transcribe or reconstruct hidden text.
- Astra coverage depends on frozen answer spans, PDF text localization, and
  scan registration as well as detector geometry.
- Full-corpus regions outside benchmark answer locations are not exhaustively
  hand labeled.
- Native 300-DPI behavior is not the release operating point; use 200 DPI.
