# Redaction Box CV 3.6

This repository is the standalone handoff of the same classical redaction-box
detector shipped in the `cv-box-detect` branch of Andrew Tang's
`redaction-extract` repository. It includes the detector, four independent
manual-label collections, a compact visual review, a complete web edition of
the items-v4 audit, a three-exhibit repository manual, and eight representative
paired CIB documents. Open `index.html` first.

## Information Boundary

The detector processes one raster page independently. It sees grayscale
pixels, line segments, contours, blank interiors, dense ink, page margins, and
page-local geometric support. It does **not** receive OCR text, filenames or
document IDs as semantic features, paired-release pixels, Astra answers, or
manual annotations. Those resources are used only after detection to evaluate
the frozen polygons.

The output distinguishes:

- `R<n>.<m>`: one physical rectangle or measured rectilinear component;
- `R<n>`: one inferred redacted reading unit containing one or more components.

## Visual Repository Manual

Open `box_scripts/manual/index.html` for a concise, interactive explanation of
the detector. Three real paired-release exhibits show the page, pixel masks,
measured lines, proposal families, guard decisions, and final grouping. The
manual then explains the independent Astra registration and target-coverage
audit. Its folder includes the 12-page printable
`REDACTION_BOX_CV_REPOSITORY_MANUAL_3_6.pdf`.

## Quick Start

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

Each page emits the source raster, a labeled overlay, and JSON containing exact
component polygons, reading-unit membership, original-image coordinates,
proposal provenance, and diagnostics.

## Final Implementation

`box_scripts/box_pipeline.py` is the public detector. The underscore-prefixed
modules are required implementation layers, not selectable historical
versions. Production evaluation uses 200 DPI. Native 300-DPI-size images are
analyzed at the validated 200-DPI physical scale and mapped back to their
original coordinates.

Release 3.6 adds conservative support for one-side-occluded and
scan-fragmented outlines, content-aware paragraph grouping, scanner-frame
suppression, and page-level gating of dense-blackout logic. Measured concave
or windmill contours remain polygons rather than being expanded into
text-covering rectangles. Neither these rules nor the label renderer uses OCR,
document identity, paired pixels, saved answers, or manual labels.

The main decision path is:

1. Deskew and build complementary dark/faint pixel masks.
2. Extract horizontal and vertical evidence, line zones, contours, corners,
   blank interiors, solid masks, and dense blackouts.
3. Propose ordinary, page-edge, overlapping, clipped, stepped, and concave
   geometry through independent candidate routes.
4. Preserve independently bounded layers and fully observed rectangle
   provenance through boundary refinement.
5. Reject glyph contours, prose-crossing frames, weak virtual borders,
   duplicate echoes, page furniture, and unsupported envelopes.
6. Keep physical components explicit; never replace a concave/overlapping
   union with a text-covering exterior rectangle.
7. Group components only through measured contact or verified reading flow.
   Visible prose between forked children prevents transitive over-grouping.

No production rule contains a benchmark ID, document-specific coordinate, or
expected answer.

## Evaluation

| Collection | Pages | Gold components | Precision | Recall | Component F1 | Region F1 |
|---|---:|---:|---:|---:|---:|---:|
| Curated | 118 | 336 | 98.19% | 97.02% | 97.60% | 95.79% |
| Dense gold | 40 | 185 | 100.00% | 100.00% | 100.00% | 97.10% |
| Final-task pilot | 40 | 200 | 99.01% | 100.00% | 99.50% | 98.09% |
| Targeted hard cases | 18 | 119 | 99.08% | 90.76% | 94.74% | 86.75% |

These are manually labeled development/regression collections, not untouched
population-level estimates. The hard set was deliberately selected from prior
failure and disagreement cases. Component and region scores are separate
because correct physical boxes can still be grouped differently.

`results/index.html` contains the compact curated and dense-gold review. Human
geometry is a heavy unfilled green outline; answer-blind CV geometry is a thin
red outline with restrained translucent fill. Manual labels are never loaded
by the production detector.

`results/items_v4/index.html` contains every one of the 1,414 items and 1,277
page-pair views in the items-v4 audit. It includes searchable target tables,
optimized labeled overlays, component geometry, registration diagnostics, and
downloadable result files. Duplicate raw-scan previews are omitted from this
web edition; the untouched local audit remains the full-resolution record.

## Relation To Astra And Items v4

Astra and Box CV perform different tasks. Astra saw paired releases and saved
semantic answers plus visible context. Box CV sees one page and emits geometry.
The optional `box_scripts/v4_audit.py` adapter then locates saved target words
in the later PDF text layer, registers the scans with SIFT/RANSAC, and measures
their post-hoc coverage by the already frozen earlier-page CV polygons. No LLM
or new OCR service is called.

The web edition of the large items-v4 audit is bundled for review. Reproducing
it still depends on the source repository's full 6,018-PDF local corpus. In
that environment run:

```bash
python -m box_scripts.v4_audit --workers 6 --overwrite
```

The complete release-3.6 audit processed all 1,414 items and 1,446 fragments
over 1,277 page pairs with zero page-pair errors. At the production 80% full
coverage threshold it reports 1,400 full assignments, 11 partial assignments,
one uncertain text localization, and two targets absent from the PDF text
layer. Against the complete 3.4 baseline, 106 release pages changed physical
geometry or grouping, with zero page-status, fragment-status, or
`ASSIGNED_FULL` regressions.

A coverage assignment means the independent CV geometry occupies the physical
location associated with saved target words. It does not mean CV transcribed
or inferred those words.

## Layout

```text
index.html            Landing page for both visual result browsers
box_scripts/          Final detector, v4 audit adapter, and regression tests
box_scripts/manual/   Interactive and printable three-exhibit repository manual
run_box_pipeline.py   Standalone command-line entry point
manual_gold/          Four manual-label collections and source images
data/example_pdfs/    16 PDFs from 8 representative document pairs
results/index.html    Compact standalone geometry-review browser
results/items_v4/     Complete optimized web edition of the items-v4 audit
tools/                Reproducible exporter for the items-v4 web edition
RELEASE_MANIFEST.json Scope, metrics, and code provenance
SHA256SUMS.txt         Integrity hashes
```

## GitHub Pages

After pushing the repository, open **Settings > Pages**, choose **Deploy from a
branch**, and select `main` with `/ (root)`. The bundled `.nojekyll` file keeps
the static audit assets unchanged. The expected public entry point is:

```text
https://jimhcyang.github.io/redaction_extract/
```

## Limits

- Geometry detection does not transcribe or reconstruct hidden text.
- Extremely faint, damaged, or document-furniture-like boxes may remain
  ambiguous.
- Grouping is less certain than component localization on selected hard cases.
- Full-corpus regions outside benchmark targets are not exhaustively labeled.
