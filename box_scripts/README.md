# Redaction Box CV 3.2

This folder contains the final classical redaction-box detector and the local
items-v4 audit adapter. Detection uses only one raster page at a time. It sees
grayscale pixels, measured line segments, contours, blank interiors, dense ink,
and page margins. It does not receive OCR text, answers, document identifiers,
paired-release pixels, Astra outputs, or benchmark labels.

The v4 adapter attaches saved benchmark information only after detection. It
locates each revealed v4 fragment in the later PDF text layer, registers the two
scans with SIFT/RANSAC, projects the target onto the earlier page, and reports
how much of the target falls inside detected geometry. No LLM, API, or new OCR
call is made.

Release 3.2 adds conservative handling for layered and overlapping masks. A
near-line-height blank cell is retained when it has independent measured
borders but would otherwise be discarded inside a larger text-crossing
candidate. Fully observed rectangles keep that provenance after line-segment
boundary refinement, so a touching mask cannot make a valid overlapping box
look like an inferred text frame. Reading-order grouping also keeps two units
separate when a broad box touches left and right continuations with visible
prose between them; the right continuation may then join a verified next-line
component. These rules use page pixels and measured layout only.

## Install

From the repository root:

```bash
source .venv/bin/activate
pip install -r box_scripts/requirements.txt
```

## Inspect the run scope

```bash
.venv/bin/python -m box_scripts.v4_audit --plan
```

## Run the frozen v4 audit

```bash
.venv/bin/python -m box_scripts.v4_audit \
  --workers 6 \
  --overwrite
```

Results are written under `box_results/items_v4/`. Open `index.html` for the
searchable review. Exact v4 word boxes and labeled CV components are separate
visual layers; each page also provides the raw scans and compact geometry JSON.
Use `--resume` instead of `--overwrite` to continue an
interrupted run; source and code hashes prevent resuming against incompatible
inputs.

The complete release-3.2 audit processed all 1,414 v4 items over 1,277 page
pairs with zero page-pair errors: 1,400 were `ASSIGNED_FULL`, 11 were
`ASSIGNED_PARTIAL`, one had uncertain target localization, and two targets were
not present in the PDF text layer. These are target-assignment outcomes, not a
claim that every redaction box on every source page has been exhaustively
labeled.

Overlay labels are capped at 82% of the detector's measured text-line height.
Nested or overlapping components try different corners and nearby exterior
lanes so their labels do not cover one another. To redraw an existing audit
after presentation-only changes without rerunning detection or scoring:

```bash
.venv/bin/python -m box_scripts.v4_audit --refresh-render --workers 6
```

For a quick local test, add `--max-page-pairs 5 --out /tmp/v4_box_smoke`.

## Output meaning

- `ASSIGNED_FULL`: every page fragment of the v4 item has at least 80% target
  word-center coverage by detected geometry.
- `ASSIGNED_PARTIAL`: some target geometry was recovered, but the whole item did
  not pass.
- `TARGET_LINKED_GEOMETRY_MISS`: the saved text was located and the scans were
  registered, but less than 30% was covered by a detected box.
- Localization and registration failures are reported separately from detector
  misses.

The public detector is `box_pipeline.py`. The underscore-prefixed modules are
required implementation layers, not selectable historical versions. Production
evaluation uses 200 DPI.

## Included manual review

`box_results/manual_validation/` contains a compact, self-contained browser for
the 118 curated pages and 40 dense-gold pages. These are manually labeled
development/regression collections, not untouched deployment tests. The browser
shows the original image, heavy unfilled green human components (`G:Rn.m`),
thin red CV components (`Rn.m`) with restrained translucent fill, one-to-one
IoU assignments, and geometry JSON. Final-pilot and hard-case development
collections are intentionally omitted from the shipped view.
