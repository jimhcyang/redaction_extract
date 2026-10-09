# Redaction Box CV 3.6

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

## Visual repository manual

Open [`manual/index.html`](manual/index.html) for the concise interactive
walkthrough. Its three real PDF exhibits let the reader switch between release
versions and inspect the page, masks, measured lines, proposal families, rule
decisions, and final reading-unit grouping. The manual then follows the
independent Astra target through text localization, scan registration,
projection, and coverage scoring. A 12-page printable companion is available at
[`manual/REDACTION_BOX_CV_REPOSITORY_MANUAL_3_6.pdf`](manual/REDACTION_BOX_CV_REPOSITORY_MANUAL_3_6.pdf).

The manual is a static, self-contained repository artifact. It uses the exact
3.6 detector outputs, but it does not place source answers, paired-page pixels,
or manual annotations in the detector path.

Release 3.2 added conservative handling for layered and overlapping masks. A
near-line-height blank cell is retained when it has independent measured
borders but would otherwise be discarded inside a larger text-crossing
candidate. Fully observed rectangles keep that provenance after line-segment
boundary refinement, so a touching mask cannot make a valid overlapping box
look like an inferred text frame. Reading-order grouping also keeps two units
separate when a broad box touches left and right continuations with visible
prose between them; the right continuation may then join a verified next-line
component. These rules use page pixels and measured layout only.

Release 3.3 adds two narrow corpus-audit fixes. An incomplete stepped outline
is recovered only when a validated three-sided seed, a longer aligned rail, a
divider rail, and two blank slabs agree; visible text must occupy the excluded
notch, which is never filled. Tiny solid contours far outside the measured
reading column are treated as scan specks unless the page contains a material
dense-ink redaction. Ordinary four-sided outline detection is unchanged.

Release 3.4 adds two further measured safeguards from a 24-page uncertainty
review. A rectangle nested inside a broader blank mask is retained only when
four measured corners, strong support on every side, a blank interior, and
side independence all agree; a candidate that reuses two enclosing sides is
rejected as a likely subdivision. Separately, a small irregular solid cap in
the extreme top or bottom page band is rejected as scan or release-mark
furniture. The latter rule remains active on dense-ink pages but does not
remove broad rectangular header masks. Both routes use page pixels and
page-local scale only.

Release 3.5 incorporates the independently labeled six-pair holdout without
using labels at runtime. It recovers release-stamp-occluded header/footer
outlines as audit-only page furniture; replaces a broad envelope with two
side-by-side physical boxes only when strong four-corner measurements tile a
wide, multi-token span; replaces synthetic text-crossing step envelopes with
their directly measured blank slabs; and represents layered dense blackouts by
maximal overlapping rectangles when those rectangles reproduce the same ink
union. Compact and stacked seam subdivisions remain one box. These routes use
only page pixels and page-local scale and remain unavailable to answer
assignment.

Release 3.6 addresses three production failure classes without reading any
document ID, answer, paired page, OCR text, or annotation at runtime. First,
one-side-occluded and fragmented-rail outlines recover otherwise blank masks
whose fourth wall is hidden by release furniture, a diagonal edge, or a broken
scan. Second, scanner-frame strips and bold-text blobs are suppressed unless a
page-level dense-ink regime and a materially large, pitch-black component both
support blackout treatment. Third, reading-unit grouping now distinguishes a
real stepped or windmill-shaped compound mask from two boxes that only share a
paragraph indentation. End-of-line decisions use the inferred text body, not
a permissive fraction of the full scanner canvas. Concave outlines remain
measured polygons rather than being converted into synthetic rectangular
parts.

## Rule families and audit counters

The implementation has five rule families. This grouping is explanatory; the
saved geometry retains the exact low-level route for every component.

1. **Page measurement** estimates skew, text-line height, reading-column
   bounds, contours, and horizontal/vertical line segments. These counters
   describe search volume, not accepted redactions.
2. **Ordinary geometry** recovers complete rectilinear zones, measured LSD
   corner refinements, and closed blank contours. These routes produce most
   output components.
3. **Special morphology** handles independently bounded overlapping or nested
   rectangles, stepped/concave outlines, page-edge boxes, and genuine dense
   blackout regions. Each route has stricter evidence gates than ordinary
   geometry.
4. **False-positive guards** reject visible prose frames, glyph-like slivers,
   subdivisions that reuse enclosing sides, and small irregular scan caps.
   Dense-ink mode relaxes only the relevant solid-mark guard; it does not turn
   every dark contour into a redaction.
5. **Reading-order grouping** joins physical components only after geometry is
   frozen. Region IDs such as `R2.1` and `R2.2` therefore describe one reading
   unit with two boxes; they do not mean one irregular bounding rectangle was
   hallucinated around both boxes.

Rule telemetry reports three distinct quantities: `pipeline_measurement` for
candidate/search counts, `rule_decision` for promotions, rejections, merges,
and suppressions, and `emitted_component` for the final provenance route of a
saved physical box. A high candidate count is not evidence that a permissive
rule fired frequently.

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

The complete release-3.3 audit processed all 1,414 v4 items over 1,277 page
pairs with zero page-pair errors: 1,400 were `ASSIGNED_FULL`, 11 were
`ASSIGNED_PARTIAL`, one had uncertain target localization, and two targets were
not present in the PDF text layer. These are target-assignment outcomes, not a
claim that every redaction box on every source page has been exhaustively
labeled. Against the frozen 3.2 audit, 11 of 2,554 release-page geometries
changed: three stepped-outline recoveries and eight margin-speck removals.
There were no item- or fragment-status transitions.

Release 3.4 preserves the exact release-3.3 metrics on all 216 previously
labeled regression pages. On the 24 newly labeled uncertainty-review pages,
body-output component F1 improved from `0.8982` to `0.9167`; when separately
audited page-frame candidates are included as physical outlines, F1 improved
from `0.9314` to `0.9489`. The exact geometry delta is four recovered nested
components on two pages and two removed top-edge scan caps on two pages. The
new nested route emitted no components on the older 216-page regression set.

The complete release-3.4 items-v4 audit then processed all 1,414 items over
1,277 page pairs (2,554 release pages) with zero errors. Relative to 3.3,
32 independently measured nested rectangles were added on 20 release pages and
21 irregular marginal solid artifacts were removed on 21 release pages. No
page status, fragment status, or `ASSIGNED_FULL` result changed. Those
unchanged assignment totals are useful non-regression evidence, but the 41
physical-geometry changes are audited separately because target coverage is
not an exhaustive box annotation.

A final six-pair, twelve-image holdout was selected only after this detector
3.4 and its complete production run were frozen. It contains three nested-outline
acceptance cases, an all-rejected shared-side boundary, an artifact-suppression
case, and a dense-ink guard case. Its source images contain no copied labels or
detector overlays. After independent manual annotation, release 3.5 improves
physical component F1 from `0.9271` to `0.9588`, region F1 from `0.9508` to
`0.9841`, and page-union IoU from `0.9679` to `0.9774`. These are post-holdout
refinement results; the original release-3.4 score remains the blind estimate.

The release-3.5 production-risk gate used frozen 3.4 diagnostics to select all
181 release pages capable of activating a changed body rule. Conservative
screening selected 106 affected v4 items for exact source-PDF replay, covering
91 page pairs and 113 fragments after cross-page expansion. All 91 registrations
passed; every item and fragment remained `ASSIGNED_FULL`; and there were zero
item, page, fragment, or `ASSIGNED_FULL` regressions. Eighteen release-page
geometries across 17 page pairs changed as expected. This is an exhaustive
affected-rule replay, not a second full 1,277-pair production audit.

Release 3.6 passed a new complete production audit over all 1,414 items,
1,446 fragments, 1,277 page pairs, and 2,554 release pages. It completed with
zero page-pair errors and retained the frozen 80%-threshold item totals:
1,400 `ASSIGNED_FULL`, 11 `ASSIGNED_PARTIAL`, one uncertain target
localization, and two targets absent from the PDF text layer. Against the
complete 3.4 baseline, 106 release pages changed geometry or grouping, but
there were zero page-status transitions, fragment-status transitions, or
regressions from `ASSIGNED_FULL`.

The same 3.6 detector was also scored after detection against 252 manually
labeled pages across six collections. Component F1 is `0.9760` on the
118-page curated set, `1.0000` on the 40-page dense set, `0.9950` on the
40-page final-task pilot, `0.9474` on the deliberately difficult 18-page hard
set, `0.9199` on the 24-page uncertainty review, and `0.8962` on the 12-page
body-only holdout. Labels are evaluation-only and are never read by the
detector. Corpus telemetry confirms the new rules remain narrow: fragmented
rail recovery emitted 29 components on 28 of 2,554 pages, scanner-frame-strip
suppression activated on two pages, and only one page emitted a dense-blackout
component.

Overlay labels are capped at 82% of the detector's measured text-line height.
For irregular components, candidate positions are generated from actual
polygon vertices and an interior distance peak; the chosen marker remains
inside the detected polygon and avoids existing markers where possible.
To redraw an existing audit
after presentation-only changes without rerunning detection or scoring:

```bash
.venv/bin/python -m box_scripts.v4_audit --refresh-render --workers 6
```

For a quick local test, add `--max-page-pairs 5 --out /tmp/v4_box_smoke`.

## Output meaning

- `ASSIGNED_FULL`: every page fragment of the v4 item has at least 80% target
  word-center coverage by detected geometry. The threshold is configurable
  with `--full-coverage-threshold`; use `0.90` only as an explicit sensitivity
  condition.
- `ASSIGNED_PARTIAL`: some target geometry was recovered, but the whole item did
  not pass.
- `TARGET_LINKED_GEOMETRY_MISS`: the saved text was located and the scans were
  registered, but less than 30% was covered by a detected box.
- Localization and registration failures are reported separately from detector
  misses.

The production default is 80%. A stricter 90% rerating changes status labels
without changing any detected geometry; such transitions must be reported as
threshold-policy sensitivity, not detector regressions.

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
