"""Build a compact, self-contained browser for manual CV regression results.

This is a packaging utility, not an evaluator. It consumes the saved output of
the coordinate-level manual evaluator and copies only the curated and dense-gold
review material needed by a collaborator. Final-pilot and targeted hard-case
development sets are deliberately excluded from the shipped browser.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import __version__
from .v4_audit import CSS, _contour_bounds, _draw_label


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "box_results" / "manual_validation"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _annotation_index(directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path)
        key = f"{payload['sample_id']}_{payload['release']}"
        if key in result:
            raise ValueError(f"Duplicate manual annotation key: {key}")
        result[key] = (path, payload)
    return result


def _prediction_polygons(payload: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    polygons: list[tuple[str, np.ndarray]] = []
    for region in payload.get("redaction_regions", []):
        for component in region.get("components", []):
            points = component.get("original_image_polygon_xy") or component.get(
                "polygon_xy", []
            )
            if len(points) >= 3:
                polygons.append(
                    (
                        str(component.get("component_id", "R?")),
                        np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32),
                    )
                )
    return polygons


def _gold_polygons(payload: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    polygons: list[tuple[str, np.ndarray]] = []
    for region in payload.get("redaction_regions", []):
        for component in region.get("components", []):
            points = component.get("polygon_xy", [])
            if len(points) >= 3:
                polygons.append(
                    (
                        str(component.get("component_id", "G?")),
                        np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32),
                    )
                )
    return polygons


def _draw_manual_comparison(
    source: Path,
    annotation: dict[str, Any],
    prediction: dict[str, Any],
    output: Path,
) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read manual-review image: {source}")
    gold = _gold_polygons(annotation)
    predicted = _prediction_polygons(prediction)
    gold_color = (49, 137, 61)
    prediction_color = (42, 53, 202)
    # Predictions carry a restrained translucent fill; independent human gold
    # remains an unfilled, heavier outline. The two geometries can therefore
    # coincide without turning the underlying document into a muddy overlay.
    overlay = image.copy()
    for _, polygon in predicted:
        cv2.fillPoly(overlay, [polygon], prediction_color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, dst=image)
    for _, polygon in predicted:
        cv2.polylines(image, [polygon], True, prediction_color, 2, cv2.LINE_AA)
    for _, polygon in gold:
        cv2.polylines(image, [polygon], True, gold_color, 5, cv2.LINE_AA)
    diagnostics = prediction.get("detector_diagnostics", {})
    detection_scale = max(
        1e-6, float(diagnostics.get("input_to_detection_scale", 1.0))
    )
    line_height = float(diagnostics.get("estimated_text_line_height", 32)) / detection_scale
    occupied: list[tuple[int, int, int, int]] = []
    for index, (component_id, polygon) in enumerate(gold):
        _draw_label(
            image,
            f"G:{component_id}",
            _contour_bounds(polygon),
            gold_color,
            line_height=line_height,
            occupied=occupied,
            preferred_corner=index % 4,
        )
    for index, (component_id, polygon) in enumerate(predicted):
        _draw_label(
            image,
            component_id,
            _contour_bounds(polygon),
            prediction_color,
            line_height=line_height,
            occupied=occupied,
            preferred_corner=(index + 1) % 4,
        )
    maximum = 1800
    scale = min(1.0, maximum / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 84])


def _save_source_preview(source: Path, output: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read manual-review image: {source}")
    maximum = 1800
    scale = min(1.0, maximum / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 84])


def _category(row: dict[str, str]) -> tuple[str, list[str]]:
    gold = int(row["gold_components"])
    predicted = int(row["prediction_components"])
    matched = int(row["component_tp_iou50"])
    gold_regions = int(row["gold_regions"])
    predicted_regions = int(row["prediction_regions"])
    matched_regions = int(row["region_tp_iou50"])
    reasons: list[str] = []
    if matched < gold:
        reasons.append(f"{gold - matched} missed manual component(s)")
    if matched < predicted:
        reasons.append(f"{predicted - matched} unmatched CV component(s)")
    if reasons:
        return "component review", reasons
    if matched_regions < gold_regions or matched_regions < predicted_regions:
        return "grouping review", ["component geometry matches, but reading-unit grouping differs"]
    if int(row["component_tp_iou75"]) < gold or _number(row["page_union_iou"]) < 0.90:
        return "geometry review", ["all components match at IoU 0.50; tighter geometry differs"]
    return "pass", ["all component and reading-unit gates pass"]


def _compact_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector_version": payload.get("detector_version"),
        "image_size_wh": payload.get("image_size_wh"),
        "coordinate_system": payload.get("coordinate_system"),
        "redaction_region_count": payload.get("redaction_region_count"),
        "physical_component_count": payload.get("physical_component_count"),
        "redaction_regions": payload.get("redaction_regions", []),
        "detector_diagnostics": payload.get("detector_diagnostics", {}),
    }


def _case_html(
    suite: str,
    case: dict[str, Any],
    previous_id: str | None,
    next_id: str | None,
) -> str:
    nav = ["<a href='../../index.html'>Manual validation</a>"]
    if previous_id:
        nav.append(f"<a href='{_escape(previous_id)}.html'>Previous</a>")
    if next_id:
        nav.append(f"<a href='{_escape(next_id)}.html'>Next</a>")
    match_rows = "".join(
        f"<tr><td>{_escape(row.get('gold_component_id') or '-')}</td><td>{_escape(row.get('prediction_component_id') or '-')}</td><td>{_number(row.get('iou')):.3f}</td><td>{_escape(row.get('match_status'))}</td><td><code>{_escape(row.get('prediction_source'))}</code></td></tr>"
        for row in case["matches"]
    )
    metrics = case["metrics"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(case['id'])} - manual audit</title><style>{CSS}</style></head><body><main class="shell">
<nav class="nav">{' / '.join(nav)}</nav><div class="page-head"><div><div class="eyebrow">{_escape(case['suite_label'])}</div><h1>{_escape(case['id'])}</h1><p class="lede">Manual polygons are independent review labels. CV components are produced from the page pixels alone; the labels are used only after detection for scoring.</p></div><span class="badge">{_escape(case['category'])}</span></div>
<section class="explain-grid"><article class="callout"><b>Manual geometry</b>Green outlines labeled <code>G:Rn.m</code> are the human-drawn reference components.</article><article class="callout"><b>CV geometry</b>Red outlines labeled <code>Rn.m</code> are answer-blind detector components.</article><article class="callout"><b>Scoring</b>Components are assigned one-to-one. IoU 0.50 is the primary match gate; IoU 0.75 and page-union IoU diagnose boundary tightness.</article></section>
<section class="viewer"><article class="panel"><h2>Original page</h2><a href="../../assets/{_escape(suite)}/{_escape(case['id'])}_source.jpg"><img src="../../assets/{_escape(suite)}/{_escape(case['id'])}_source.jpg" alt="Original page"></a></article><article class="panel"><h2>Manual vs CV</h2><a href="../../assets/{_escape(suite)}/{_escape(case['id'])}_comparison.jpg"><img src="../../assets/{_escape(suite)}/{_escape(case['id'])}_comparison.jpg" alt="Manual and CV geometry"></a></article></section>
<section class="stats"><div class="stat"><strong>{metrics['gold components']}</strong>manual components</div><div class="stat"><strong>{metrics['predicted components']}</strong>CV components</div><div class="stat"><strong>{metrics['matches at 0.50']}</strong>matches at IoU 0.50</div><div class="stat"><strong>{metrics['page union IoU']:.1%}</strong>page-union IoU</div><div class="stat"><strong>{metrics['pixel precision']:.1%}</strong>pixel precision</div><div class="stat"><strong>{metrics['pixel recall']:.1%}</strong>pixel recall</div></section>
<section class="panel" style="padding:16px"><h2>Component assignment</h2><div class="table-wrap"><table><thead><tr><th>Manual</th><th>CV</th><th>IoU</th><th>Status</th><th>Detector route</th></tr></thead><tbody>{match_rows}</tbody></table></div><p><a href="../../geometry/{_escape(suite)}/{_escape(case['id'])}.json">CV geometry JSON</a> / <a href="../../annotations/{_escape(suite)}/{_escape(case['id'])}.json">manual annotation JSON</a></p></section></main></body></html>"""


def _suite(
    suite_id: str,
    suite_label: str,
    evaluation: Path,
    annotations_dir: Path,
    collection_dir: Path,
    output: Path,
) -> dict[str, Any]:
    annotations = _annotation_index(annotations_dir)
    matches: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(evaluation / "component_matches.csv"):
        matches[row["item_key"]].append(row)
    cases: list[dict[str, Any]] = []
    for row in _read_csv(evaluation / "per_image.csv"):
        key = row["item_key"]
        annotation_path, annotation = annotations[key]
        source = collection_dir / annotation["image"]["relative_path"]
        prediction_path = Path(row["prediction_file"])
        prediction = _read_json(prediction_path)
        category, reasons = _category(row)
        _save_source_preview(
            source, output / "assets" / suite_id / f"{key}_source.jpg"
        )
        _draw_manual_comparison(
            source,
            annotation,
            prediction,
            output / "assets" / suite_id / f"{key}_comparison.jpg",
        )
        _write_json(
            output / "geometry" / suite_id / f"{key}.json",
            _compact_prediction(prediction),
        )
        annotation_output = output / "annotations" / suite_id / f"{key}.json"
        annotation_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            annotation_path,
            annotation_output,
        )
        cases.append(
            {
                "id": key,
                "suite": suite_id,
                "suite_label": suite_label,
                "release": row["release"],
                "category": category,
                "reasons": reasons,
                "metrics": {
                    "gold components": int(row["gold_components"]),
                    "predicted components": int(row["prediction_components"]),
                    "matches at 0.50": int(row["component_tp_iou50"]),
                    "matches at 0.75": int(row["component_tp_iou75"]),
                    "gold regions": int(row["gold_regions"]),
                    "predicted regions": int(row["prediction_regions"]),
                    "region matches": int(row["region_tp_iou50"]),
                    "page union IoU": _number(row["page_union_iou"]),
                    "pixel precision": _number(row["page_pixel_precision"]),
                    "pixel recall": _number(row["page_pixel_recall"]),
                },
                "matches": sorted(
                    matches.get(key, []),
                    key=lambda item: (
                        item.get("match_status", ""),
                        item.get("gold_component_id", ""),
                    ),
                ),
            }
        )
    cases.sort(key=lambda case: (case["category"] == "pass", case["id"]))
    page_dir = output / "pages" / suite_id
    page_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        previous_id = cases[index - 1]["id"] if index else None
        next_id = cases[index + 1]["id"] if index + 1 < len(cases) else None
        (page_dir / f"{case['id']}.html").write_text(
            _case_html(suite_id, case, previous_id, next_id), encoding="utf-8"
        )
    summary = _read_json(evaluation / "summary.json")
    return {"id": suite_id, "label": suite_label, "summary": summary, "cases": cases}


def _index_html(suites: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    cards: list[str] = []
    for suite in suites:
        summary = suite["summary"]
        cards.append(
            f"<article class='stat'><strong>{int(summary['annotation_files'])}</strong>{_escape(suite['label'])}<span class='small'><br>component F1 {float(summary['component_f1_iou50']):.1%}</span></article>"
        )
        for case in suite["cases"]:
            metrics = case["metrics"]
            search = " ".join(
                [case["id"], suite["label"], case["release"], case["category"], *case["reasons"]]
            ).lower()
            rows.append(
                f"<tr data-suite='{_escape(suite['id'])}' data-category='{_escape(case['category'])}' data-search='{_escape(search)}'><td><a href='pages/{_escape(suite['id'])}/{_escape(case['id'])}.html'>{_escape(case['id'])}</a></td><td>{_escape(suite['label'])}<br><span class='small'>{_escape(case['release'])}</span></td><td><span class='badge'>{_escape(case['category'])}</span><br><span class='small'>{_escape('; '.join(case['reasons']))}</span></td><td>{metrics['matches at 0.50']}/{metrics['gold components']}<br><span class='small'>predicted {metrics['predicted components']}</span></td><td>{metrics['page union IoU']:.1%}</td></tr>"
            )
    options = "".join(
        f"<option value='{_escape(suite['id'])}'>{_escape(suite['label'])}</option>"
        for suite in suites
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Manual box validation</title><style>{CSS}</style></head><body><main class="shell"><nav class="nav"><a href="../index.html">Box results</a></nav><div class="eyebrow">Redaction Box CV {__version__}</div><h1>Manual geometry validation</h1><p class="lede">The curated and dense-gold collections are manually labeled development and regression sets. They test physical component recovery and reading-unit grouping; they are not presented as untouched estimates of deployment accuracy.</p><section class="stats">{''.join(cards)}</section>
<section class="callout" style="margin-bottom:18px"><b>How to read the comparison</b>Green <code>G:Rn.m</code> outlines are human labels. Red <code>Rn.m</code> outlines are detector outputs. The primary component gate is one-to-one IoU at 0.50; tighter IoU and page-union scores diagnose boundary differences.</section>
<div class="controls"><input id="query" placeholder="Search case, release, or review reason"><select id="suite"><option value="">Both collections</option>{options}</select><select id="category"><option value="">All categories</option><option>component review</option><option>grouping review</option><option>geometry review</option><option>pass</option></select><span id="visible" class="small"></span></div>
<div class="table-wrap"><table><thead><tr><th>Case</th><th>Collection</th><th>Review</th><th>Component match</th><th>Page-union IoU</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></main><script>const q=document.getElementById('query'),s=document.getElementById('suite'),c=document.getElementById('category'),v=document.getElementById('visible');function f(){{let n=0;document.querySelectorAll('tbody tr').forEach(x=>{{const ok=(!q.value||x.dataset.search.includes(q.value.toLowerCase()))&&(!s.value||x.dataset.suite===s.value)&&(!c.value||x.dataset.category===c.value);x.hidden=!ok;if(ok)n++}});v.textContent=n.toLocaleString()+' pages'}}[q,s,c].forEach(x=>x.addEventListener('input',f));f();</script></body></html>"""


def _landing_html(suites: list[dict[str, Any]]) -> str:
    curated = suites[0]["summary"]
    dense = suites[1]["summary"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Redaction Box CV results</title><style>{CSS}</style></head><body><main class="shell"><div class="eyebrow">Redaction Extract / CV branch</div><h1>Redaction box review</h1><p class="lede">Two complementary audits separate physical box detection from saved v4 answer assignment. The manual collections score detector geometry directly; the v4 audit registers paired releases and asks whether detected boxes cover exact answer words.</p>
<section class="viewer"><article class="panel"><h2>Items v4 audit</h2><p>Runs over 1,414 saved v4 items spanning 1,277 page pairs. Exact target-word boxes are localized from the later PDF text layer only after answer-blind CV detection.</p><p><a href="items_v4/index.html">Open locally generated v4 audit</a></p><p class="small">Generate it with <code>python -m box_scripts.v4_audit --workers 6 --overwrite</code>. This large derived directory is not committed.</p></article>
<article class="panel"><h2>Manual geometry validation</h2><p>Curated: {int(curated['annotation_files'])} pages, component F1 {float(curated['component_f1_iou50']):.1%}. Dense gold: {int(dense['annotation_files'])} pages, component F1 {float(dense['component_f1_iou50']):.1%}.</p><p><a href="manual_validation/index.html">Open curated and dense-gold review</a></p><p class="small">These are labeled development/regression collections, not untouched test sets.</p></article></section>
<section class="callout" style="margin-top:18px"><b>Separation of roles</b>The detector sees one page's pixels only. Manual polygons and v4/Astra text are attached afterward to evaluate geometry. Scan registration and text localization are reported separately from box-detection failures.</section></main></body></html>"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.out.resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    workspace = args.annotation_workspace.resolve()
    evaluation = args.evaluation_root.resolve()
    specs = (
        (
            "curated",
            "Curated images",
            evaluation / "manual_gold_evaluation",
            workspace / "annotations" / "manual_gold" / "curated_images",
            workspace / "annotation_collections" / "curated_images",
        ),
        (
            "dense",
            "Dense gold",
            evaluation / "dense_manual_gold_evaluation",
            workspace / "annotations" / "manual_gold" / "dense_gold_200dpi",
            workspace / "annotation_collections" / "dense_gold_200dpi",
        ),
    )
    suites = [_suite(*spec, output) for spec in specs]
    (output / "index.html").write_text(_index_html(suites), encoding="utf-8")
    landing = output.parent / "index.html"
    landing.write_text(_landing_html(suites), encoding="utf-8")
    manifest = {
        "detector_version": __version__,
        "collections": {
            suite["id"]: {
                "pages": len(suite["cases"]),
                "component_f1_iou50": suite["summary"]["component_f1_iou50"],
                "component_precision_iou50": suite["summary"]["component_precision_iou50"],
                "component_recall_iou50": suite["summary"]["component_recall_iou50"],
            }
            for suite in suites
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--evaluation-root", type=Path, required=True)
    result.add_argument("--annotation-workspace", type=Path, required=True)
    result.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    result = build(args)
    print(json.dumps(result, indent=2))
    print(f"Open: {(args.out.resolve().parent / 'index.html').as_uri()}")


if __name__ == "__main__":
    main()
