#!/usr/bin/env python3
"""Build a size-bounded static web edition of the local items-v4 audit."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from typing import Any

import cv2
from tqdm import tqdm


EXPECTED_ITEMS = 1_414
EXPECTED_PAGE_PAIRS = 1_277
MAX_IMAGE_DIMENSION = 1_400
JPEG_QUALITY = 72

RAW_CONTROLS = re.compile(
    r'<div class="image-tools"><button class="active" data-view="overlay" '
    r'data-image="(?P<release>earlier|later)">Detection \+ target</button>'
    r'<button data-view="raw" data-image="(?P=release)">Original pixels</button></div>'
)
VIEW_SCRIPT = re.compile(
    r"\n<script>document\.querySelectorAll\('\[data-view\]'\).*?</script>",
    re.DOTALL,
)
DATA_IMAGE_ATTRIBUTES = re.compile(
    r' data-overlay="[^"]+" data-raw="[^"]+"'
)
WEB_NOTICE = (
    '<section class="callout" style="margin-bottom:18px">'
    '<b>Web edition.</b>This page retains the complete scored case and its '
    'optimized labeled overlays. Duplicate raw-scan images are omitted from '
    'the online bundle; the full-resolution local audit remains authoritative.'
    '</section>'
)


class _ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del tag
        for key, value in attrs:
            if key not in {"href", "src"} or not value:
                continue
            if value.startswith(("#", "http:", "https:", "data:", "mailto:")):
                continue
            self.references.append(value.split("#", 1)[0].split("?", 1)[0])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_html(text: str) -> str:
    text = RAW_CONTROLS.sub(
        '<div class="image-tools"><span class="small">'
        'Optimized detection + target overlay</span></div>',
        text,
    )
    text = DATA_IMAGE_ATTRIBUTES.sub("", text)
    text = VIEW_SCRIPT.sub("", text)
    text = text.replace(
        '<a href="../manual_validation/index.html">Manual validation</a>',
        '<a href="../index.html">Manual validation</a>',
    )
    return text.replace('<main class="shell">', f'<main class="shell">{WEB_NOTICE}', 1)


def _sanitize_geometry(source: Path, destination: Path) -> None:
    payload = _read_json(source)
    source_path = payload.get("source_path")
    if source_path:
        payload["source_path"] = Path(str(source_path)).name
    _write_json(destination, payload)


def _sanitize_page_results(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8") as source_handle, destination.open(
        "w", encoding="utf-8"
    ) as destination_handle:
        for line in source_handle:
            payload = json.loads(line)
            for key in ("redacted_pdf", "later_pdf"):
                if payload.get(key):
                    payload[key] = Path(str(payload[key])).name
            payload.pop("earlier_raw_asset", None)
            payload.pop("later_raw_asset", None)
            payload["raw_assets_omitted_from_web_edition"] = True
            destination_handle.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _copy_overlay(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read overlay: {source}")
    scale = min(1.0, MAX_IMAGE_DIMENSION / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    parameters = [
        cv2.IMWRITE_JPEG_QUALITY,
        JPEG_QUALITY,
        cv2.IMWRITE_JPEG_PROGRESSIVE,
        1,
        cv2.IMWRITE_JPEG_OPTIMIZE,
        1,
    ]
    if not cv2.imwrite(str(destination), image, parameters):
        raise RuntimeError(f"Could not write overlay: {destination}")


def _validate_links(root: Path) -> dict[str, int]:
    missing: list[tuple[str, str]] = []
    html_count = 0
    reference_count = 0
    for html_path in sorted(root.rglob("*.html")):
        html_count += 1
        parser = _ReferenceParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        for reference in parser.references:
            reference_count += 1
            if reference and not (html_path.parent / reference).resolve().exists():
                missing.append((str(html_path.relative_to(root)), reference))
    if missing:
        preview = "\n".join(f"{page}: {reference}" for page, reference in missing[:20])
        raise RuntimeError(f"Web bundle has {len(missing)} missing links:\n{preview}")
    return {
        "html_pages": html_count,
        "local_references": reference_count,
        "missing_references": 0,
    }


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build(source: Path, output: Path, overwrite: bool) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    summary_path = source / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing completed audit summary: {summary_path}")
    summary = _read_json(summary_path)
    if not summary.get("complete_scope"):
        raise SystemExit("Refusing to publish an incomplete items-v4 audit")
    if summary.get("items_processed") != EXPECTED_ITEMS:
        raise SystemExit(f"Expected {EXPECTED_ITEMS} items, found {summary.get('items_processed')}")
    if summary.get("page_pairs_processed") != EXPECTED_PAGE_PAIRS:
        raise SystemExit(
            f"Expected {EXPECTED_PAGE_PAIRS} page pairs, "
            f"found {summary.get('page_pairs_processed')}"
        )
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists; add --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name in (
        "summary.json",
        "item_results.csv",
        "fragment_results.csv",
        "errors.json",
    ):
        shutil.copy2(source / name, output / name)
    _sanitize_page_results(source / "page_results.jsonl", output / "page_results.jsonl")

    run_config = _read_json(source / "run_config.json")
    for key in ("items", "astra"):
        if run_config.get(key):
            run_config[key] = Path(str(run_config[key])).name
    run_config["docs_root"] = "not bundled in web edition"
    _write_json(output / "run_config.json", run_config)

    index_text = _rewrite_html((source / "index.html").read_text(encoding="utf-8"))
    (output / "index.html").write_text(index_text, encoding="utf-8")

    page_sources = sorted((source / "pages").glob("*.html"))
    if len(page_sources) != EXPECTED_PAGE_PAIRS:
        raise SystemExit(f"Expected {EXPECTED_PAGE_PAIRS} case pages, found {len(page_sources)}")
    pages_output = output / "pages"
    pages_output.mkdir()
    for path in page_sources:
        rewritten = _rewrite_html(path.read_text(encoding="utf-8"))
        (pages_output / path.name).write_text(rewritten, encoding="utf-8")

    geometry_sources = sorted((source / "geometry").glob("*.json"))
    expected_geometry = EXPECTED_PAGE_PAIRS * 2
    if len(geometry_sources) != expected_geometry:
        raise SystemExit(
            f"Expected {expected_geometry} geometry files, found {len(geometry_sources)}"
        )
    for path in tqdm(geometry_sources, desc="Sanitize geometry", unit="file"):
        _sanitize_geometry(path, output / "geometry" / path.name)

    overlay_sources = sorted(
        path
        for path in (source / "assets").glob("*.jpg")
        if not path.name.endswith("_raw.jpg")
    )
    if len(overlay_sources) != expected_geometry:
        raise SystemExit(
            f"Expected {expected_geometry} overlays, found {len(overlay_sources)}"
        )
    for path in tqdm(overlay_sources, desc="Optimize overlays", unit="image"):
        _copy_overlay(path, output / "assets" / path.name)

    readme = f"""# Items v4 Web Edition

This static bundle contains every one of the {EXPECTED_ITEMS:,} scored items
across {EXPECTED_PAGE_PAIRS:,} source-page pairs. It retains the searchable
index, all case pages, optimized labeled overlays, linked detector geometry,
summary counts, and item-level results.

To keep the GitHub Pages site below its size limit, duplicate raw-scan previews
are omitted. The local `box_results/items_v4/` audit remains the authoritative
full-resolution output. Web overlays are bounded to {MAX_IMAGE_DIMENSION:,}
pixels on their longest side and encoded as progressive JPEG at quality
{JPEG_QUALITY}.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    link_report = _validate_links(output)
    manifest = {
        "bundle": "items-v4-web-edition",
        "source_summary_sha256": _sha256(summary_path),
        "items": EXPECTED_ITEMS,
        "page_pairs": EXPECTED_PAGE_PAIRS,
        "case_html_pages": len(page_sources),
        "geometry_files": len(geometry_sources),
        "overlay_images": len(overlay_sources),
        "raw_images_omitted": EXPECTED_PAGE_PAIRS * 2,
        "max_image_dimension": MAX_IMAGE_DIMENSION,
        "jpeg_quality": JPEG_QUALITY,
        "link_validation": link_report,
    }
    _write_json(output / "web_bundle_manifest.json", manifest)
    manifest["bundle_bytes"] = _directory_bytes(output)
    _write_json(output / "web_bundle_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build(args.source, args.output, args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
