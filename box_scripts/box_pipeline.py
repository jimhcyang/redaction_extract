from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


@dataclass
class PDFPair:
    pair_key: str
    unredacted_numeric_id: str
    redacted_doc_id: str
    row: list[str]
    row_index_1based: int
    redacted_pdf: Path
    unredacted_pdf: Path


@dataclass
class PairCollectionStats:
    rows_total: int = 0
    rows_missing_required_fields: int = 0
    rows_invalid_unredacted_id_format: int = 0
    rows_missing_redacted_pdf: int = 0
    rows_missing_unredacted_pdf: int = 0
    rows_duplicate_unredacted_id: int = 0
    rows_duplicate_redacted_id: int = 0
    rows_valid: int = 0


@dataclass
class InputRecord:
    item_key: str
    source_path: Path
    rendered_image_path: Path
    source_kind: str
    page_no_1based: int | None
    pair_key: str | None


def _progress(iterable: Any, *, total: int | None, desc: str) -> Any:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.2)


def _clean_filename_component(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "x"


def _normalize_csv_row(row: list[str], width: int = 11) -> list[str]:
    out = list(row)
    if len(out) < width:
        out.extend([""] * (width - len(out)))
    return out


def _parse_unredacted_id(raw: str) -> int | None:
    s = str(raw).strip()
    if not s or not s.isdigit():
        return None
    if len(s) > 8:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _canonical_unredacted_pdf_path(unredacted_id_int: int, unredacted_dir: Path) -> Path:
    return unredacted_dir / f"cib_{unredacted_id_int:08d}.pdf"


def collect_pdf_pairs_with_stats(
    csv_path: Path,
    redacted_dir: Path,
    unredacted_dir: Path,
    max_pairs: int | None = None,
) -> tuple[list[PDFPair], PairCollectionStats]:
    pairs: list[PDFPair] = []
    stats = PairCollectionStats()
    seen_unred_ids: set[str] = set()
    seen_red_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, raw_row in enumerate(reader, start=1):
            stats.rows_total += 1
            row = _normalize_csv_row(raw_row)
            unred_id_raw = str(row[1]).strip()
            red_doc_id = str(row[5]).strip()

            if not unred_id_raw or not red_doc_id:
                stats.rows_missing_required_fields += 1
                continue

            unred_id_int = _parse_unredacted_id(unred_id_raw)
            if unred_id_int is None:
                stats.rows_invalid_unredacted_id_format += 1
                continue

            unred_id_norm = str(unred_id_int)
            if unred_id_norm in seen_unred_ids:
                stats.rows_duplicate_unredacted_id += 1
                continue
            if red_doc_id in seen_red_ids:
                stats.rows_duplicate_redacted_id += 1
                continue

            red_pdf = redacted_dir / f"{red_doc_id}.pdf"
            if not red_pdf.exists():
                stats.rows_missing_redacted_pdf += 1
                continue

            unred_pdf = _canonical_unredacted_pdf_path(unred_id_int, unredacted_dir)
            if not unred_pdf.exists():
                stats.rows_missing_unredacted_pdf += 1
                continue

            seen_unred_ids.add(unred_id_norm)
            seen_red_ids.add(red_doc_id)
            pair_key = _clean_filename_component(f"{unred_id_norm}_{red_doc_id}")
            pairs.append(
                PDFPair(
                    pair_key=pair_key,
                    unredacted_numeric_id=unred_id_norm,
                    redacted_doc_id=red_doc_id,
                    row=row,
                    row_index_1based=i,
                    redacted_pdf=red_pdf,
                    unredacted_pdf=unred_pdf,
                )
            )
            stats.rows_valid += 1
            if max_pairs is not None and len(pairs) >= max_pairs:
                break

    return pairs, stats


def _edge_density(edges: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 3) -> float:
    xa, ya, xb, yb = x1 + pad, y1 + pad, x2 - pad, y2 - pad
    if xb <= xa or yb <= ya:
        return 1.0
    roi = edges[ya:yb, xa:xb]
    if roi.size == 0:
        return 1.0
    return float(np.mean(roi > 0))


def _binarize_dark(gray: np.ndarray) -> np.ndarray:
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hard_inv = ((gray < 170).astype(np.uint8) * 255)
    return cv2.bitwise_or(otsu_inv, hard_inv)


def _inner_region(x1: int, y1: int, x2: int, y2: int, pad: int) -> tuple[int, int, int, int]:
    return x1 + pad, y1 + pad, x2 - pad, y2 - pad


def _interior_stats(gray: np.ndarray, dark: np.ndarray, box: tuple[int, int, int, int], pad: int = 3) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    xa, ya, xb, yb = _inner_region(x1, y1, x2, y2, pad)
    if xb <= xa or yb <= ya:
        return 0.0, 1.0
    interior_gray = gray[ya:yb, xa:xb]
    interior_dark = dark[ya:yb, xa:xb]
    if interior_gray.size == 0:
        return 0.0, 1.0
    return float(interior_gray.mean()), float(np.mean(interior_dark > 0))


def _side_dark_frac(dark: np.ndarray, box: tuple[int, int, int, int], thickness: int = 2) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    t = max(1, thickness)
    top = dark[y1:y1 + t, x1:x2]
    bot = dark[y2 - t:y2, x1:x2]
    lef = dark[y1:y2, x1:x1 + t]
    rig = dark[y1:y2, x2 - t:x2]
    if min(top.size, bot.size, lef.size, rig.size) == 0:
        return 0.0, 0.0, 0.0, 0.0

    def frac(mask: np.ndarray) -> float:
        return float(np.mean(mask > 0))

    return frac(top), frac(bot), frac(lef), frac(rig)


def _split_stacked_boxes(h_lines: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    roi = h_lines[y1:y2, x1:x2]
    if roi.size == 0:
        return [box]

    row_strength = (roi > 0).mean(axis=1)
    seams = np.where(row_strength > 0.60)[0]
    seams = seams[(seams > 4) & (seams < (y2 - y1 - 5))]
    if seams.size == 0:
        return [box]

    groups = []
    start = int(seams[0])
    prev = int(seams[0])
    for s in seams[1:]:
        s = int(s)
        if s == prev + 1:
            prev = s
            continue
        groups.append((start, prev))
        start = s
        prev = s
    groups.append((start, prev))

    cuts = [y1]
    for a, b in groups:
        cuts.append(y1 + int(round((a + b) / 2)))
    cuts.append(y2)

    parts: list[tuple[int, int, int, int]] = []
    for ya, yb in zip(cuts[:-1], cuts[1:]):
        if yb - ya >= 8:
            parts.append((x1, ya, x2, yb))
    return parts if parts else [box]


def _is_redaction_like(gray: np.ndarray, dark: np.ndarray, edges: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w < 30 or h < 10:
        return False
    if w * h < 300:
        return False

    interior_mean, interior_dark = _interior_stats(gray, dark, box, pad=3)
    if interior_mean < 220:
        return False
    if interior_dark > 0.12:
        return False
    if _edge_density(edges, x1, y1, x2, y2, pad=3) > 0.08:
        return False
    topf, botf, leff, rigf = _side_dark_frac(dark, box, thickness=2)
    if min(topf, botf, leff, rigf) < 0.08:
        return False
    return True


def detect_redaction_boxes_with_artifacts(gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict[str, np.ndarray]]:
    dark = _binarize_dark(gray)
    h, w = gray.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, w // 10), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 4)))
    h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel, iterations=1)
    lines = cv2.bitwise_or(h_lines, v_lines)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        candidate = (x, y, x + bw, y + bh)
        for sub in _split_stacked_boxes(h_lines, candidate):
            if _is_redaction_like(gray, dark, edges, sub):
                line_boxes.append(sub)

    contour_mask = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours2, _ = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours2:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4 or len(approx) > 8:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 10:
            continue
        rect_area = float(bw * bh)
        if rect_area <= 0:
            continue
        fill_ratio = float(cv2.contourArea(cnt) / rect_area)
        if fill_ratio < 0.55:
            continue
        box = (x, y, x + bw, y + bh)
        if _is_redaction_like(gray, dark, edges, box):
            contour_boxes.append(box)

    boxes = _dedupe_boxes(line_boxes + contour_boxes)
    return boxes, {
        "dark_mask": dark,
        "lines_mask": lines,
        "contours_mask": contour_mask,
        "edges": edges,
    }


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], margin: int = 2) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 + margin and oy1 <= iy1 + margin and ox2 >= ix2 - margin and oy2 >= iy2 - margin


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    clean = [tuple(int(v) for v in box) for box in boxes if _area(box) > 0]
    clean.sort(key=_area)
    kept: list[tuple[int, int, int, int]] = []
    for box in clean:
        skip = False
        for prev in kept:
            if _iou(box, prev) >= 0.65:
                skip = True
                break
            if _contains(box, prev, margin=2):
                skip = True
                break
        if not skip:
            kept.append(box)
    kept.sort(key=lambda b: (b[1], b[0]))
    return kept


def render_pdf_to_images(pdf_path: Path, out_dir: Path, prefix: str, dpi: int, max_pages: int | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF rendering backend missing. Install pypdfium2.") from exc

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(doc)
        if max_pages is not None:
            page_count = min(page_count, max(0, int(max_pages)))
        images: list[Path] = []
        scale = max(0.1, float(dpi) / 72.0)
        for page_index in range(page_count):
            page = doc.get_page(page_index)
            try:
                bmp = page.render(scale=scale)
                pil = bmp.to_pil()
                out_path = out_dir / f"{prefix}_p{page_index + 1:04d}.png"
                pil.save(out_path)
                images.append(out_path)
            finally:
                page.close()
        return images
    finally:
        doc.close()


def iter_input_records(
    *,
    input_path: Path | None,
    docs_root: Path | None,
    docs_subdir: str,
    out_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    source_kind: str,
    dpi: int,
    max_files: int | None,
    max_pages: int | None,
) -> tuple[list[InputRecord], dict[str, Any]]:
    records: list[InputRecord] = []
    stats: dict[str, Any] = {"mode": None}
    render_root = out_root / "_rendered_inputs"
    render_root.mkdir(parents=True, exist_ok=True)

    if input_path is not None:
        stats["mode"] = "input_path"
        files: list[Path] = []
        if input_path.is_file():
            files = [input_path]
        elif input_path.is_dir():
            files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS))
        else:
            raise SystemExit(f"Input path not found: {input_path}")
        if max_files is not None:
            files = files[: max(0, int(max_files))]

        for src in files:
            suffix = src.suffix.lower()
            if suffix in IMAGE_EXTS:
                item_key = _clean_filename_component(src.stem)
                records.append(InputRecord(item_key=item_key, source_path=src, rendered_image_path=src, source_kind="image", page_no_1based=None, pair_key=None))
            elif suffix in PDF_EXTS:
                pdf_render_dir = render_root / _clean_filename_component(src.stem)
                pages = render_pdf_to_images(src, pdf_render_dir, _clean_filename_component(src.stem), dpi=dpi, max_pages=max_pages)
                for page_idx, image_path in enumerate(pages, start=1):
                    item_key = _clean_filename_component(f"{src.stem}_p{page_idx:04d}")
                    records.append(InputRecord(item_key=item_key, source_path=src, rendered_image_path=image_path, source_kind="pdf", page_no_1based=page_idx, pair_key=None))
        stats["input_file_count"] = len(files)
        stats["record_count"] = len(records)
        return records, stats

    if docs_root is None:
        raise SystemExit("Provide either --input or --docs_root.")

    stats["mode"] = "docs_root"
    pairs, pair_stats = collect_pdf_pairs_with_stats(
        csv_path=docs_root / csv_name,
        redacted_dir=docs_root / redacted_dir_name,
        unredacted_dir=docs_root / unredacted_dir_name,
        max_pairs=None,
    )
    stats["pair_stats"] = pair_stats.__dict__
    if max_files is not None:
        pairs = pairs[: max(0, int(max_files))]

    for pair in pairs:
        if source_kind in {"redacted", "both"}:
            pdf_path = pair.redacted_pdf
            pdf_render_dir = render_root / pair.pair_key / "redacted"
            pages = render_pdf_to_images(pdf_path, pdf_render_dir, f"{pair.pair_key}_redacted", dpi=dpi, max_pages=max_pages)
            for page_idx, image_path in enumerate(pages, start=1):
                item_key = _clean_filename_component(f"{pair.pair_key}_redacted_p{page_idx:04d}")
                records.append(InputRecord(item_key=item_key, source_path=pdf_path, rendered_image_path=image_path, source_kind="redacted_pdf", page_no_1based=page_idx, pair_key=pair.pair_key))
        if source_kind in {"unredacted", "both"}:
            pdf_path = pair.unredacted_pdf
            pdf_render_dir = render_root / pair.pair_key / "unredacted"
            pages = render_pdf_to_images(pdf_path, pdf_render_dir, f"{pair.pair_key}_unredacted", dpi=dpi, max_pages=max_pages)
            for page_idx, image_path in enumerate(pages, start=1):
                item_key = _clean_filename_component(f"{pair.pair_key}_unredacted_p{page_idx:04d}")
                records.append(InputRecord(item_key=item_key, source_path=pdf_path, rendered_image_path=image_path, source_kind="unredacted_pdf", page_no_1based=page_idx, pair_key=pair.pair_key))

    stats["record_count"] = len(records)
    return records, stats


def process_record(record: InputRecord, *, out_root: Path, save_debug_masks: bool) -> dict[str, Any]:
    gray = cv2.imread(str(record.rendered_image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {record.rendered_image_path}")

    boxes, artifacts = detect_redaction_boxes_with_artifacts(gray)
    item_dir = out_root / record.item_key
    item_dir.mkdir(parents=True, exist_ok=True)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, f"R{idx}", (x1 + 2, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    source_copy_path = item_dir / f"{record.item_key}.source.png"
    overlay_path = item_dir / f"{record.item_key}.redaction_boxes.png"
    metadata_path = item_dir / f"{record.item_key}.redaction_boxes.json"
    cv2.imwrite(str(source_copy_path), gray)
    cv2.imwrite(str(overlay_path), overlay)

    debug_paths: dict[str, str] = {}
    if save_debug_masks:
        debug_dir = item_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for key, mask in artifacts.items():
            out_path = debug_dir / f"{key}.png"
            cv2.imwrite(str(out_path), mask)
            debug_paths[key] = str(out_path)

    payload = {
        "item_key": record.item_key,
        "source_path": str(record.source_path),
        "rendered_image_path": str(record.rendered_image_path),
        "source_kind": record.source_kind,
        "pair_key": record.pair_key,
        "page_no_1based": record.page_no_1based,
        "image_size_wh": [int(gray.shape[1]), int(gray.shape[0])],
        "redaction_box_count": len(boxes),
        "redaction_boxes_xyxy": [list(map(int, box)) for box in boxes],
        "output_files": {
            "source_png": str(source_copy_path),
            "overlay_png": str(overlay_path),
            "metadata_json": str(metadata_path),
            "debug_masks": debug_paths,
        },
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_box_pipeline(
    *,
    out_root: Path,
    input_path: Path | None = None,
    docs_root: Path | None = None,
    docs_subdir: str = "redacted_pdfs",
    source_kind: str = "redacted",
    csv_name: str = "cibcia.csv",
    redacted_dir_name: str = "redacted_pdfs",
    unredacted_dir_name: str = "unredacted_pdfs",
    dpi: int = 300,
    max_files: int | None = None,
    max_pages: int | None = None,
    save_debug_masks: bool = False,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    records, stats = iter_input_records(
        input_path=input_path,
        docs_root=docs_root,
        docs_subdir=docs_subdir,
        out_root=out_root,
        csv_name=csv_name,
        redacted_dir_name=redacted_dir_name,
        unredacted_dir_name=unredacted_dir_name,
        source_kind=source_kind,
        dpi=dpi,
        max_files=max_files,
        max_pages=max_pages,
    )
    manifest_rows: list[dict[str, Any]] = []
    for record in _progress(records, total=len(records), desc="Detect redaction boxes"):
        try:
            manifest_rows.append(process_record(record, out_root=out_root, save_debug_masks=save_debug_masks))
        except Exception as exc:
            manifest_rows.append({
                "item_key": record.item_key,
                "source_path": str(record.source_path),
                "rendered_image_path": str(record.rendered_image_path),
                "error": str(exc),
            })

    manifest_json = out_root / "box_manifest.json"
    manifest_jsonl = out_root / "box_manifest.jsonl"
    summary_json = out_root / "box_summary.json"
    manifest_json.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    success_count = sum(1 for row in manifest_rows if "redaction_box_count" in row)
    error_count = len(manifest_rows) - success_count
    summary = {
        "stats": stats,
        "record_count": len(records),
        "success_count": success_count,
        "error_count": error_count,
        "manifest_json": str(manifest_json),
        "manifest_jsonl": str(manifest_jsonl),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
