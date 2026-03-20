"""
image_handler.py
────────────────
Handles image references produced by marker_single (e.g. ![](_page_2_Figure_0.jpeg)).

Pipeline per image:
  1. VLM summary of the full figure (image only — no OCR text sent to VLM)
  2. Orientation detection: rotate image 0°/90°/180°/270°, run YOLO on each,
     pick the rotation with the most detections (no confidence threshold applied)
  3. YOLO table detector → detect individual table bboxes in best orientation
  4. Crop each detected table with padding (coords mapped back to original image)
  5. OCR each crop with Surya RecognitionPredictor → best-effort markdown table
  6. Assemble into ImageResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from PIL import Image

import config
from vlm_utils import query_vlm

# ── Lazy model cache ──────────────────────────────────────────────────────────
_detection_predictor    = None
_foundation_predictor   = None
_recognition_predictor  = None
_yolo_predictor         = None


def _get_detection_predictor():
    global _detection_predictor
    if _detection_predictor is None:
        print("  Loading Surya DetectionPredictor…")
        from surya.detection import DetectionPredictor
        _detection_predictor = DetectionPredictor()
        print("  Surya DetectionPredictor loaded.")
    return _detection_predictor


def _get_foundation_predictor():
    global _foundation_predictor
    if _foundation_predictor is None:
        print("  Loading Surya FoundationPredictor…")
        from surya.foundation import FoundationPredictor
        _foundation_predictor = FoundationPredictor()
        print("  Surya FoundationPredictor loaded.")
    return _foundation_predictor


def _get_recognition_predictor():
    global _recognition_predictor
    if _recognition_predictor is None:
        print("  Loading Surya RecognitionPredictor…")
        from surya.recognition import RecognitionPredictor
        _recognition_predictor = RecognitionPredictor(_get_foundation_predictor())
        print("  Surya RecognitionPredictor loaded.")
    return _recognition_predictor


def _get_yolo_predictor():
    """Lazy-load the YOLO table detector from config.FIGURE_PARSER."""
    global _yolo_predictor
    if _yolo_predictor is None:
        print("  Loading YOLO table detector…")
        from ultralytics import YOLO
        _yolo_predictor = YOLO(config.FIGURE_PARSER)
        print("  YOLO table detector loaded.")
    return _yolo_predictor


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TableResult:
    """OCR result for a single detected table region."""
    table_index: int
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in full-image coords
    ocr_text: str                      # Markdown table reconstructed from OCR
    flags: list[str] = field(default_factory=list)
    confidence: str = "high"           # "high" | "low"


@dataclass
class ImageResult:
    """Processing result for a single figure image."""
    image_path: Path
    figure_summary: str               # VLM summary of the whole figure
    tables: list[TableResult]
    rotation_degrees: int = 0         # winning orientation (0/90/180/270)
    flags: list[str] = field(default_factory=list)


# ── YOLO-based table detection with orientation search ────────────────────────

# The four candidate rotation angles (counter-clockwise, matching PIL convention).
_CANDIDATE_ROTATIONS = [0, 90, 180, 270]


def _rotate_image(image: Image.Image, degrees: int) -> Image.Image:
    """
    Rotate image by `degrees` counter-clockwise (PIL convention).
    expand=True keeps the full image visible after rotation.
    """
    if degrees == 0:
        return image
    return image.rotate(degrees, expand=True)


def _run_yolo(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Run the YOLO table detector on `image` (no confidence threshold).

    Returns a list of (x1, y1, x2, y2) integer bounding boxes, one per
    detected table.  All detections are kept — filtering by score is
    intentionally omitted so that the orientation vote is as sensitive
    as possible.
    """
    model = _get_yolo_predictor()
    # conf=0.01 is effectively "no threshold" while still producing valid results
    results = model(image, conf=0.01, verbose=False)

    bboxes: list[tuple[int, int, int, int]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))

    return bboxes


def _map_bbox_to_original(
    bbox: tuple[int, int, int, int],
    degrees: int,
    orig_w: int,
    orig_h: int,
) -> tuple[int, int, int, int]:
    """
    Map a bounding box from the rotated image coordinate space back to the
    original (unrotated) image coordinate space.

    PIL rotates counter-clockwise with expand=True, so the rotated canvas size
    and axis directions change depending on the angle.

      0°   → identity
      90°  → rotated (w,h) = (orig_h, orig_w)
              x_orig = y_rot,  y_orig = orig_w - x_rot - 1
    180°   → rotated (w,h) = (orig_w, orig_h)
              x_orig = orig_w - x_rot - 1,  y_orig = orig_h - y_rot - 1
    270°   → rotated (w,h) = (orig_h, orig_w)
              x_orig = orig_h - y_rot - 1,  y_orig = x_rot
    """
    x1, y1, x2, y2 = bbox

    if degrees == 0:
        return (x1, y1, x2, y2)

    if degrees == 90:
        # rotated canvas: (orig_h × orig_w)
        ox1 = y1
        oy1 = orig_w - x2
        ox2 = y2
        oy2 = orig_w - x1
        return (_clamp(ox1, orig_w), _clamp(oy1, orig_h),
                _clamp(ox2, orig_w), _clamp(oy2, orig_h))

    if degrees == 180:
        ox1 = orig_w - x2
        oy1 = orig_h - y2
        ox2 = orig_w - x1
        oy2 = orig_h - y1
        return (_clamp(ox1, orig_w), _clamp(oy1, orig_h),
                _clamp(ox2, orig_w), _clamp(oy2, orig_h))

    if degrees == 270:
        # rotated canvas: (orig_h × orig_w)
        ox1 = orig_h - y2
        oy1 = x1
        ox2 = orig_h - y1
        oy2 = x2
        return (_clamp(ox1, orig_w), _clamp(oy1, orig_h),
                _clamp(ox2, orig_w), _clamp(oy2, orig_h))

    raise ValueError(f"Unsupported rotation: {degrees}")


def _clamp(v: int, upper: int) -> int:
    return max(0, min(v, upper - 1))


def _detect_tables_with_rotation(
    image: Image.Image,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """
    Internal: run YOLO at all four orientations, return (bboxes, best_rotation).

    Bboxes are mapped back to original (unrotated) image coordinates.
    Separated from detect_tables() so the public API stays list-only while
    process_image() can still read the winning rotation angle.
    """
    orig_w, orig_h = image.size

    best_rotation = 0
    best_bboxes_rotated: list[tuple[int, int, int, int]] = []
    best_count = -1

    print("    Orientation search: ", end="", flush=True)
    for degrees in _CANDIDATE_ROTATIONS:
        rotated = _rotate_image(image, degrees)
        bboxes = _run_yolo(rotated)
        count = len(bboxes)
        print(f"{degrees}°→{count} ", end="", flush=True)

        if count > best_count:
            best_count = count
            best_rotation = degrees
            best_bboxes_rotated = bboxes

    print(f"| best={best_rotation}° ({best_count} detection(s))")

    bboxes_orig = [
        _map_bbox_to_original(b, best_rotation, orig_w, orig_h)
        for b in best_bboxes_rotated
    ]

    return bboxes_orig, best_rotation


def detect_tables(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Detect table bounding boxes in `image` using the YOLO model.

    Tries all four orientations (0°/90°/180°/270°) and picks the one that
    yields the most detections.  Returns bounding boxes in the original
    (unrotated) image coordinate space.
    """
    bboxes, _ = _detect_tables_with_rotation(image)
    return bboxes


# ── Crop with padding ─────────────────────────────────────────────────────────

def crop_table(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    padding: int = config.TABLE_CROP_PADDING,
) -> Image.Image:
    """Crop a table region from the full image with optional padding."""
    w, h = image.size
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)
    return image.crop((x1, y1, x2, y2))


# ── OCR each table crop ───────────────────────────────────────────────────────

def _group_into_rows(text_lines: list) -> list[list]:
    """
    Group text lines into rows by vertical overlap.

    Lines are sorted by their top-y coordinate. A line joins the current row
    if it overlaps vertically with the row's running bounding box; otherwise a
    new row starts.  Each row is then sorted left-to-right by x position.
    """
    if not text_lines:
        return []

    by_top = sorted(text_lines, key=lambda l: l.bbox[1])
    rows: list[list] = [[by_top[0]]]
    current_bottom = by_top[0].bbox[3]

    for line in by_top[1:]:
        if line.bbox[1] < current_bottom:          # vertical overlap → same row
            rows[-1].append(line)
            current_bottom = max(current_bottom, line.bbox[3])
        else:                                       # gap → new row
            rows.append([line])
            current_bottom = line.bbox[3]

    for row in rows:
        row.sort(key=lambda l: l.bbox[0])

    return rows


def _text_lines_to_markdown(text_lines: list) -> str:
    """Convert a flat list of recognized text lines to a markdown table string."""
    rows = _group_into_rows(text_lines)
    if not rows:
        return ""

    n_cols = max(len(row) for row in rows)
    md: list[str] = []
    for i, row in enumerate(rows):
        cells = [l.text.strip() for l in row]
        cells += [""] * (n_cols - len(cells))   # pad short rows
        md.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md.append("|" + "|".join([" --- "] * n_cols) + "|")

    return "\n".join(md)


def ocr_table(crop: Image.Image) -> str:
    """
    OCR a cropped table image.

    Uses Surya's DetectionPredictor to locate text lines and
    RecognitionPredictor to read them, then reconstructs a markdown table
    by grouping lines into rows based on vertical overlap.

    Returns an empty string if no text is found.
    """
    det_pred = _get_detection_predictor()
    rec_pred = _get_recognition_predictor()

    results = rec_pred([crop], det_predictor=det_pred)
    text_lines = results[0].text_lines

    if not text_lines:
        return ""

    return _text_lines_to_markdown(text_lines)


# ── VLM summary of the full figure ───────────────────────────────────────────

_FIGURE_PROMPT = (
    "This is a figure from an environmental monitoring document. "
    "Provide a concise summary of what this figure shows, "
    "including any key data, labels, or spatial patterns visible."
)


def _vlm_summarize(image_path: Path, client: OpenAI, model: str, prompt: str) -> str:
    history = query_vlm(
        client=client,
        model=model,
        query=prompt,
        image_path=image_path,
        max_tokens=512,
        temperature=0.1,
    )
    return history[-1]["content"].strip()


# ── Main entry point ──────────────────────────────────────────────────────────

def process_image(
    image_path: Path,
    temp_dir: Path,
    client: OpenAI,
    model: str,
) -> ImageResult:
    """
    Full pipeline for one figure image reference from Marker output.

    1. VLM summary of the full figure (image only — no OCR text sent to VLM)
    2. Orientation detection: run YOLO at 0°/90°/180°/270°, pick the rotation
       with the most detections
    3. YOLO table detection on the best orientation → bboxes in original coords
    4. For each table: crop → OCR with Surya RecognitionPredictor → markdown
    5. Return ImageResult
    """
    image = Image.open(image_path).convert("RGB")
    image_flags: list[str] = []
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ── Full-figure VLM summary ───────────────────────────────────────────────
    print(f"    Summarizing figure {image_path.name}…")
    try:
        figure_summary = _vlm_summarize(image_path, client, model, _FIGURE_PROMPT)
    except Exception as exc:
        figure_summary = "(VLM unavailable)"
        image_flags.append(f"Figure summary error: {exc}")

    # ── Table detection (YOLO + orientation search) ───────────────────────────
    print(f"    Detecting tables in {image_path.name}…")
    try:
        bboxes, rotation = _detect_tables_with_rotation(image)
    except Exception as exc:
        image_flags.append(f"YOLO detection error: {exc}")
        return ImageResult(
            image_path=image_path,
            figure_summary=figure_summary,
            tables=[],
            rotation_degrees=0,
            flags=image_flags,
        )

    if not bboxes:
        image_flags.append("No tables detected — manual review recommended.")
        return ImageResult(
            image_path=image_path,
            figure_summary=figure_summary,
            tables=[],
            rotation_degrees=rotation,
            flags=image_flags,
        )

    print(f"    {len(bboxes)} table(s) detected (image rotated {rotation}°).")

    # ── OCR — crop from the *original* (unrotated) image ─────────────────────
    # Bboxes were already mapped back to original coordinates, so we crop from
    # the unrotated image then rotate each crop to match the winning orientation
    # so that the OCR engine sees upright text.
    tables: list[TableResult] = []

    for idx, bbox in enumerate(bboxes):
        print(f"    Table {idx + 1}/{len(bboxes)}: OCR…", end=" ", flush=True)

        crop = crop_table(image, bbox)

        if rotation != 0:
            crop = _rotate_image(crop, rotation)

        try:
            ocr_text = ocr_table(crop)
            flags: list[str] = []
            confidence = "high"
        except Exception as exc:
            ocr_text = "(OCR failed)"
            flags = [f"OCR error: {exc}"]
            confidence = "low"

        print("done" if confidence != "low" else "failed")

        tables.append(TableResult(
            table_index=idx,
            bbox=bbox,
            ocr_text=ocr_text,
            flags=flags,
            confidence=confidence,
        ))

    return ImageResult(
        image_path=image_path,
        figure_summary=figure_summary,
        tables=tables,
        rotation_degrees=rotation,
        flags=image_flags,
    )


def image_result_to_markdown(result: ImageResult) -> str:
    """Convert an ImageResult to a markdown string for inclusion in final output."""
    rotation_note = (
        f", rotated {result.rotation_degrees}°" if result.rotation_degrees else ""
    )
    lines: list[str] = [
        f"<!-- IMAGE: {result.image_path.name}"
        f" | {len(result.tables)} table(s) detected{rotation_note} -->"
    ]

    for f in result.flags:
        lines.append(f"> ⚠ {f}")

    lines.append("")
    lines.append("**Figure Summary:**")
    lines.append(result.figure_summary)

    if result.tables:
        lines.append("")
        lines.append(f"**Detected Tables ({len(result.tables)}):**")
        for table in result.tables:
            lines.append("")
            lines.append(f"*Table {table.table_index + 1}:*")
            lines.append(table.ocr_text or "(no text recognized)")
            for flag in table.flags:
                lines.append(f"> ⚠ {flag}")

    return "\n".join(lines)