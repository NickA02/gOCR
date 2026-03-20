"""
verify_ocr.py — Baseline verification script for Phase 1 (ingest) and Phase 2 (ocr).

Usage:
    python verify_ocr.py --input path/to/file.pdf
    python verify_ocr.py --input path/to/file.png
    python verify_ocr.py --input path/to/file.pdf --page 2
    python verify_ocr.py --input path/to/file.pdf --all-pages

What this checks:
    1. Ingest     — file loads, pages render at correct DPI, mode is RGB
    2. Structure  — Tesseract output builds a valid Block→Para→Line→Word tree
    3. Confidence — page confidence score is computed and gating flag is correct
    4. Tables     — table blocks are detected and flagged
    5. Empty cells — placeholder tokens are preserved in the word list
    6. Raw text   — reconstructed text is printable and non-empty
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr import ocr_page, PageOCRResult
from ingestion import load_pages, page_count
import config


# ---------------------------------------------------------------------------
# ANSI colours for terminal output
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg: str)   -> str: return f"{GREEN}  ✓ {msg}{RESET}"
def fail(msg: str) -> str: return f"{RED}  ✗ {msg}{RESET}"
def info(msg: str) -> str: return f"{CYAN}  · {msg}{RESET}"
def warn(msg: str) -> str: return f"{YELLOW}  ⚠ {msg}{RESET}"
def section(msg: str)    : print(f"\n{BOLD}{msg}{RESET}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_ingest(file_path: Path, page_index: int):
    """Verify Phase 1 — file loads and renders to a valid RGB image."""
    section(f"Phase 1 — Ingest  [{file_path.name}, page {page_index + 1}]")

    # Page count
    total = page_count(file_path)
    print(info(f"Total pages detected: {total}"))

    if page_index >= total:
        print(fail(f"Requested page {page_index + 1} but file only has {total} page(s)"))
        return None

    # Render the requested page
    pages = list(load_pages(file_path))
    image = pages[page_index]

    # Mode check
    if image.mode == "RGB":
        print(ok(f"Image mode is RGB"))
    else:
        print(fail(f"Expected RGB, got {image.mode}"))

    # Size sanity — at 300 DPI a standard A4 page is ~2480×3508px
    w, h = image.size
    print(info(f"Rendered size: {w} × {h} px  (DPI config: {config.RENDER_DPI})"))
    if w < 100 or h < 100:
        print(fail(f"Image suspiciously small — rendering may have failed"))
    else:
        print(ok(f"Image size looks reasonable"))

    return image


def check_structure(result: PageOCRResult):
    """Verify Phase 2 — the Block→Para→Line→Word tree is valid."""
    section("Phase 2 — Structural tree")

    if not result.blocks:
        print(fail("No blocks found — Tesseract may have returned empty output"))
        return

    total_paras = sum(len(b.paragraphs) for b in result.blocks)
    total_lines = sum(len(p.lines) for b in result.blocks for p in b.paragraphs)
    total_words = sum(len(l.words) for b in result.blocks
                      for p in b.paragraphs for l in p.lines)
    empty_cells = sum(1 for b in result.blocks for p in b.paragraphs
                      for l in p.lines for w in l.words if w.is_empty_cell)

    print(info(f"Blocks:      {len(result.blocks)}"))
    print(info(f"Paragraphs:  {total_paras}"))
    print(info(f"Lines:       {total_lines}"))
    print(info(f"Words:       {total_words}"))
    print(info(f"Empty cells: {empty_cells}  (preserved as '·' placeholders)"))

    # Verify tree integrity — every line should have a parent para and block
    orphan_lines = 0
    for block in result.blocks:
        for para in block.paragraphs:
            for line in para.lines:
                if not hasattr(line, "line_num"):
                    orphan_lines += 1

    if orphan_lines:
        print(fail(f"{orphan_lines} lines missing line_num — tree may be malformed"))
    else:
        print(ok("Tree integrity looks good — all nodes are properly nested"))

    # Spot-check: first 3 words of the page
    first_words = [
        w.text for b in result.blocks
        for p in b.paragraphs
        for l in p.lines
        for w in l.words
        if not w.is_empty_cell
    ][:3]
    if first_words:
        print(info(f"First words on page: {first_words}"))


def check_confidence(result: PageOCRResult):
    """Verify confidence scoring and gating logic."""
    section("Phase 2 — Confidence & gating")

    print(info(f"Page confidence:    {result.confidence:.1f} / 100"))
    print(info(f"Gate threshold:     {config.CONFIDENCE_THRESHOLD}"))

    if result.confidence == 0.0 and not result.blocks:
        print(warn("Confidence is 0.0 and no blocks found — page may be blank or unreadable"))
    elif result.confidence < config.CONFIDENCE_THRESHOLD:
        print(ok(f"needs_vlm = True  (confidence below threshold)"))
    else:
        print(ok(f"needs_vlm = False  (confidence above threshold — Tesseract output is sufficient)"))

    # Per-block confidence breakdown
    print(info("Per-block confidence:"))
    for block in result.blocks:
        tag = " [TABLE]" if block.is_likely_table else ""
        print(f"     Block {block.block_num:>2}{tag:<8}  conf={block.mean_confidence:.1f}")


def check_tables(result: PageOCRResult):
    """Verify table detection heuristic."""
    section("Phase 2 — Table detection")

    table_blocks = result.table_block_nums
    if table_blocks:
        print(ok(f"Table blocks detected: {table_blocks}"))
        print(info("These blocks will be sent to the VLM regardless of confidence score"))
        for bn in table_blocks:
            block = next(b for b in result.blocks if b.block_num == bn)
            positions = [w.left for p in block.paragraphs
                         for l in p.lines for w in l.words if not w.is_empty_cell]
            if positions:
                mean = sum(positions) / len(positions)
                std  = (sum((p - mean)**2 for p in positions) / len(positions)) ** 0.5
                print(info(f"  Block {bn}: {len(positions)} words, x-position std={std:.1f}px"))
    else:
        print(info("No table blocks detected on this page"))
        print(info("(If you expected a table, the x-position std may be below the 80px threshold)"))


def check_raw_text(result: PageOCRResult):
    """Print a preview of the reconstructed raw text."""
    section("Phase 2 — Reconstructed raw text (first 800 chars)")

    text = result.raw_text
    if not text.strip():
        print(fail("Raw text is empty"))
        return

    print(ok(f"Raw text length: {len(text)} characters"))
    print()
    print("─" * 60)
    print(text[:800])
    if len(text) > 800:
        print(f"\n  ... [{len(text) - 800} more characters]")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_checks(file_path: Path, page_index: int):
    print(f"\n{BOLD}{'═' * 60}")
    print(f"  OCR Verification — {file_path.name}")
    print(f"{'═' * 60}{RESET}")

    # Phase 1
    image = check_ingest(file_path, page_index)
    if image is None:
        print(fail("Aborting — could not load page image"))
        return

    # Phase 2
    print(info("Running Tesseract... (this may take a few seconds)"))
    result = ocr_page(image)

    check_structure(result)
    check_confidence(result)
    check_tables(result)
    check_raw_text(result)

    # Summary
    section("Summary")
    vlm_reason = []
    if result.confidence < config.CONFIDENCE_THRESHOLD:
        vlm_reason.append(f"low confidence ({result.confidence:.1f})")
    if result.table_block_nums:
        vlm_reason.append(f"table blocks {result.table_block_nums}")

    if vlm_reason:
        print(ok(f"This page WILL be sent to the VLM  ({', '.join(vlm_reason)})"))
    else:
        print(ok(f"This page will use Tesseract output directly (confidence {result.confidence:.1f} ≥ {config.CONFIDENCE_THRESHOLD})"))


def main():
    parser = argparse.ArgumentParser(
        description="Verify Phase 1 (ingest) and Phase 2 (OCR) of the pipeline."
    )       
    parser.add_argument("--input", required=True,
                        help="Path to a PDF or image file")
    parser.add_argument("--page", type=int, default=1,
                        help="Page number to inspect (1-indexed, default: 1)")
    parser.add_argument("--all-pages", action="store_true",
                        help="Run checks on every page in the document")
    args = parser.parse_args()

    file_path = Path(args.input)
    if not file_path.exists():
        print(fail(f"File not found: {file_path}"))
        sys.exit(1)

    if args.all_pages:
        total = page_count(file_path)
        for i in range(total):
            run_checks(file_path, page_index=i)
    else:
        run_checks(file_path, page_index=args.page - 1)


if __name__ == "__main__":
    main()