"""
pipeline.py
───────────
Main entry point for the OCR pipeline.

Usage:
    python pipeline.py <input.pdf> [--output output.md] [--model MODEL]

Flow:
    1. Run marker_single on the input PDF  →  raw markdown + image files
    2. Post-process the markdown:
         • Clean Marker artifacts on digital pages
         • Detect tables in image references using Surya
         • Extract & verify each table with VLM grounding
    3. Write the final unified markdown to the output path
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config
from markdown_processor import process_markdown
from vlm_utils import get_client


# ── Step 1: Run marker_single ─────────────────────────────────────────────────

def run_marker(pdf_path: Path, marker_out_dir: Path) -> Path:
    """
    Call marker_single as a subprocess and return the path to its
    output markdown file.

    marker_single writes:
        <marker_out_dir>/<pdf_stem>/<pdf_stem>.md
        <marker_out_dir>/<pdf_stem>/<image files>
    """
    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_dir", str(marker_out_dir),
        "--use_llm",
        "--llm_service",
        "--ollama_base_url", config.API_ROOT,
        "--ollama_model", config.VLM_MODEL
    ]

    print(f"Running marker_single on '{pdf_path.name}'…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("marker_single stderr:", result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"marker_single failed with exit code {result.returncode}"
        )

    # Locate the output markdown file
    stem = pdf_path.stem
    md_path = marker_out_dir / stem / f"{stem}.md"
    if not md_path.exists():
        # Some marker versions flatten the output directory
        candidates = list(marker_out_dir.rglob("*.md"))
        if not candidates:
            raise FileNotFoundError(
                f"marker_single ran successfully but no .md file found under {marker_out_dir}"
            )
        md_path = candidates[0]

    print(f"Marker output: {md_path}")
    return md_path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: Path,
    output_path: Path,
    model: str = config.VLM_MODEL,
) -> str:
    """
    Run the full pipeline on a PDF and write the result to output_path.

    Returns the final markdown string.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    # Set up working directories
    temp_dir       = Path(config.TEMP_DIR)
    marker_out_dir = temp_dir / "marker_output"
    temp_dir.mkdir(exist_ok=True)
    marker_out_dir.mkdir(exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_client()

    try:
        # ── Step 1: Marker ────────────────────────────────────────────────────
        md_path = run_marker(pdf_path, marker_out_dir)
        raw_markdown = md_path.read_text(encoding="utf-8")

        # The directory containing the .md file also contains Marker's images
        marker_page_dir = md_path.parent

        #Verify that the pipeline runs without refinement

        # ── Step 2: Post-process ──────────────────────────────────────────────
        print("Post-processing Marker output…")
        final_markdown, review_flags = process_markdown(
            raw_markdown=raw_markdown,
            marker_output_dir=marker_page_dir,
            temp_dir=temp_dir,
            client=client,
            model=model,
        )
        final_markdown = raw_markdown
        review_flags = []

        # ── Step 3: Write output ──────────────────────────────────────────────
        output_path.write_text(final_markdown, encoding="utf-8")
        print(f"\nDone. Output written to '{output_path}'.")

        if review_flags:
            print(f"⚠  {len(review_flags)} section(s) flagged for human review:")
            for flag in review_flags:
                print(f"   • {flag}")
        else:
            print("✓  All sections high confidence.")

        return final_markdown

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PDF to clean, LLM-ready markdown.\n"
            "Digital pages are processed by Marker; scanned map images are\n"
            "handled by Surya table detection + VLM-grounded extraction."
        )
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output markdown path (default: output/<pdf_stem>.md)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.VLM_MODEL,
        help=f"VLM model name in LMStudio (default: {config.VLM_MODEL})",
    )
    args = parser.parse_args()

    output_path = args.output or Path(config.OUTPUT_DIR) / f"{args.pdf.stem}.md"

    try:
        run_pipeline(
            pdf_path=args.pdf,
            output_path=output_path,
            model=args.model,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
