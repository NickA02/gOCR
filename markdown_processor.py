"""
markdown_processor.py
─────────────────────
Post-processes the raw markdown output from marker_single.

Responsibilities:
  • Parse the markdown to find image references (![](...))
  • Route each image reference to image_handler.py for table extraction
  • Clean known Marker artifacts in digital table output
  • Reassemble everything into a single clean markdown document with
    a metadata header flagging any pages requiring human review
"""

from __future__ import annotations

import re
from pathlib import Path

from openai import OpenAI

from image_handler import ImageResult, image_result_to_markdown, process_image


# ── Image reference pattern ───────────────────────────────────────────────────
# Matches Marker's figure references, e.g.: ![](_page_2_Figure_0.jpeg)
_IMAGE_REF_RE = re.compile(r"!\[.*?\]\((.+?)\)")


# ── Marker artifact cleaning ──────────────────────────────────────────────────

def clean_marker_output(text: str) -> str:
    """
    Light cleaning of Marker's raw markdown output.

    Marker is generally good on digital pages but has known issues:
      - Corrupted/merged header cells in wide tables
      - Stray non-printable characters
      - Excessive blank lines
    We fix what we can without altering actual data values.
    """
    # Remove non-printable characters (keep tab, LF, CR)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix hyphenated line-break artifacts (e.g. "environ-\nmental" → "environmental")
    text = re.sub(r"-\n([a-z])", r"\1", text)

    return text.strip()


# ── Main processor ────────────────────────────────────────────────────────────

def process_markdown(
    raw_markdown: str,
    marker_output_dir: Path,
    temp_dir: Path,
    client: OpenAI,
    model: str,
) -> tuple[str, list[str]]:
    """
    Process the raw Marker markdown output end-to-end.

    Args:
        raw_markdown:      The full markdown string from marker_single.
        marker_output_dir: Directory where Marker saved its output files
                           (used to resolve relative image paths).
        temp_dir:          Scratch directory for intermediate files.
        client:            VLM client.
        model:             VLM model name.

    Returns:
        (processed_markdown, review_flags)
        review_flags is a list of human-readable warnings about low-confidence
        sections that should be reviewed before use in production.
    """
    review_flags: list[str] = []
    segments: list[str] = []

    # Split the document at image references so we can process each section
    # separately while preserving the surrounding text in order.
    last_end = 0
    for match in _IMAGE_REF_RE.finditer(raw_markdown):
        # Text before this image reference
        before = raw_markdown[last_end : match.start()]
        if before.strip():
            segments.append(("text", clean_marker_output(before)))

        # Resolve image path relative to Marker's output directory
        image_filename = match.group(1)
        image_path = marker_output_dir / image_filename
        if not image_path.exists():
            # Try the temp dir (in case images were copied there)
            image_path = temp_dir / image_filename

        if not image_path.exists():
            segments.append(("text", f"<!-- IMAGE NOT FOUND: {image_filename} -->"))
            review_flags.append(f"Image file not found: {image_filename}")
        else:
            segments.append(("image", image_path))

        last_end = match.end()

    # Any trailing text after the last image reference
    tail = raw_markdown[last_end:]
    if tail.strip():
        segments.append(("text", clean_marker_output(tail)))

    # ── Process each segment ──────────────────────────────────────────────────
    output_parts: list[str] = []
    image_results: list[ImageResult] = []

    for kind, content in segments:
        if kind == "text":
            output_parts.append(content)
        else:
            image_path: Path = content
            print(f"  Processing image: {image_path.name}")
            result = process_image(
                image_path=image_path,
                temp_dir=temp_dir,
                client=client,
                model=model,
            )
            image_results.append(result)

            # Collect flags for low-confidence tables
            for table in result.tables:
                if table.confidence != "high":
                    review_flags.append(
                        f"Image '{image_path.name}', table {table.table_index + 1} "
                        f"({table.confidence} confidence): "
                        + "; ".join(table.flags)
                    )
            for flag in result.flags:
                review_flags.append(f"Image '{image_path.name}': {flag}")

            output_parts.append(image_result_to_markdown(result))

    # ── Assemble final document ───────────────────────────────────────────────
    body = "\n\n---\n\n".join(p for p in output_parts if p.strip())
    header = _build_header(review_flags)

    return f"{header}\n\n---\n\n{body}", review_flags


def _build_header(review_flags: list[str]) -> str:
    """Build the document-level metadata header."""
    lines = ["# OCR PIPELINE OUTPUT"]

    if review_flags:
        lines.append(f"\n**⚠ {len(review_flags)} section(s) require human review before use "
                     "in financial or compliance contexts.**\n")
        for flag in review_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("\n✓ All sections extracted with high confidence.")

    lines.append(
        "\n> Values marked `[UNCERTAIN]` or `[NEEDS REVIEW]` should be verified "
        "against the source document before use in audit or reporting workflows.\n"
        "> `NOT PROVIDED` = field present but blank in source. "
        "`—` = field not applicable."
    )

    return "\n".join(lines)
