"""
conftest.py
───────────
Pytest fixtures shared across the test suite.

Key fixture: marker_output
    Session-scoped. Runs marker_single on sample.pdf ONCE per test session
    and caches the result to tests/.marker_cache/. On subsequent runs the
    cache is reused immediately — no 2-minute wait.

Cache invalidation:
    Delete tests/.marker_cache/ to force a fresh run:
        rm -rf tests/.marker_cache

    Or run with --no-marker-cache:
        pytest --no-marker-cache
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ── Project root (one level up from this conftest) ────────────────────────────
ROOT       = Path(__file__).parent.parent
SAMPLE_PDF = ROOT / "sample.pdf"
CACHE_DIR  = Path(__file__).parent / ".marker_cache"


# ── CLI option to force cache invalidation ────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--no-marker-cache",
        action="store_true",
        default=False,
        help="Ignore existing marker cache and re-run marker_single.",
    )


# ── Session-scoped marker output fixture ─────────────────────────────────────

@pytest.fixture(scope="session")
def marker_output(request) -> dict:
    """
    Run marker_single on sample.pdf once per session, cache the result.

    Returns a dict with:
        md_path    : Path   — path to the output .md file
        md_content : str    — the markdown text
        page_dir   : Path   — directory containing .md + image files
    """
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not found — place it in the project root")

    force_refresh = request.config.getoption("--no-marker-cache")
    cached_md = CACHE_DIR / "sample" / "sample.md"

    # ── Use cache if available ────────────────────────────────────────────────
    if cached_md.exists() and not force_refresh:
        print(f"\n  [marker_output] Using cached marker output: {cached_md}")
        return {
            "md_path":    cached_md,
            "md_content": cached_md.read_text(encoding="utf-8"),
            "page_dir":   cached_md.parent,
        }

    # ── Run marker_single ─────────────────────────────────────────────────────
    if force_refresh and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "marker_single",
        str(SAMPLE_PDF),
        "--output_dir", str(CACHE_DIR),
    ]

    print(f"\n  [marker_output] Running marker_single (this takes ~2 min, result will be cached)…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(
            f"marker_single failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Locate output .md — marker writes to <output_dir>/<stem>/<stem>.md
    candidates = list(CACHE_DIR.rglob("*.md"))
    if not candidates:
        pytest.fail(
            f"marker_single succeeded but no .md file found under {CACHE_DIR}"
        )

    md_path = candidates[0]
    print(f"  [marker_output] Cached to {md_path}")

    return {
        "md_path":    md_path,
        "md_content": md_path.read_text(encoding="utf-8"),
        "page_dir":   md_path.parent,
    }
