"""
tests/test_pipeline.py
──────────────────────
Sanity checks for each pipeline layer, runnable incrementally as you
re-enable components.

Test groups:
  TestConfig          — config values are sane
  TestVLMUtils        — VLM client + encoding (no live call)
  TestMarkdownParser  — image reference detection in markdown
  TestImageHandler    — Surya model loading + table detection on sample image
  TestMarkerStep      — marker_single subprocess produces a .md file
  TestFullPipeline    — end-to-end run with post_process flag (skipped if disabled)

Run all:
    pytest tests/test_pipeline.py -v

Run only the cheap unit tests (no Surya, no Marker, no VLM):
    pytest tests/test_pipeline.py -v -m "not slow"

Run only the Marker sanity check:
    pytest tests/test_pipeline.py -v -m "marker"
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

# ── Make sure the project root is importable ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from vlm_utils import encode_image_base64, get_client

# ── Paths ──────────────────────────────────────────────────────────────────────
SAMPLE_PDF   = Path(__file__).parent.parent / "sample.pdf"
SAMPLE_IMAGE = Path(__file__).parent.parent / "_page_2_Figure_0.jpeg"

# ── Markers ────────────────────────────────────────────────────────────────────
# Mark slow tests so they can be skipped during rapid iteration
slow   = pytest.mark.slow
marker = pytest.mark.marker


# ══════════════════════════════════════════════════════════════════════════════
# TestConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_api_root_is_set(self):
        assert config.API_ROOT.startswith("http"), \
            "API_ROOT should be a valid URL"

    def test_vlm_model_is_set(self):
        assert isinstance(config.VLM_MODEL, str) and config.VLM_MODEL, \
            "VLM_MODEL must be a non-empty string"

    def test_table_crop_padding_positive(self):
        assert config.TABLE_CROP_PADDING > 0

    def test_min_table_area_positive(self):
        assert config.MIN_TABLE_AREA > 0

    def test_output_dir_is_string(self):
        assert isinstance(config.OUTPUT_DIR, str)

    def test_temp_dir_is_string(self):
        assert isinstance(config.TEMP_DIR, str)


# ══════════════════════════════════════════════════════════════════════════════
# TestVLMUtils
# ══════════════════════════════════════════════════════════════════════════════

class TestVLMUtils:
    def test_get_client_returns_openai_client(self):
        from openai import OpenAI
        client = get_client()
        assert isinstance(client, OpenAI)

    def test_get_client_uses_config_url(self):
        client = get_client()
        assert config.API_ROOT in str(client.base_url)

    def test_encode_image_base64_returns_string(self, tmp_path):
        """Create a tiny test PNG and verify base64 encoding works."""
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        result = encode_image_base64(img_path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_image_base64_is_valid_base64(self, tmp_path):
        import base64
        img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        result = encode_image_base64(img_path)
        # Should not raise
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_image_base64_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            encode_image_base64(tmp_path / "nonexistent.png")


# ══════════════════════════════════════════════════════════════════════════════
# TestMarkdownParser
# ══════════════════════════════════════════════════════════════════════════════

# TODO -- Implemented but waiting on integration until 1st stage is stable
_IMAGE_REF_RE = re.compile(r"!\[.*?\]\((.*?)\)") # Matches ![alt](path) and captures the path

# ══════════════════════════════════════════════════════════════════════════════
# TestImageHandler
# ══════════════════════════════════════════════════════════════════════════════

# TODO -- Implemented but waiting on integration until 1st and 2nd stage ar stable

# ══════════════════════════════════════════════════════════════════════════════
# TestMarkerStep
# ══════════════════════════════════════════════════════════════════════════════

class TestMarkerStep:
    """
    Sanity check that marker_single is installed and produces output.
    Marked slow + marker.
    """

    @slow
    @marker
    def test_marker_single_is_on_path(self):
        result = subprocess.run(
            ["marker_single", "--help"],
            capture_output=True,
            text=True,
        )
        # marker_single --help exits 0 or 1 depending on version — just check
        # it's found (not FileNotFoundError / exit 127)
        assert result.returncode in (0, 1, 2), \
            "marker_single not found on PATH — run: pip install marker-pdf"

    @slow
    @marker
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_marker_produces_markdown_file(self, tmp_path):
        """Run marker_single on the sample PDF and verify a .md is produced."""
        from pipeline import run_marker
        md_path = run_marker(SAMPLE_PDF, tmp_path)
        assert md_path.exists(), f"Expected markdown file at {md_path}"
        assert md_path.suffix == ".md"

    @slow
    @marker
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_marker_output_is_non_empty(self, tmp_path):
        from pipeline import run_marker
        md_path = run_marker(SAMPLE_PDF, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 100, \
            "Marker output is unexpectedly short — possible extraction failure"

    @slow
    @marker
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_marker_output_contains_table_markdown(self, tmp_path):
        """Verify Marker found at least one markdown table in the digital pages."""
        from pipeline import run_marker
        md_path = run_marker(SAMPLE_PDF, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        assert "|" in content, \
            "Expected markdown table pipes in Marker output — no tables found"

    @slow
    @marker
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_marker_output_contains_image_reference(self, tmp_path):
        """Verify Marker emitted at least one image reference for the scanned page."""
        from pipeline import run_marker
        md_path = run_marker(SAMPLE_PDF, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        matches = list(_IMAGE_REF_RE.finditer(content))
        assert len(matches) > 0, \
            "Expected at least one ![](...) image reference in Marker output"

    @slow
    @marker
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_marker_image_files_exist_on_disk(self, tmp_path):
        """Verify Marker actually wrote the referenced image files."""
        from pipeline import run_marker
        md_path = run_marker(SAMPLE_PDF, tmp_path)
        content = md_path.read_text(encoding="utf-8")
        for match in _IMAGE_REF_RE.finditer(content):
            img_path = md_path.parent / match.group(1)
            assert img_path.exists(), \
                f"Marker referenced image not found on disk: {img_path}"


# ══════════════════════════════════════════════════════════════════════════════
# TestFullPipeline  (skipped until post-processing is re-enabled)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    End-to-end tests. Add --post_process flag to pipeline.run_pipeline()
    before enabling these.
    """

    @slow
    @pytest.mark.skip(reason="Post-processing disabled — re-enable when ready")
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_full_pipeline_produces_output_file(self, tmp_path):
        from pipeline import run_pipeline
        output_path = tmp_path / "output.md"
        run_pipeline(SAMPLE_PDF, output_path)
        assert output_path.exists()

    @slow
    @pytest.mark.skip(reason="Post-processing disabled — re-enable when ready")
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_full_pipeline_output_contains_mw_well_labels(self, tmp_path):
        """Final output should contain monitoring well labels from the map page."""
        from pipeline import run_pipeline
        output_path = tmp_path / "output.md"
        run_pipeline(SAMPLE_PDF, output_path)
        content = output_path.read_text(encoding="utf-8")
        # At least one MW- label should appear in the extracted map tables
        assert re.search(r"MW-\d+S", content), \
            "Expected monitoring well labels (MW-XXXS) in pipeline output"

    @slow
    @pytest.mark.skip(reason="Post-processing disabled — re-enable when ready")
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_full_pipeline_output_contains_no_raw_image_refs(self, tmp_path):
        """Raw ![](...) references should be replaced, not passed through."""
        from pipeline import run_pipeline
        output_path = tmp_path / "output.md"
        run_pipeline(SAMPLE_PDF, output_path)
        content = output_path.read_text(encoding="utf-8")
        raw_refs = list(_IMAGE_REF_RE.finditer(content))
        assert len(raw_refs) == 0, \
            f"Found {len(raw_refs)} unprocessed image reference(s) in final output"

    @slow
    @pytest.mark.skip(reason="Post-processing disabled — re-enable when ready")
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample.pdf not found")
    def test_full_pipeline_header_present(self, tmp_path):
        from pipeline import run_pipeline
        output_path = tmp_path / "output.md"
        run_pipeline(SAMPLE_PDF, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "OCR PIPELINE OUTPUT" in content
