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

Run only the fast unit tests (no models, no subprocess):
    pytest tests/test_pipeline.py -m "not slow" -v

Run Marker tests (uses cache — only runs marker_single once per session):
    pytest tests/test_pipeline.py -m "marker" -v

Force a fresh marker run (ignores cache):
    pytest tests/test_pipeline.py -m "marker" --no-marker-cache -v

Run everything:
    pytest tests/test_pipeline.py -v
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# ── Paths ──────────────────────────────────────────────────────────────────────
SAMPLE_PDF   = Path(__file__).parent.parent / "sample.pdf"
SAMPLE_IMAGE = Path(__file__).parent.parent / "_page_2_Figure_0.jpeg"

# ── Pytest marks ──────────────────────────────────────────────────────────────
slow   = pytest.mark.slow
marker = pytest.mark.marker


# ── Lazy imports (keeps unrelated tests runnable if a dep is missing) ─────────

def _import_vlm():
    from vlm_utils import encode_image_base64, get_client
    return encode_image_base64, get_client


def _import_markdown():
    from markdown_processor import _IMAGE_REF_RE, clean_marker_output
    return _IMAGE_REF_RE, clean_marker_output


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
        _, get_client = _import_vlm()
        client = get_client()
        assert isinstance(client, OpenAI)

    def test_get_client_uses_config_url(self):
        _, get_client = _import_vlm()
        client = get_client()
        assert config.API_ROOT in str(client.base_url)

    def test_encode_image_base64_returns_string(self, tmp_path):
        encode_image_base64, _ = _import_vlm()
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))
        result = encode_image_base64(img_path)
        assert isinstance(result, str) and len(result) > 0

    def test_encode_image_base64_is_valid_base64(self, tmp_path):
        import base64
        encode_image_base64, _ = _import_vlm()
        img = Image.new("RGB", (10, 10), color=(0, 255, 0))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))
        decoded = base64.b64decode(encode_image_base64(img_path))
        assert len(decoded) > 0

    def test_encode_image_base64_missing_file_raises(self, tmp_path):
        encode_image_base64, _ = _import_vlm()
        with pytest.raises(FileNotFoundError):
            encode_image_base64(tmp_path / "nonexistent.png")


# ══════════════════════════════════════════════════════════════════════════════
# TestMarkdownParser
# ══════════════════════════════════════════════════════════════════════════════

class TestMarkdownParser:
    def test_image_ref_regex_matches_marker_format(self):
        _IMAGE_REF_RE, _ = _import_markdown()
        matches = list(_IMAGE_REF_RE.finditer("Some text\n![](_page_2_Figure_0.jpeg)\nMore"))
        assert len(matches) == 1
        assert matches[0].group(1) == "_page_2_Figure_0.jpeg"

    def test_image_ref_regex_matches_with_alt_text(self):
        _IMAGE_REF_RE, _ = _import_markdown()
        matches = list(_IMAGE_REF_RE.finditer("![caption](_page_1_Figure_1.png)"))
        assert len(matches) == 1
        assert matches[0].group(1) == "_page_1_Figure_1.png"

    def test_image_ref_regex_no_match_on_plain_text(self):
        _IMAGE_REF_RE, _ = _import_markdown()
        assert len(list(_IMAGE_REF_RE.finditer("No images here."))) == 0

    def test_image_ref_regex_finds_multiple_references(self):
        _IMAGE_REF_RE, _ = _import_markdown()
        md = "A\n![](_page_1_Figure_0.jpeg)\nB\n![](_page_2_Figure_0.jpeg)\n"
        matches = list(_IMAGE_REF_RE.finditer(md))
        assert len(matches) == 2
        assert matches[0].group(1) == "_page_1_Figure_0.jpeg"
        assert matches[1].group(1) == "_page_2_Figure_0.jpeg"

    def test_clean_removes_non_printable(self):
        _, clean = _import_markdown()
        result = clean("good\x00text\x01here")
        assert "\x00" not in result and "\x01" not in result
        assert "goodtexthere" in result

    def test_clean_collapses_blank_lines(self):
        _, clean = _import_markdown()
        result = clean("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result
        assert "line1" in result and "line2" in result

    def test_clean_fixes_hyphenated_linebreak(self):
        _, clean = _import_markdown()
        assert "environmental" in clean("environ-\nmental remediation")

    def test_clean_preserves_table_pipes(self):
        _, clean = _import_markdown()
        result = clean("| Col A | Col B |\n|-------|-------|\n| val1  | val2  |")
        assert "|" in result and "Col A" in result and "val1" in result

    def test_clean_normalizes_line_endings(self):
        _, clean = _import_markdown()
        assert "\r" not in clean("line1\r\nline2\rline3")


# ══════════════════════════════════════════════════════════════════════════════
# TestImageHandler
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# TestMarkerStep
# ══════════════════════════════════════════════════════════════════════════════

class TestMarkerStep:
    """
    Sanity checks on Marker output.

    Uses the session-scoped `marker_output` fixture from conftest.py —
    marker_single runs ONCE per session and is cached to tests/.marker_cache/.
    Subsequent runs skip the 2-minute wait and read from cache instantly.

    To force a fresh run:   pytest -m marker --no-marker-cache
    To clear cache:         rm -rf tests/.marker_cache
    """

    @slow
    @marker
    def test_marker_single_is_on_path(self):
        result = subprocess.run(
            ["marker_single", "--help"], capture_output=True, text=True,
        )
        assert result.returncode in (0, 1, 2), \
            "marker_single not found on PATH — run: pip install marker-pdf"

    @slow
    @marker
    def test_marker_produces_markdown_file(self, marker_output):
        md_path = marker_output["md_path"]
        assert md_path.exists() and md_path.suffix == ".md"

    @slow
    @marker
    def test_marker_output_is_non_empty(self, marker_output):
        assert len(marker_output["md_content"].strip()) > 100, \
            "Marker output is unexpectedly short"

    @slow
    @marker
    def test_marker_output_contains_table_markdown(self, marker_output):
        assert "|" in marker_output["md_content"], \
            "Expected markdown table pipes in Marker output"

    @slow
    @marker
    def test_marker_output_contains_image_reference(self, marker_output):
        _IMAGE_REF_RE, _ = _import_markdown()
        matches = list(_IMAGE_REF_RE.finditer(marker_output["md_content"]))
        assert len(matches) > 0, \
            "Expected at least one ![](...) image reference in Marker output"

    @slow
    @marker
    def test_marker_image_files_exist_on_disk(self, marker_output):
        _IMAGE_REF_RE, _ = _import_markdown()
        md_path = marker_output["md_path"]
        for match in _IMAGE_REF_RE.finditer(marker_output["md_content"]):
            img_path = md_path.parent / match.group(1)
            assert img_path.exists(), \
                f"Marker referenced image not found on disk: {img_path}"





# ══════════════════════════════════════════════════════════════════════════════
# TestFullPipeline  (skipped until post-processing is re-enabled)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    End-to-end post-processing tests.
    Uses cached marker output as input — no re-running marker_single.
    Remove the @pytest.mark.skip decorators when you re-enable post-processing.
    """

    @slow
    @pytest.mark.skip(reason="Post-processing disabled — re-enable when ready")
    def test_full_pipeline_produces_output_file(self, marker_output, tmp_path):
        from markdown_processor import process_markdown
        _, get_client = _import_vlm()
        final_md, _ = process_markdown(
            raw_markdown=marker_output["md_content"],
            marker_output_dir=marker_output["page_dir"],
            temp_dir=tmp_path / "temp",
            client=get_client(),
            model=config.VLM_MODEL,
        )
        out = tmp_path / "output.md"
        out.write_text(final_md, encoding="utf-8")
        assert out.exists()


    @slow
    @pytest.mark.skip(reason="Post-processing disabled — re-enable when ready")
    def test_full_pipeline_no_raw_image_refs_in_output(self, marker_output, tmp_path):
        _IMAGE_REF_RE, _ = _import_markdown()
        from markdown_processor import process_markdown
        _, get_client = _import_vlm()
        final_md, _ = process_markdown(
            raw_markdown=marker_output["md_content"],
            marker_output_dir=marker_output["page_dir"],
            temp_dir=tmp_path / "temp",
            client=get_client(),
            model=config.VLM_MODEL,
        )
        raw_refs = list(_IMAGE_REF_RE.finditer(final_md))
        assert len(raw_refs) == 0, \
            f"Found {len(raw_refs)} unprocessed image reference(s) in final output"

    @slow
    @pytest.mark.skip(reason="Post-processing disabled — re-enable when ready")
    def test_full_pipeline_header_present(self, marker_output, tmp_path):
        from markdown_processor import process_markdown
        _, get_client = _import_vlm()
        final_md, _ = process_markdown(
            raw_markdown=marker_output["md_content"],
            marker_output_dir=marker_output["page_dir"],
            temp_dir=tmp_path / "temp",
            client=get_client(),
            model=config.VLM_MODEL,
        )
        assert "OCR PIPELINE OUTPUT" in final_md
