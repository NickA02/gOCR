"""
tests/test_pipeline.py
──────────────────────
Sanity checks for each pipeline layer, runnable incrementally as you
re-enable components.

Test groups:
  TestConfig          — config values are sane
  TestVLMUtils        — VLM client + encoding (no live call)
  TestMarkdownParser  — image reference detection in markdown
  TestImageHandler    — YOLO model loading + table detection on sample image
  TestFigureParser    — process_image and image_result_to_markdown
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
SAMPLE_PDF    = Path(__file__).parent.parent / "sample.pdf"
SAMPLE_FIGURE = Path(__file__).parent.parent / "_page_2_Figure_0.jpeg"

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

class TestImageHandler:
    """YOLO model loading, table detection, and cropping."""

    @slow
    def test_yolo_predictor_loads(self):
        from image_handler import _get_yolo_predictor
        pred = _get_yolo_predictor()
        assert pred is not None

    @slow
    def test_detect_tables_returns_list(self):
        from image_handler import detect_tables
        result = detect_tables(Image.new("RGB", (100, 100)))
        assert isinstance(result, list)

    @slow
    @pytest.mark.skipif(not SAMPLE_FIGURE.exists(), reason="Sample figure not found")
    def test_detect_tables_on_sample_figure_finds_multiple(self):
        from image_handler import detect_tables
        img = Image.open(SAMPLE_FIGURE).convert("RGB")
        bboxes = detect_tables(img)
        assert len(bboxes) > 1, \
            f"Expected multiple tables in sample figure, got {len(bboxes)}"

    @slow
    @pytest.mark.skipif(not SAMPLE_FIGURE.exists(), reason="Sample figure not found")
    def test_detected_bboxes_within_image_bounds(self):
        from image_handler import detect_tables
        img = Image.open(SAMPLE_FIGURE).convert("RGB")
        w, h = img.size
        for x1, y1, x2, y2 in detect_tables(img):
            assert 0 <= x1 < x2 <= w, f"Invalid x range: {x1}, {x2}"
            assert 0 <= y1 < y2 <= h, f"Invalid y range: {y1}, {y2}"

    def test_crop_table_respects_image_bounds(self):
        from image_handler import crop_table
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        crop = crop_table(img, (0, 0, 50, 50), padding=30)
        w, h = crop.size
        assert 0 < w <= 200 and 0 < h <= 200


# ══════════════════════════════════════════════════════════════════════════════
# TestFigureParser
# ══════════════════════════════════════════════════════════════════════════════

class TestFigureParser:
    """
    Tests for process_image and image_result_to_markdown.

    Fast tests use unittest.mock to patch detect_tables and _vlm_summarize —
    no YOLO models or live VLM required.

    Slow tests run against the real sample figure with a dead VLM URL to
    verify that table detection still succeeds when the LLM is unavailable.
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_image_result(self, tables=None, summary="Figure summary.", flags=None):
        from image_handler import ImageResult
        return ImageResult(
            image_path=Path("test.jpeg"),
            figure_summary=summary,
            tables=tables or [],
            rotation_degrees=0,
            flags=flags or [],
        )

    def _make_table_result(self, idx=0, ocr_text="Table data.", confidence="high", flags=None):
        from image_handler import TableResult
        return TableResult(
            table_index=idx,
            bbox=(10, 10, 100, 100),
            ocr_text=ocr_text,
            confidence=confidence,
            flags=flags or [],
        )

    # ── image_result_to_markdown ──────────────────────────────────────────────

    def test_markdown_no_tables_contains_summary(self):
        from image_handler import image_result_to_markdown
        result = self._make_image_result(summary="A site map.")
        md = image_result_to_markdown(result)
        assert "A site map." in md
        assert "0 table(s)" in md

    def test_markdown_with_tables_lists_all(self):
        from image_handler import image_result_to_markdown
        tables = [self._make_table_result(i, f"Data for table {i}.") for i in range(3)]
        result = self._make_image_result(tables=tables)
        md = image_result_to_markdown(result)
        assert "3 table(s)" in md
        for i in range(3):
            assert f"Data for table {i}." in md

    def test_markdown_with_image_flags_shows_warnings(self):
        from image_handler import image_result_to_markdown
        result = self._make_image_result(flags=["No tables detected — manual review recommended."])
        md = image_result_to_markdown(result)
        assert "No tables detected" in md

    def test_markdown_vlm_crash_shows_placeholder(self):
        from image_handler import image_result_to_markdown
        result = self._make_image_result(
            summary="(VLM unavailable)",
            flags=["Figure summary error: connection refused"],
        )
        md = image_result_to_markdown(result)
        assert "(VLM unavailable)" in md
        assert "Figure summary error" in md

    # ── process_image with mocked internals (no models needed) ────────────────

    def test_process_image_mocked_vlm_returns_all_tables(self, tmp_path):
        from unittest.mock import patch
        from image_handler import process_image
        from openai import OpenAI

        img = Image.new("RGB", (400, 300), color=(200, 200, 200))
        img_path = tmp_path / "fig.png"
        img.save(str(img_path))

        fake_bboxes = [(10, 10, 190, 140), (210, 10, 390, 140), (10, 160, 390, 290)]

        with patch("image_handler._detect_tables_with_rotation",
                   return_value=(fake_bboxes, 0)), \
             patch("image_handler._vlm_summarize", return_value="Mock summary."):
            result = process_image(img_path, tmp_path, OpenAI(api_key="x"), "mock")

        assert len(result.tables) == 3
        assert result.figure_summary == "Mock summary."
        assert result.rotation_degrees == 0
        for t in result.tables:
            assert t.confidence == "high"

    def test_process_image_vlm_crash_still_returns_tables(self, tmp_path):
        from unittest.mock import patch
        from image_handler import process_image
        from openai import OpenAI

        img = Image.new("RGB", (400, 300), color=(200, 200, 200))
        img_path = tmp_path / "fig.png"
        img.save(str(img_path))

        fake_bboxes = [(10, 10, 190, 140), (210, 10, 390, 140)]

        with patch("image_handler._detect_tables_with_rotation",
                   return_value=(fake_bboxes, 0)), \
             patch("image_handler._vlm_summarize",
                   side_effect=RuntimeError("LLM crashed")):
            result = process_image(img_path, tmp_path, OpenAI(api_key="x"), "mock")

        assert len(result.tables) == 2
        assert result.figure_summary == "(VLM unavailable)"
        assert any("Figure summary error" in f for f in result.flags)
        # Tables are still returned with high confidence — VLM crash only
        # affects the figure summary, not OCR
        for t in result.tables:
            assert t.confidence == "high"

    def test_process_image_no_tables_detected(self, tmp_path):
        from unittest.mock import patch
        from image_handler import process_image
        from openai import OpenAI

        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        img_path = tmp_path / "blank.png"
        img.save(str(img_path))

        with patch("image_handler._detect_tables_with_rotation",
                   return_value=([], 0)), \
             patch("image_handler._vlm_summarize", return_value="A blank figure."):
            result = process_image(img_path, tmp_path, OpenAI(api_key="x"), "mock")

        assert result.tables == []
        assert result.figure_summary == "A blank figure."
        assert any("No tables detected" in f for f in result.flags)

    # ── slow: real YOLO + dead VLM (tests LLM-crash resilience end-to-end) ───

    @slow
    @pytest.mark.skipif(not SAMPLE_FIGURE.exists(), reason="Sample figure not found")
    def test_process_image_dead_vlm_returns_tables(self, tmp_path):
        """YOLO finds tables; dead VLM should not prevent them being returned."""
        from image_handler import process_image
        from openai import OpenAI

        dead_client = OpenAI(base_url="http://localhost:9999/v1", api_key="dummy")
        result = process_image(SAMPLE_FIGURE, tmp_path, dead_client, "none")

        assert result is not None
        assert len(result.tables) > 0, \
            "Should still return detected tables even when VLM is unavailable"
        assert result.figure_summary == "(VLM unavailable)"
        # VLM crash only affects figure_summary; table OCR confidence is independent
        for t in result.tables:
            assert t.confidence in ("high", "low")

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
# TestFullPipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    End-to-end post-processing tests.
    Uses cached marker output as input — no re-running marker_single.
    Remove the @pytest.mark.skip decorators when you re-enable post-processing.
    """

    @slow
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