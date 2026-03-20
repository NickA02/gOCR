# gOCR

A Python-based OCR pipeline for converting PDF documents into structured markdown, with emphasis on extracting and transcribing tables from scanned or mixed-content PDFs.

## What it does

gOCR processes a PDF through a multi-stage pipeline:

1. **PDF to Markdown** — Uses [Marker](https://github.com/VikParuchuri/marker) to convert digital and scanned PDF pages to markdown, extracting embedded images.
2. **Image Processing** — For each image reference, a YOLO model detects tables, runs orientation search (0°/90°/180°/270°) to handle rotated content, and uses [Surya](https://github.com/VikParuchuri/surya) OCR to transcribe table contents.
3. **VLM Summarization** — A local Vision Language Model (via LMStudio) summarizes figures and supplements table extraction.
4. **Assembly** — Post-processes and assembles all sections into a final markdown document with a metadata header flagging any low-confidence sections for human review.

## Prerequisites

- Python 3.x
- [LMStudio](https://lmstudio.ai/) running locally with a vision-capable model (e.g. `qwen3-vl-8b`) on `http://localhost:1234` (Can change to use any OpenAI API compatible LLM Server)
- YOLO table detection model at `models/tables.pt`

## Setup

Install dependencies (a `requirements.txt` or `pyproject.toml` should cover this):

```bash
pip install requirements.txt
```

Edit `config.py` to match your environment:

```python
API_KEY = "lm-studio"                  # API key for local VLM
API_ROOT = "http://localhost:1234/v1"  # LMStudio endpoint
VLM_MODEL = "qwen3-vl-8b"             # Vision language model name
```

## Usage

### Convert a PDF

```bash
python pipeline.py <input.pdf> [--output output/result.md] [--model MODEL]
```

Example:

```bash
python pipeline.py sample.pdf --output output/sample.md
```

## Testing

```bash
# Fast tests (no models loaded)
pytest tests/ -m "not slow" -v

# Include Marker tests (uses cache to avoid repeated 2-min runs)
pytest tests/ -m "marker" -v

# Force a fresh Marker run
pytest tests/ -m "marker" --no-marker-cache -v

# Run everything
pytest tests/ -v
```

## Project structure

```
gOCR/
├── pipeline.py           # Main orchestrator / CLI entry point
├── image_handler.py      # YOLO table detection, Surya OCR, orientation search
├── markdown_processor.py # Marker output post-processing and final assembly
├── vlm_utils.py          # OpenAI-compatible VLM client utilities
├── config.py             # Configuration (API endpoints, model paths, thresholds)
├── models/
│   └── tables.pt         # YOLO table detection model
├── tests/
│   ├── conftest.py       # Pytest fixtures (session-scoped Marker cache)
│   └── test_pipeline.py  # Test suite
└── output/               # Generated markdown output
```