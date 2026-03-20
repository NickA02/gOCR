"""OCR Utilities for gOCR. Uses Tesseract with VLM verification."""
from .ocr import *
from .ocr_classes import *

__all__ = ["ocr_page", "WordBox", "PageOCRResult"]
