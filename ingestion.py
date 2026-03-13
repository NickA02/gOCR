"""File ingestion and processing utilities for gOCR."""

import base64
from pathlib import Path
from typing import Generator
from PIL import Image
import pymupdf

from config import * 

def load_pages(document_path: Path) -> Generator[Image.Image, None, None]:
    """Load a document (PDF or image) and yield each page as a PIL Image."""
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    doc_type = document_path.suffix.lower()
    
    if doc_type == ".pdf":
        # Load PDF with pymupdf
        doc = pymupdf.open(document_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield img
    elif doc_type in SUPPORTED_IMAGE_FORMATS:
        # Load single image
        yield Image.open(document_path)

def page_count(file_path: Path) -> int:
    """
    Return the number of pages in a PDF, or 1 for image files.
 
    Useful for progress reporting without rendering all pages.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
 
    if suffix == ".pdf":
        with pymupdf.open(file_path) as doc:
            return doc.page_count
 
    if suffix in SUPPORTED_IMAGE_FORMATS:
        return 1
 
    raise ValueError(f"Unsupported file type: '{suffix}'")

def _normalize(img: Image.Image) -> Image.Image:
    """Convert PIL Image to consistent RGB mode."""
    if img.mode == "RGB":
        return img
 
    if img.mode == "RGBA":
        # Composite transparent regions onto white before discarding alpha
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 = alpha channel
        return background
 
    return img.convert("RGB")

