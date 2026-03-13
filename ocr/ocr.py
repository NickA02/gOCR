"""OCR Utilities for gOCR. Uses Tesseract with VLM verification."""
from dataclasses import dataclass, field
from PIL import Image
import pytesseract
import pandas as pd
from .ocr_classes import *
 
from config import *

def ocr_page(image: Image.Image) -> PageOCRResult:
    """Run Tesseract OCR on a PIL Image and return structured results.
    Args:
        image: PIL Image to process.
    Returns:
        PageOCRResult containing hierarchical OCR data and confidence.
        """
    data = pytesseract.image_to_data(
        image,
        lang=TESSERACT_LANG,
        output_type=pytesseract.Output.DICT,
    )
 
    blocks = _build_structure(data)
    confidence = _page_confidence(blocks)
 
    return PageOCRResult(blocks=blocks, confidence=confidence)

def _build_structure(data: dict) -> list[Block]:
    """Convert Tesseract's flat data output into a hierarchical structure of Blocks, Paragraphs, Lines, and WordBoxes."""
    blocks: dict[int, Block] = {}
 
    n = len(data["text"])
 
    for i in range(n):
        level = data["level"][i]
        block_num = data["block_num"][i]
        par_num = data["par_num"][i]
        line_num = data["line_num"][i]
        conf = int(data["conf"][i])
        raw_text = data["text"][i]
 
        # Ensure the Block node exists
        if block_num not in blocks:
            blocks[block_num] = Block(block_num=block_num)
 
        block = blocks[block_num]
 
        # Ensure the Paragraph node exists within the block
        para = _get_or_create_paragraph(block, par_num)
 
        # Ensure the Line node exists within the paragraph
        line = _get_or_create_line(para, line_num)
 
        # Only level-5 tokens carry actual word content
        if level != 5:
            continue
 
        # Skip true layout tokens — level 5 with conf=-1 are ambiguous
        # boundary markers that don't correspond to visible content
        if conf == -1:
            continue
 
        text = raw_text.strip()
 
        if text:
            word = WordBox(
                text=text,
                confidence=float(conf),
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                is_empty_cell=False,
            )
        else:
            # Empty string with real confidence = empty table cell.
            # Preserve position with a placeholder so column alignment
            # is not lost when reconstructing table structure.
            word = WordBox(
                text="·",
                confidence=float(conf),
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                is_empty_cell=True,
            )
 
        line.words.append(word)
 
    # Remove empty nodes that received no word tokens
    return _prune(list(blocks.values()))

def _get_or_create_paragraph(block: Block, par_num: int) -> Paragraph:
    for para in block.paragraphs:
        if para.para_num == par_num:
            return para
    para = Paragraph(para_num=par_num)
    block.paragraphs.append(para)
    return para
 
 
def _get_or_create_line(para: Paragraph, line_num: int) -> Line:
    for line in para.lines:
        if line.line_num == line_num:
            return line
    line = Line(line_num=line_num)
    para.lines.append(line)
    return line
 
 
def _prune(blocks: list[Block]) -> list[Block]:
    """Remove structural nodes that contain no word tokens."""
    result = []
    for block in blocks:
        pruned_paras = []
        for para in block.paragraphs:
            pruned_lines = [l for l in para.lines if l.words]
            if pruned_lines:
                pruned_paras.append(
                    Paragraph(para_num=para.para_num, lines=pruned_lines)
                )
        if pruned_paras:
            block.paragraphs = pruned_paras
            result.append(block)
    return result
 
def _page_confidence(blocks: list[Block]) -> float:
    """
    Compute mean word confidence across all real words on the page.
 
    Empty cell placeholders are excluded from the average — they represent
    absent content, not uncertain OCR.
 
    Returns 0.0 for blank pages so they always route to the VLM.
    """
    all_words = [
        w for block in blocks
        for para in block.paragraphs
        for line in para.lines
        for w in line.words
        if not w.is_empty_cell
    ]
    if not all_words:
        return 0.0
    return sum(w.confidence for w in all_words) / len(all_words)