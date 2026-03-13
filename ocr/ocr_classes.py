from dataclasses import dataclass, field
from config import CONFIDENCE_THRESHOLD




@dataclass
class WordBox:
    """
    A single word token detected by Tesseract.
 
    text is a placeholder marker '·' for empty cells (conf >= 0, no text)
    so that column positions are preserved in table reconstruction.
    """
    text: str               # word text, or '·' for empty cell
    confidence: float       # 0–100; real OCR confidence
    left: int               # bounding box origin x
    top: int                # bounding box origin y
    width: int
    height: int
    is_empty_cell: bool = False  # True when text was blank but conf >= 0
 
 
@dataclass
class Line:
    """A single line within a paragraph."""
    line_num: int
    words: list[WordBox] = field(default_factory=list)
 
    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)
 
    @property
    def mean_confidence(self) -> float:
        real = [w for w in self.words if not w.is_empty_cell]
        if not real:
            return 0.0
        return sum(w.confidence for w in real) / len(real)
 
    @property
    def left_positions(self) -> list[int]:
        """X positions of all words — used for column alignment detection."""
        return [w.left for w in self.words]
 
 
@dataclass
class Paragraph:
    """A paragraph within a block — one or more lines."""
    para_num: int
    lines: list[Line] = field(default_factory=list)
 
    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)
 
    @property
    def mean_confidence(self) -> float:
        all_words = [w for line in self.lines for w in line.words
                     if not w.is_empty_cell]
        if not all_words:
            return 0.0
        return sum(w.confidence for w in all_words) / len(all_words)
 
 
@dataclass
class Block:
    """
    A contiguous region of the page detected by Tesseract.
 
    Blocks typically correspond to paragraphs, tables, headers, or
    figure captions. A table is usually a single block whose internal
    line structure encodes rows.
 
    The `is_likely_table` property uses word x-position variance as a
    heuristic — table cells scatter horizontally more than prose.
 
    Future enhancement: replace heuristic with a trained classifier or
    pass bounding box clusters to the VLM as an explicit hint.
    """
    block_num: int
    paragraphs: list[Paragraph] = field(default_factory=list)
 
    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)
 
    @property
    def mean_confidence(self) -> float:
        all_words = [w for para in self.paragraphs
                     for line in para.lines
                     for w in line.words
                     if not w.is_empty_cell]
        if not all_words:
            return 0.0
        return sum(w.confidence for w in all_words) / len(all_words)
 
    @property
    def all_words(self) -> list[WordBox]:
        return [w for para in self.paragraphs
                for line in para.lines
                for w in line.words]
 
    @property
    def is_likely_table(self) -> bool:
        """
        Heuristic: high variance in word x-positions within a block
        suggests columnar layout (i.e. a table).
 
        A standard deviation above 80px at 300 DPI is a reasonable signal.
        Tune this threshold against your sample documents.
        """
        positions = [w.left for w in self.all_words if not w.is_empty_cell]
        if len(positions) < 4:
            return False
        mean = sum(positions) / len(positions)
        variance = sum((p - mean) ** 2 for p in positions) / len(positions)
        return variance ** 0.5 > 80
 
 
@dataclass
class PageOCRResult:
    """
    Full structured Tesseract output for a single page.
 
    Attributes
    ----------
    blocks : List[Block]
        The full structural tree for the page.
    confidence : float
        Mean word confidence across the page (0–100).
    needs_vlm : bool
        True when confidence is below config.CONFIDENCE_THRESHOLD,
        or when the page contains likely table blocks (which Tesseract
        cannot reconstruct accurately regardless of confidence).
    raw_text : str
        Plain text reconstruction with structural markers, used as
        grounding context in the VLM prompt.
    table_block_nums : List[int]
        Block numbers flagged as likely tables — passed to the VLM
        prompt so it knows where to apply table reconstruction.
    """
    blocks: list[Block]
    confidence: float
 
    @property
    def needs_vlm(self) -> bool:
        return (
            self.confidence < CONFIDENCE_THRESHOLD
            or bool(self.table_block_nums)
        )
 
    @property
    def table_block_nums(self) -> list[int]:
        return [b.block_num for b in self.blocks if b.is_likely_table]
 
    @property
    def raw_text(self) -> str:
        """
        Reconstruct page text from the structural tree.
 
        Blocks are separated by blank lines.
        Paragraphs within a block are separated by blank lines.
        Lines within a paragraph are separated by newlines.
        Table blocks are annotated with a marker so the VLM knows to
        apply table reconstruction to that region.
        """
        sections = []
        for block in self.blocks:
            if block.is_likely_table:
                sections.append(f"[TABLE BLOCK {block.block_num}]")
            sections.append(block.text)
        return "\n\n".join(sections)
 
 