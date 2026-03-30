# glyph_loaders.py
# Lindsey Hill - AIT626 Term Project - Glyph
#
# File loading ONLY. Preprocessing (tokenization, stopwords, lemmatization)
# is intentionally kept separate in glyph_preprocess.py.
#
# Supported formats: .txt  .html/.htm  .pdf
#
# Design note: each format needs different extraction logic to produce a clean
# text string, but once we have that string the same normalization pipeline
# applies to all of them. Keeping loaders separate makes adding new formats easy.

from pathlib import Path
from bs4 import BeautifulSoup


# =============================================================================
# FORMAT-SPECIFIC LOADERS
# =============================================================================

def load_txt(path: Path) -> str:
    """
    Read a plain-text file and return it as a string.
    errors="ignore" silently skips non-UTF-8 bytes common in Gutenberg files.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def load_html(path: Path) -> str:
    """
    Parse an HTML file and return its visible text content.

    Steps:
      1. Parse with BeautifulSoup (html.parser — no lxml dependency needed).
      2. Remove known Gutenberg structural blocks by id (pg-header, pg-footer,
         pg-machine-header, project-gutenberg-license) so boilerplate doesn't
         pollute keyword scoring.
      3. get_text(separator="\n") prevents adjacent inline tags from merging:
         e.g. <b>Romeo</b><i>Juliet</i> → "Romeo\nJuliet" not "RomeoJuliet".
    """
    raw  = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    for bad_id in ("pg-header", "pg-footer", "pg-machine-header",
                   "project-gutenberg-license"):
        tag = soup.find(id=bad_id)
        if tag:
            tag.decompose()

    return soup.get_text(separator="\n")


def _extract_two_column_page(page) -> str:
    """
    Extract text from a two-column PDF page (e.g. the Yellow Wall-Paper scan).

    WHY THIS EXISTS:
        pdfplumber's default extract_text() reads lines left-to-right across
        the full page width, so two-column layouts get merged line-by-line:
            "nary P""ople like John laughs at me, of course..."
        This makes every sentence unreadable and garbles scoring.

    HOW IT WORKS:
        1. Extract all words with their bounding boxes.
        2. Find the horizontal midpoint of the page.
        3. Split words into left column (x1 ≤ mid+20) and right (x0 ≥ mid-20).
           The ±20pt overlap zone catches words that straddle the midpoint.
        4. For each column, group words into lines by vertical position
           (bucketed into 5pt bands so words on the same line cluster together).
        5. Concatenate: left column text + right column text.

    Args:
        page: a pdfplumber Page object

    Returns:
        Clean text string with the two columns read in the correct order.
    """
    words = page.extract_words()
    if not words:
        return page.extract_text() or ""

    x_min = min(w["x0"] for w in words)
    x_max = max(w["x1"] for w in words)
    mid   = (x_min + x_max) / 2

    left_words  = [w for w in words if w["x1"] <= mid + 20]
    right_words = [w for w in words if w["x0"] >= mid - 20]

    def _words_to_text(word_list: list) -> str:
        if not word_list:
            return ""
        # Group words into horizontal lines by bucketed vertical position
        lines: dict = {}
        for w in word_list:
            bucket = round(w["top"] / 5) * 5
            lines.setdefault(bucket, []).append(w)
        result = []
        for top in sorted(lines):
            row = sorted(lines[top], key=lambda w: w["x0"])
            result.append(" ".join(w["text"] for w in row))
        return "\n".join(result)

    return _words_to_text(left_words) + "\n" + _words_to_text(right_words)


def load_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber.

    Automatically detects two-column layout pages and uses column-aware
    extraction on them. Single-column pages use the standard path.

    Detection heuristic: if the horizontal spread of word-start positions
    is more than 45% of the page width, the page is likely two-column.
    This catches magazine and journal layouts (like the Yellow Wall-Paper
    scan) without false-positives on normal single-column books.
    """
    import pdfplumber  # deferred import — only required if PDFs are used

    chunks = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            if not words:
                chunks.append("")
                continue

            x_positions = [w["x0"] for w in words]
            page_width  = page.width or 1.0
            spread      = (max(x_positions) - min(x_positions)) / page_width

            if spread > 0.45:
                chunks.append(_extract_two_column_page(page))
            else:
                chunks.append(page.extract_text() or "")

    return "\n".join(chunks)


# =============================================================================
# PUBLIC DISPATCH
# =============================================================================

def load_text(path: Path) -> str:
    """
    Dispatch to the correct loader based on file extension.
    This is the only function the rest of the pipeline needs to call.

    Raises:
        ValueError for unsupported extensions.
    """
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_txt(path)
    if ext in {".html", ".htm"}:
        return load_html(path)
    if ext == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {ext!r}  (path: {path})")
