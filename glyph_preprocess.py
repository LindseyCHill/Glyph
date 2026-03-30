# glyph_preprocess.py
# Lindsey Hill - AIT626 Term Project - Glyph
#
# Text normalization and tokenization pipeline used by BOTH document
# preprocessing and query preprocessing, so the token space is consistent —
# a query token and its matching document token go through exactly the
# same steps.
#
# Pipeline (applied to docs and queries alike):
#   1. strip_gutenberg_boilerplate() — strip Project Gutenberg headers/footers
#   2. normalize_text()             — fix PDF artifacts, hyphen line-breaks, whitespace
#   3. sent_tokenize()              — sentence split (docs only; for display)
#   4. wordpunct_tokenize()         — word/punctuation tokens
#   5. remove_punctuation()         — strip punctuation characters from tokens
#   6. stopwords filter             — drop common English function words
#   7. WordNetLemmatizer            — reduce to base form (houses→house, ran→run)
#
# Matches the Lab 1 preprocessing pattern used in class:
#   wordpunct_tokenize → remove_punctuation → stopwords → lemmatize

import re
import string
from typing import List, Tuple

import nltk
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Single shared instance — WordNetLemmatizer is stateless so reuse is fine
_lemmatizer = WordNetLemmatizer()


# =============================================================================
# NLTK DATA VERIFICATION
# =============================================================================

def require_nltk_data() -> None:
    """
    Verify that all required NLTK datasets are present on disk.

    Does NOT auto-download — the README tells you exactly what to run once.
    This way a launch fails loudly with a clear message instead of silently
    trying to download data mid-session.

    NLTK stores corpora as either folders or .zip files depending on how
    they were downloaded, so we check for both forms of each resource.

    Raises:
        RuntimeError with download instructions if anything is missing.
    """
    def _has_any(paths: List[str]) -> bool:
        for p in paths:
            try:
                nltk.data.find(p)
                return True
            except LookupError:
                pass
        return False

    missing = []

    # Sentence tokenizer — newer NLTK uses punkt_tab, older uses punkt
    if not _has_any(["tokenizers/punkt", "tokenizers/punkt_tab"]):
        missing.append("punkt")

    # English stopwords list
    if not _has_any(["corpora/stopwords", "corpora/stopwords.zip"]):
        missing.append("stopwords")

    # WordNet — used for lemmatization AND synonym expansion in glyph_main.py
    if not _has_any(["corpora/wordnet", "corpora/wordnet.zip"]):
        missing.append("wordnet")

    # Open Multilingual WordNet — required companion package for WordNet
    if not _has_any(["corpora/omw-1.4", "corpora/omw-1.4.zip"]):
        missing.append("omw-1.4")

    if missing:
        raise RuntimeError(
            "Missing NLTK data: " + ", ".join(missing) + "\n"
            "Run this once in your Anaconda environment:\n"
            "  python -m nltk.downloader punkt stopwords wordnet omw-1.4"
        )


def _get_stopwords() -> set:
    """
    Load and return the English stopwords set.
    Called after require_nltk_data() confirms the corpus exists.
    Returns a Python set for O(1) membership testing in the filter loop.
    """
    return set(stopwords.words("english"))


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

# Pre-compiled patterns — compiled once at module load, not per call
_multi_space      = re.compile(r"\s+")
_page_num_line    = re.compile(r"^\s*\d+\s*$")
_hyphen_linebreak = re.compile(r"(\w)-\s*\n\s*(\w)")

# Project Gutenberg wraps content between START and END marker lines.
# Both "THIS PROJECT GUTENBERG EBOOK" and "THE PROJECT GUTENBERG EBOOK"
# variants appear across different Gutenberg editions.
_gutenberg_start = re.compile(
    r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_gutenberg_end = re.compile(
    r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove the Project Gutenberg header and footer from .txt editions.

    Gutenberg wraps the actual literary text between:
        *** START OF THE PROJECT GUTENBERG EBOOK ... ***
        *** END OF THE PROJECT GUTENBERG EBOOK ... ***

    Everything outside those markers is legal boilerplate, donation text,
    and metadata that would pollute keyword scoring. We extract only the
    content between the markers.

    If the markers aren't found (non-Gutenberg files), returns the original
    text unchanged — safe to call on any .txt file.
    """
    if not text:
        return ""
    start_match = _gutenberg_start.search(text)
    end_match   = _gutenberg_end.search(text)
    if start_match and end_match and start_match.end() < end_match.start():
        return text[start_match.end():end_match.start()].strip()
    return text


def normalize_text(text: str) -> str:
    """
    Apply lightweight cleanup to fix common PDF and encoding artifacts.

    Intentionally conservative — we're reducing noise, not rewriting text.
    Heavy-handed normalization destroys passage readability.

    Steps:
      1. Remove soft hyphens (U+00AD) — invisible in most editors but cause
         tokenization failures ("ex\xadample" → "ex" + "ample")
      2. Rejoin hyphenated line breaks — "ex-\nexample" → "example"
         (very common in typeset/scanned PDFs where words wrap at hyphens)
      3. Drop page-number-only lines — bare numbers contribute noise
      4. Collapse all whitespace — tabs and multiple spaces → single space
    """
    if not text:
        return ""
    text = text.replace("\xad", "")                  # soft hyphens
    text = _hyphen_linebreak.sub(r"\1\2", text)      # hyphen line breaks
    lines = [ln for ln in text.splitlines()
             if not _page_num_line.match(ln)]         # page numbers
    text = "\n".join(lines)
    text = text.replace("\t", " ")
    text = _multi_space.sub(" ", text)                # collapse whitespace
    return text.strip()


# =============================================================================
# TOKENIZATION
# =============================================================================

def remove_punctuation(token: str) -> str:
    """
    Strip all punctuation characters from a single token.

    wordpunct_tokenize splits on punctuation boundaries, leaving some
    tokens as pure punctuation ("," or "."). After this step, those
    become empty strings and are filtered out in the next step.
    Matches the Lab 1 approach: remove_punctuation applied token-by-token.
    """
    return "".join(c for c in token if c not in string.punctuation)


def preprocess_text(text: str, strip_gutenberg: bool = False) -> Tuple[List[str], List[str]]:
    """
    Full preprocessing pipeline for a document.

    Returns two parallel outputs derived from the same source text:

    sentences — readable strings for display in the UI and evidence
                passages. Produced from normalized but NOT lowercased
                text so they look natural to the user.

    tokens    — flat list of preprocessed tokens for scoring/ranking.
                Lowercased, punctuation removed, stopwords removed,
                lemmatized. Consistent with preprocess_query() so that
                query tokens and document tokens are comparable.

    Args:
        text:            raw file text
        strip_gutenberg: if True, strip Gutenberg markers first.
                         Only for .txt — HTML loader handles this structurally.
    """
    if strip_gutenberg:
        text = strip_gutenberg_boilerplate(text)
    text = normalize_text(text)

    # Sentences from normalized-but-readable text (not lowercased)
    sentences = sent_tokenize(text)

    # Tokens from fully normalized, lowercased text
    lowered    = text.lower()
    stop_words = _get_stopwords()
    tokens     = wordpunct_tokenize(lowered)
    tokens     = [remove_punctuation(t) for t in tokens]
    tokens     = [t for t in tokens if t]                   # drop empty strings
    tokens     = [t for t in tokens if t not in stop_words]
    tokens     = [_lemmatizer.lemmatize(t) for t in tokens]

    return sentences, tokens


def preprocess_query(query: str) -> List[str]:
    """
    Preprocess a user query into tokens using the same pipeline as documents.

    Consistency is critical: if document tokens are lemmatized and
    stopword-filtered, the query tokens must be too — otherwise "running"
    in a query won't match "run" in a document's token list.
    """
    query      = normalize_text(query).lower()
    stop_words = _get_stopwords()
    toks       = wordpunct_tokenize(query)
    toks       = [remove_punctuation(t) for t in toks]
    toks       = [t for t in toks if t]
    toks       = [t for t in toks if t not in stop_words]
    toks       = [_lemmatizer.lemmatize(t) for t in toks]
    return toks
