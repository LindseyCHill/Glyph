# glyph_document.py
# Lindsey Hill - AIT626 Term Project - Glyph
#
# Defines the Document dataclass — the core data container passed through
# the entire Glyph pipeline (loading → preprocessing → scoring → display).
#
# A dataclass auto-generates __init__, __repr__, and __eq__ so we don't
# have to write boilerplate. Every field is typed so it's easy to see
# what a Document holds at each stage of the pipeline.

from dataclasses import dataclass, field
from typing import List


@dataclass
class Document:
    """
    One document in the Glyph library.

    Identity fields (set at load time by load_library):
        doc_id -- 1-based integer, stable within a session
        title  -- filename stem used for display (e.g. "Romeo_and_Juliet")
        path   -- full file path as str (not Path, for simplicity)

    Content fields (populated by preprocess_documents in glyph_main.py):
        raw_text  -- original text string, preserved for reference
        sentences -- readable sentence strings for evidence display;
                     NOT lowercased so they look natural in the UI
        tokens    -- flat list of preprocessed tokens (lowercase,
                     stopwords removed, lemmatized) used for scoring
    """

    # --- Set at load time ---
    doc_id: int
    title:  str
    path:   str

    # --- Set during preprocessing; default to empty so the object is
    #     valid even before preprocess_documents() has been called ---
    raw_text:  str       = ""
    sentences: List[str] = field(default_factory=list)
    tokens:    List[str] = field(default_factory=list)
