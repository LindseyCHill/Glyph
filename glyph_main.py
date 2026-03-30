# glyph_main.py
# Lindsey Hill - AIT626 Term Project - Glyph
#
# UI entry point. Run from Anaconda PowerShell with:
#   python glyph_main.py
#
# -----------------------------------------------------------------------
# RETRIEVAL PIPELINE (per query):
#
#  1. Document ranking   — TF-IDF cosine similarity (glyph_doc_scoring.py)
#                          IDF down-weights terms common across all docs so
#                          corpus-generic words ("art", "state", "man") don't
#                          hijack rankings. Specific terms matter more.
#
#  2. Must-have filter   — soft proper-noun filter (see extract_must_have)
#                          Capitalized query words that aren't question words
#                          are used as a SOFT filter: if a document contains
#                          NONE of the proper nouns, it's excluded. But only
#                          ONE of the must-haves needs to appear — not all.
#                          This prevents "Who is Juliet's father?" from failing
#                          because the possessive "Juliet's" doesn't tokenize
#                          the same way across documents.
#
#  3. Query expansion    — WordNet synonym expansion (nltk.corpus.wordnet)
#                          Action/concept words in the query are expanded so
#                          sentences using related vocabulary still score well.
#                          E.g. "die" → also matches "poison", "dead", "slay".
#                          Same WordNet corpus used for lemmatization in Lab 1.
#
#  4. Anchor/Signal scoring — passage-level sentence ranking
#                          Query tokens are split into two categories:
#                            ANCHOR tokens — proper nouns (Romeo, Mercutio).
#                              Confirm we're on-topic but shouldn't dominate
#                              scoring since they appear everywhere.
#                              Contribute only a small flat bonus.
#                            SIGNAL tokens — action/event/concept words
#                              (die, kill, house, treat, poison + synonyms).
#                              Drive the main score at SIGNAL_WEIGHT per hit.
#                          Prevents "O Romeo Romeo Romeo" from beating the
#                          actual death scene for "How does Romeo die?".
#
#  5. Boilerplate filter — suppress stage directions, scene headings,
#                          Gutenberg metadata, translator commentary
#
#  6. LLM ANSWER         — Ollama RAG via subprocess (Lab 4 pattern).
#                          Passages re-ranked by TF-IDF cosine before
#                          being sent as context. Shown first in output.
#
#  7. RAG ANSWER         — extractive stitch of top 3 passages
#
#  8. ONE-SENTENCE       — best single sentence by signal score + length
#
#  9. N-GRAM SUMMARY     — blended unigram + n-gram query-focused summary
#
# 10. Evidence passages  — retrieved sentences with scores
#
# 11. Logging            — every query + full output → glyph_log.txt
# -----------------------------------------------------------------------

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Tuple, Optional, Set
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from glyph_document import Document
from glyph_preprocess import require_nltk_data, preprocess_text, preprocess_query
from glyph_doc_scoring import tfidf_rank_documents
from glyph_loaders import load_text
from glyph_summarize_ngram import summarize_sentences
from glyph_rag_ollama import list_models, rag_answer


SUPPORTED_EXTS = {".txt", ".pdf", ".html", ".htm"}
LOG_PATH       = Path(__file__).resolve().parent / "glyph_log.txt"
NO_MODEL_LABEL = "(No model — extractive only)"

_lemmatizer = WordNetLemmatizer()


# =============================================================================
# SECTION 1: LOGGING
# =============================================================================

def _log_append(block: str) -> None:
    """Append one formatted block to glyph_log.txt. Failures are silent."""
    try:
        with LOG_PATH.open("a", encoding="utf-8", errors="ignore") as f:
            f.write(block)
            if not block.endswith("\n"):
                f.write("\n")
    except Exception:
        pass


def build_log_block(question: str, selection: str, output_text: str) -> str:
    """
    Format a log entry with box-drawing characters so entries are visually
    distinct in any plain-text viewer (Notepad, VS Code, etc.).
    """
    ts  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    top = "╔" + "═" * 88 + "╗"
    mid = "╠" + "═" * 88 + "╣"
    bot = "╚" + "═" * 88 + "╝"
    return (
        f"{top}\n"
        f"║  Glyph Log Entry\n"
        f"║  Timestamp : {ts}\n"
        f"║  Selection : {selection}\n"
        f"{mid}\n"
        f"║  QUESTION\n"
        f"║  {question}\n"
        f"{mid}\n"
        f"║  OUTPUT\n"
        f"{output_text.rstrip()}\n"
        f"{bot}\n\n"
    )


# =============================================================================
# SECTION 2: LIBRARY LOADING AND PREPROCESSING
# =============================================================================

def load_library(data_dir: str) -> List[Document]:
    """
    Scan data_dir and create Document objects for every supported file.
    Doc IDs are 1-based and stable within a session (sorted glob order).
    """
    docs   = []
    doc_id = 0
    for p in sorted(Path(data_dir).glob("*")):
        if p.is_dir() or p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        doc_id += 1
        docs.append(Document(doc_id=doc_id, title=p.stem, path=str(p)))
    return docs


def preprocess_documents(docs: List[Document]) -> None:
    """
    Load and preprocess every document in place.

    strip_gutenberg=True only for .txt files — the HTML loader already removes
    Gutenberg boilerplate structurally via BeautifulSoup, and the PDF loader
    doesn't encounter Gutenberg markers.
    """
    for d in docs:
        raw = load_text(Path(d.path))
        strip = Path(d.path).suffix.lower() == ".txt"
        d.raw_text = raw
        d.sentences, d.tokens = preprocess_text(raw, strip_gutenberg=strip)


# =============================================================================
# SECTION 3: MUST-HAVE TOKEN EXTRACTION (soft proper-noun filter)
# =============================================================================

# Question words that start sentences — capitalized but not named entities
_QUESTION_WORDS = {
    "what", "where", "who", "when", "why", "how", "which", "whose", "whom",
    "does", "did", "do", "is", "are", "was", "were", "can", "could",
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "and", "or",
}


def extract_must_have_tokens(raw_query: str) -> List[str]:
    """
    Extract proper-noun tokens from the query to use as a soft document filter.

    Uses capitalization as a lightweight named-entity proxy. Words that are
    capitalized but not common question/article words are treated as named
    entities that should appear somewhere in a relevant document.

    IMPORTANT — this is a SOFT filter (see _doc_contains_any):
        We only require that at least ONE of these tokens appears in the
        document, not ALL of them. This prevents possessives and compound
        names from causing false-negative matches.

        Example: "Who is Juliet's father?" → must_tokens = ["juliet"]
            The apostrophe-s gets stripped in tokenization so "juliet" is
            the right form to check. Only one token, so any doc with "juliet"
            passes.

        Example: "Who wrote The Communist Manifesto?" → must_tokens = ["communist", "manifesto"]
            Only ONE of these needs to appear. The Manifesto doc has both;
            other docs have neither — so it works as a filter.

    Args:
        raw_query: original query string before preprocessing

    Returns:
        List of lowercase token strings (without punctuation).
    """
    must = []
    # Strip punctuation that wordpunct_tokenize would keep, then split
    clean = raw_query.replace("?","").replace("!","").replace(",","").replace("'s","")
    for w in clean.split():
        w_clean = w.strip("\"'()[]")
        if (len(w_clean) >= 2
                and w_clean[0].isupper()
                and any(c.isalpha() for c in w_clean)
                and w_clean.lower() not in _QUESTION_WORDS):
            must.append(w_clean.lower())
    return must


def _doc_contains_any(doc: Document, must_tokens: List[str]) -> bool:
    """
    Soft must-have filter: return True if doc contains ANY of the must-have tokens.

    Using ANY instead of ALL prevents proper-noun parsing edge cases
    (possessives, hyphenated names, title words) from filtering out
    documents that clearly are relevant. If must_tokens is empty, all
    documents pass.
    """
    if not must_tokens:
        return True
    doc_set = set(doc.tokens)
    return any(t in doc_set for t in must_tokens)


# =============================================================================
# SECTION 4: QUERY EXPANSION (WordNet synonyms — same corpus as Lab 1)
# =============================================================================

def _wordnet_synonyms(word: str, max_synonyms: int = 8) -> Set[str]:
    """
    Look up synonyms for a word via WordNet synsets.

    Uses the same WordNet corpus as Lab 1 lemmatization but here for
    query expansion: collect lemma names across synsets, clean them,
    and return a set of related terms.

    Why this helps with retrieval:
        A user asks "How does Romeo die?" using the word "die", but the
        actual death scene uses "poison", "dead", "tomb", "vault".
        Without expansion those sentences score 0 (no shared tokens).
        With expansion, "die" pulls in related vocabulary so the right
        passages surface.

    Args:
        word:         single lowercase token to expand
        max_synonyms: cap to avoid bloating the query with marginal terms

    Returns:
        Set of lowercase lemmatized synonym strings (original word excluded).
    """
    synonyms: Set[str] = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if " " not in name and name != word and name.isalpha():
                synonyms.add(_lemmatizer.lemmatize(name))
            if len(synonyms) >= max_synonyms:
                return synonyms
    return synonyms


def _should_expand(word: str) -> bool:
    """
    Return True if this word is a good candidate for synonym expansion.

    Skip: very short words, words with no WordNet entry, words that only
    appear as adjectives or adverbs (less useful to expand). We only
    expand verbs and nouns because those carry the action/concept meaning
    that's most likely to have alternative vocabulary in the source text.
    """
    if len(word) < 4:
        return False
    synsets = wordnet.synsets(word)
    if not synsets:
        return False
    return bool({s.pos() for s in synsets} & {"v", "n"})


def expand_query_tokens(q_tokens: List[str], raw_query: str) -> List[str]:
    """
    Expand query tokens with WordNet synonyms and rule-based additions.

    Stage 1 — Rule-based:
        "house"/"family" + Romeo/Juliet → add montague, capulet.
        Explicit name preservation for names that might be stripped.

    Stage 2 — WordNet expansion:
        For each ORIGINAL query token (not newly added ones) that passes
        _should_expand(), add up to 8 synonyms from WordNet. Expanding
        only original tokens prevents synonym chains (synonyms of synonyms).

    Args:
        q_tokens:  preprocessed query tokens (from preprocess_query)
        raw_query: original query string (for rule-based name checks)

    Returns:
        Deduplicated list of all query tokens + expansion terms.
    """
    q_set = set(q_tokens)
    raw_lower = raw_query.lower()

    # Rule-based family name expansion for Romeo & Juliet
    if ("house" in q_set or "family" in q_set) and ("romeo" in q_set or "juliet" in q_set):
        q_set.update({"montague", "capulet"})
    if "montague" in raw_lower:
        q_set.add("montague")
    if "capulet" in raw_lower:
        q_set.add("capulet")

    # WordNet expansion — only on the original tokens, not newly added ones
    for token in list(q_tokens):
        if _should_expand(token):
            q_set.update(_wordnet_synonyms(token))

    return list(q_set)


# =============================================================================
# SECTION 5: ANCHOR / SIGNAL CLASSIFICATION
# =============================================================================

def classify_query_tokens(
    q_tokens_expanded: List[str],
    must_tokens: List[str],
) -> Tuple[Set[str], Set[str]]:
    """
    Split the expanded token set into ANCHOR tokens and SIGNAL tokens.

    WHY THIS EXISTS — the "Romeo Romeo Romeo" problem:
        Under a flat coverage score, a sentence like "O Romeo Romeo Romeo
        wherefore art thou Romeo" scores 28.0 for the query "How does Romeo
        die?" because "romeo" and a synonym of "die" both appear.
        But the ACTUAL death scene — "he drank the poison and fell dead" —
        scores 0 because "romeo" doesn't appear in it.

    THE FIX:
        ANCHOR tokens  = proper nouns from must_tokens (romeo, mercutio, john,
                         sun, tzu...). These confirm the sentence is on-topic
                         but are NOT good discriminators because they appear
                         everywhere in their document. Contribute only a small
                         flat ANCHOR_BONUS (3.0) regardless of frequency.

        SIGNAL tokens  = everything else in the expanded set — the action,
                         event, and concept words the question is ASKING ABOUT,
                         plus all their WordNet synonyms (die, dead, poison,
                         slay, kill, house, treat...). Drive the main score
                         at SIGNAL_WEIGHT (12.0) per unique token hit.

    Result: "he drank the poison and fell dead" → 2 unique signals (poison,
    dead) = 12*2 + 6 multi-bonus + 3 anchor = 33.0, beats "O Romeo Romeo
    Romeo" → 0 signals = just 3.0 anchor bonus.
    """
    anchor_set = set(must_tokens)
    signal_set = {t for t in q_tokens_expanded if t not in anchor_set}
    return anchor_set, signal_set


# Scoring weights — tuned by testing against the Romeo/Mercutio/death queries
SIGNAL_WEIGHT = 12.0   # points per unique signal token matched
ANCHOR_BONUS  =  3.0   # flat bonus if ANY anchor token is present
MULTI_BONUS   =  6.0   # extra bonus for matching 2+ distinct signal tokens
REPEAT_CAP    =  1.5   # max extra credit for repeated signal token hits


# =============================================================================
# SECTION 6: PASSAGE SCORING
# =============================================================================

def _sentence_tokens(sentence: str) -> List[str]:
    """
    Tokenize a sentence using the same preprocessing pipeline as the query,
    so sentence tokens and query tokens are directly comparable.
    """
    return preprocess_query(sentence)


# Boilerplate markers — sentences matching these are filtered out before scoring.
# The Art of War has extensive translator commentary in [square brackets] that
# tends to score highly because "fight" / "enemy" appear in it. Stage directions
# from Romeo & Juliet and Gutenberg metadata are also suppressed.
_BOILERPLATE_STARTS = ("scene", "act ", "room in", "hall in", "a room", "a hall")
_BOILERPLATE_CONTAINS = (
    "project gutenberg", "gutenberg", "table of contents", "contents",
    "preface", "introduction", "translated", "translation", "etext",
    "edited by", "publisher", "copyright", "license", "scanner",
    "proofreading", "chapter", "volume", "footnote", "critical notes",
    "lionel giles",
)


def _is_boilerplate_sentence(sentence: str) -> bool:
    """
    Return True if this sentence is boilerplate that should be excluded.

    Filters:
      - Short (≤8 words) lines starting with stage/scene direction words
      - Lines containing Gutenberg metadata markers
      - Translator commentary in [square brackets] (Art of War footnotes)
        These have high keyword overlap but aren't part of the primary text.
    """
    s = sentence.strip().lower()

    # Stage direction: short line starting with scene/act/room words
    if len(s.split()) <= 8 and s.startswith(_BOILERPLATE_STARTS):
        return True

    # Gutenberg and editorial metadata markers
    if any(m in s for m in _BOILERPLATE_CONTAINS):
        return True

    # Translator commentary: entire sentence is wrapped in square brackets
    # e.g. "[Chang Yu says: If he can fight, he advances...]"
    stripped = s.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return True

    return False


def score_sentence_for_query(
    sentence:   str,
    signal_set: Set[str],
    anchor_set: Set[str],
) -> float:
    """
    Score a sentence using anchor/signal weighted scoring.

    A sentence with 0 signal matches AND 0 anchor matches scores 0.0 and
    will be excluded from results. A sentence needs at least one anchor hit
    OR at least one signal hit to appear in evidence at all.

    Formula (for non-boilerplate sentences with at least one match):
        score = (unique_signal_hits * SIGNAL_WEIGHT)
              + (MULTI_BONUS  if unique_signal_hits >= 2)
              + (capped bonus for repeated signal hits beyond unique count)
              + (ANCHOR_BONUS if any anchor token appears at all)
    """
    if _is_boilerplate_sentence(sentence):
        return 0.0

    s_tokens = _sentence_tokens(sentence)
    if not s_tokens:
        return 0.0

    s_set          = set(s_tokens)
    signal_matches = s_set & signal_set
    has_anchor     = bool(s_set & anchor_set)

    if not signal_matches and not has_anchor:
        return 0.0

    unique_signals = len(signal_matches)
    score = unique_signals * SIGNAL_WEIGHT

    # Multi-signal bonus: reward sentences that cover several signal concepts
    if unique_signals >= 2:
        score += MULTI_BONUS

    # Repetition bonus: small reward for repeated signal token occurrences
    # (capped so one word repeated 10 times doesn't inflate the score)
    repeat_hits = sum(1 for t in s_tokens if t in signal_set)
    score += min(REPEAT_CAP, max(0, repeat_hits - unique_signals) * 0.25)

    # Anchor bonus: small reward for mentioning the subject of the question
    if has_anchor:
        score += ANCHOR_BONUS

    return score


def top_passages_for_doc(
    doc:        Document,
    signal_set: Set[str],
    anchor_set: Set[str],
    top_k:      int = 8,
) -> List[Tuple[int, str, float]]:
    """
    Retrieve the top_k most relevant sentences from a document.

    Scores every sentence, filters out score=0, takes top_k by score,
    then restores reading order so the returned passages read coherently.

    Args:
        doc:        Document with .sentences populated
        signal_set: signal tokens (drive scoring)
        anchor_set: anchor tokens (small flat bonus)
        top_k:      maximum number of passages to return

    Returns:
        List of (sentence_index, text, score) in original reading order.
    """
    scored = [
        (i, s, score_sentence_for_query(s, signal_set, anchor_set))
        for i, s in enumerate(doc.sentences)
    ]
    scored = [(i, s, sc) for i, s, sc in scored if sc > 0]
    scored.sort(key=lambda x: x[2], reverse=True)
    best = scored[:top_k]
    best.sort(key=lambda x: x[0])   # restore reading order
    return best


def stitch_summary(passages: List[Tuple[int, str, float]], max_sentences: int = 3) -> str:
    """
    Produce an extractive RAG-style answer by joining the top passages.
    Passages are already in reading order from top_passages_for_doc().
    """
    if not passages:
        return "(No matching passages found.)"
    return " ".join(s.strip() for _, s, _ in passages[:max_sentences])


def extract_one_sentence_answer(
    passages:   List[Tuple[int, str, float]],
    signal_set: Set[str],
    anchor_set: Set[str],
) -> str:
    """
    Pick the single best sentence as a direct one-sentence answer.

    Applies a length penalty (0.3 points per word over 18) to prefer
    concise, direct answers over long narrative sentences that happen
    to score well on signal coverage.
    """
    if not passages:
        return "(No answer found.)"

    best_sent  = None
    best_score = float("-inf")

    for _idx, sent, base_sc in passages:
        s = sent.strip()
        if not s or _is_boilerplate_sentence(s):
            continue
        length_penalty = max(0, len(s.split()) - 18) * 0.3
        adjusted       = base_sc - length_penalty
        if adjusted > best_score:
            best_score = adjusted
            best_sent  = s

    return best_sent if best_sent else passages[0][1].strip()


# =============================================================================
# SECTION 7: QUERY PREPARATION AND RUNNERS
# =============================================================================

def _prepare_query(raw_query: str) -> Tuple[List[str], Set[str], Set[str]]:
    """
    Shared query preparation — called by both query runner functions.

    Steps:
      1. Extract must-have anchor tokens (capitalized proper nouns)
      2. Preprocess the raw query into base tokens
      3. Expand with WordNet synonyms + rule-based additions
      4. Classify tokens into anchor set and signal set

    Returns:
        (must_tokens, signal_set, anchor_set)
        must_tokens is kept separate for the document-level soft filter.
    """
    must_tokens            = extract_must_have_tokens(raw_query)
    q_tokens               = preprocess_query(raw_query)
    q_tokens_expanded      = expand_query_tokens(q_tokens, raw_query)
    anchor_set, signal_set = classify_query_tokens(q_tokens_expanded, must_tokens)
    return must_tokens, signal_set, anchor_set


def _format_doc_block(
    doc:        Document,
    raw_query:  str,
    signal_set: Set[str],
    anchor_set: Set[str],
    llm_answer: Optional[str] = None,
) -> str:
    """
    Build the full output block for one document.

    Output order (LLM answer first when available):
      1. LLM ANSWER           — Ollama-generated from retrieved context
      2. RAG ANSWER           — extractive stitch of top 3 passages
      3. ONE-SENTENCE ANSWER  — best single sentence
      4. N-GRAM SUMMARY       — blended unigram + n-gram re-ranking
      5. EVIDENCE PASSAGES    — all retrieved sentences with scores
    """
    passages        = top_passages_for_doc(doc, signal_set, anchor_set, top_k=8)
    rag_answer_text = stitch_summary(passages, max_sentences=3)
    one_sentence    = extract_one_sentence_answer(passages, signal_set, anchor_set)

    # N-gram summary is re-ranked from the already-retrieved passages
    # using a combined unigram + query n-gram score
    ng_best = summarize_sentences(
        sentences  = [s for _i, s, _sc in passages],
        doc_tokens = doc.tokens,
        query_text = " ".join(list(signal_set) + list(anchor_set)),
        top_n      = 3,
        ngram_n    = 3,
        alpha      = 0.6,
    )
    ngram_summary = (
        " ".join(s.strip() for _, s, _ in ng_best)
        if ng_best else "(No n-gram summary found.)"
    )

    out: List[str] = []
    out.append("\n" + "=" * 78 + "\n")
    out.append(f"DOCUMENT: {doc.doc_id}: {doc.title}\n")
    out.append("=" * 78 + "\n")
    out.append(f"\nQUESTION:\n{raw_query}\n")

    if llm_answer is not None:
        out.append("\nLLM ANSWER (Ollama RAG — generated from retrieved passages):\n")
        out.append(llm_answer + "\n")

    out.append("\nRAG ANSWER (stitched extractive):\n")
    out.append(rag_answer_text + "\n")

    out.append("\nONE-SENTENCE ANSWER (extractive):\n")
    out.append(one_sentence + "\n")

    out.append("\nN-GRAM SUMMARY ANSWER (query-focused extractive):\n")
    out.append(ngram_summary + "\n")

    out.append("\nEVIDENCE PASSAGES USED (retrieved sentences):\n")
    if not passages:
        out.append("(No good matches found in this document.)\n")
    else:
        for _, sent, sc in passages:
            out.append(f"  [{sc:.2f}]  {sent.strip()}\n")

    return "".join(out)


def run_query_all_docs(
    docs:      List[Document],
    raw_query: str,
    model:     Optional[str] = None,
    top_docs:  int = 2,
) -> str:
    """
    Run a query against the full library.

    Pipeline:
      1. Prepare query (expand tokens, classify anchor/signal)
      2. Rank all documents by TF-IDF cosine similarity
      3. Apply soft must-have filter (ANY proper noun must appear)
      4. Take top_docs documents
      5. Retrieve and score passages for each selected document
      6. Call Ollama if a model is selected
      7. Format and return the full result string
    """
    raw_query = raw_query.strip()
    if not raw_query:
        return "Type a question first.\n"

    q_base = preprocess_query(raw_query)
    if not q_base:
        return "Your question turned into nothing after preprocessing. Try rephrasing.\n"

    must_tokens, signal_set, anchor_set = _prepare_query(raw_query)
    all_tokens = list(signal_set | anchor_set)

    ranked = tfidf_rank_documents(docs, all_tokens)
    # Soft filter: document must contain at least ONE must-have proper noun
    ranked = [(d, sc) for d, sc in ranked if _doc_contains_any(d, must_tokens)]
    ranked = ranked[:top_docs]

    if not ranked:
        return (
            "No documents matched the required terms in your question.\n"
            f"(Proper nouns searched for: {must_tokens})\n"
        )

    use_llm = bool(model and model != NO_MODEL_LABEL)

    out = ["Top documents:\n"]
    for d, sc in ranked:
        out.append(f"  {d.doc_id}: {d.title}  (score={sc:.4f})\n")

    for d, _ in ranked:
        llm_text: Optional[str] = None
        if use_llm:
            passages       = top_passages_for_doc(d, signal_set, anchor_set, top_k=8)
            ok, llm_result = rag_answer(raw_query, passages, d.title, model)
            llm_text       = llm_result if ok else f"(Ollama error: {llm_result})"
        out.append(_format_doc_block(d, raw_query, signal_set, anchor_set, llm_answer=llm_text))

    return "".join(out)


def run_query_one_doc(
    docs:      List[Document],
    raw_query: str,
    doc_id:    int,
    model:     Optional[str] = None,
) -> str:
    """
    Run a query against a single user-selected document.

    Skips document ranking (user already chose the document). Still applies
    the full query expansion and anchor/signal scoring pipeline.
    """
    raw_query = raw_query.strip()
    if not raw_query:
        return "Type a question first.\n"

    q_base = preprocess_query(raw_query)
    if not q_base:
        return "Your question turned into nothing after preprocessing. Try rephrasing.\n"

    must_tokens, signal_set, anchor_set = _prepare_query(raw_query)

    doc = next((d for d in docs if d.doc_id == doc_id), None)
    if not doc:
        return "Selected document not found.\n"

    llm_text: Optional[str] = None
    if model and model != NO_MODEL_LABEL:
        passages       = top_passages_for_doc(doc, signal_set, anchor_set, top_k=8)
        ok, llm_result = rag_answer(raw_query, passages, doc.title, model)
        llm_text       = llm_result if ok else f"(Ollama error: {llm_result})"

    header = f"Selected document: {doc.doc_id}: {doc.title}\n" + "-" * 78 + "\n"
    return header + _format_doc_block(doc, raw_query, signal_set, anchor_set, llm_answer=llm_text)


# =============================================================================
# SECTION 8: TKINTER UI
# =============================================================================

class GlyphApp(tk.Tk):
    """
    Main application window.

    Layout:
      Row 1: Question: [entry] [Run] [Open Log]  Search: [doc dropdown] [Refresh Library]
      Row 2: LLM Model: [model dropdown] [Refresh Models]  [Ollama status label]
      Output: scrollable text box

    Design notes:
      - Run button is immediately right of the question field for quick keyboard use.
      - Ollama is probed in a background thread at startup so the window opens
        immediately without blocking on the network call.
      - LLM queries run in a background thread to keep the UI responsive
        (Ollama generation can take 30+ seconds on CPU).
      - All Tkinter widget updates from background threads use self.after()
        because Tkinter is not thread-safe.
    """

    def __init__(self):
        super().__init__()
        self.title("Glyph")
        self.geometry("980x740")

        try:
            require_nltk_data()
        except Exception as e:
            messagebox.showerror("Glyph startup error", str(e))
            raise

        self.data_dir = "data"
        self.docs: List[Document] = []
        try:
            self.reload_library()
        except Exception as e:
            messagebox.showerror("Glyph library error",
                                 f"Failed to load/preprocess documents.\n\n{e}")
            raise

        # ── Row 1: question entry, action buttons, document selector ──────
        row1 = ttk.Frame(self)
        row1.pack(fill="x", padx=10, pady=(10, 2))

        ttk.Label(row1, text="Question:").pack(side="left")
        self.query_entry = ttk.Entry(row1, width=46)
        self.query_entry.pack(side="left", padx=(6, 4))
        self.query_entry.focus_set()

        ttk.Button(row1, text="Run",      command=self.on_run      ).pack(side="left", padx=(0, 6))
        ttk.Button(row1, text="Open Log", command=self.on_open_log ).pack(side="left", padx=(0, 10))

        ttk.Label(row1, text="Search:").pack(side="left")
        self.doc_var = tk.StringVar()
        self.doc_dropdown = ttk.Combobox(
            row1, textvariable=self.doc_var,
            values=self.build_dropdown_options(),
            state="readonly", width=34,
        )
        self.doc_dropdown.current(0)
        self.doc_dropdown.pack(side="left", padx=(6, 6))
        ttk.Button(row1, text="Refresh Library", command=self.on_refresh).pack(side="left")

        # ── Row 2: Ollama model selector and status ────────────────────────
        row2 = ttk.Frame(self)
        row2.pack(fill="x", padx=10, pady=(2, 8))

        ttk.Label(row2, text="LLM Model:").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            row2, textvariable=self.model_var,
            values=[NO_MODEL_LABEL],
            state="readonly", width=38,
        )
        self.model_dropdown.current(0)
        self.model_dropdown.pack(side="left", padx=(6, 6))
        ttk.Button(row2, text="Refresh Models", command=self.on_refresh_models).pack(side="left")

        self.ollama_status_var = tk.StringVar(value="Checking Ollama...")
        self.ollama_status_lbl = ttk.Label(
            row2, textvariable=self.ollama_status_var, foreground="gray"
        )
        self.ollama_status_lbl.pack(side="left", padx=(12, 0))

        # ── Output text box ────────────────────────────────────────────────
        self.output_box = tk.Text(self, wrap="word")
        self.output_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.write_output(
            f"Loaded {len(self.docs)} documents from ./{self.data_dir}\n"
            f"Log: {LOG_PATH}\n\n"
            "Type a question and press Run (or Enter).\n"
        )

        # Probe Ollama in background — don't block startup
        threading.Thread(target=self._probe_ollama_background, daemon=True).start()
        self.bind("<Return>", lambda _e: self.on_run())

    # ── Ollama ─────────────────────────────────────────────────────────────

    def _probe_ollama_background(self) -> None:
        """Background thread: probe Ollama and populate the model dropdown."""
        ok, models, err = list_models()
        if not ok or not models:
            msg = err or "No models found. Is Ollama running?"
            self.after(0, lambda: self._set_ollama_unavailable(msg))
        else:
            self.after(0, lambda: self._set_ollama_available(models))

    def _set_ollama_unavailable(self, message: str) -> None:
        self.model_dropdown["values"] = [NO_MODEL_LABEL]
        self.model_dropdown.current(0)
        self.ollama_status_var.set(f"⚠  {message}")
        self.ollama_status_lbl.configure(foreground="red")

    def _set_ollama_available(self, models: List[str]) -> None:
        self.model_dropdown["values"] = [NO_MODEL_LABEL] + models
        self.model_dropdown.current(1)   # auto-select first available model
        count = len(models)
        self.ollama_status_var.set(
            f"✓  Ollama connected  ({count} model{'s' if count != 1 else ''} available)"
        )
        self.ollama_status_lbl.configure(foreground="green")

    def on_refresh_models(self) -> None:
        """Re-probe Ollama and refresh the model dropdown."""
        self.ollama_status_var.set("Refreshing...")
        self.ollama_status_lbl.configure(foreground="gray")
        threading.Thread(target=self._probe_ollama_background, daemon=True).start()

    # ── Library ────────────────────────────────────────────────────────────

    def build_dropdown_options(self) -> List[str]:
        opts = [f"All Documents ({len(self.docs)})"]
        opts.extend(f"{d.doc_id}: {d.title}" for d in self.docs)
        return opts

    def reload_library(self) -> None:
        """Re-scan data_dir and preprocess all documents."""
        self.docs = load_library(self.data_dir)
        preprocess_documents(self.docs)

    def write_output(self, text: str) -> None:
        """Replace the entire output box contents."""
        self.output_box.delete("1.0", "end")
        self.output_box.insert("end", text)

    # ── Event handlers ─────────────────────────────────────────────────────

    def on_refresh(self) -> None:
        try:
            self.reload_library()
            self.doc_dropdown["values"] = self.build_dropdown_options()
            self.doc_dropdown.current(0)
            self.write_output(f"Refreshed. {len(self.docs)} documents loaded.\n")
        except Exception as e:
            messagebox.showerror("Refresh error", str(e))

    def on_open_log(self) -> None:
        """Open glyph_log.txt in the system default text editor (Windows)."""
        try:
            import os
            if not LOG_PATH.exists():
                _log_append("=== Glyph Log Created ===\n\n")
            os.startfile(str(LOG_PATH))  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror("Open Log error", str(e))

    def on_run(self) -> None:
        """
        Dispatch a query. LLM queries run in a background thread;
        extractive-only runs synchronously (fast enough for UI).
        """
        query = self.query_entry.get().strip()
        if not query:
            self.write_output("Type a question first.\n")
            return

        selection = self.doc_var.get().strip()
        model     = self.model_var.get().strip()
        use_llm   = bool(model and model != NO_MODEL_LABEL)

        if use_llm:
            self.write_output("Running — waiting for Ollama response...\n")
            threading.Thread(
                target=self._run_query_thread,
                args=(query, selection, model),
                daemon=True,
            ).start()
        else:
            self._execute_query(query, selection, model=None)

    def _run_query_thread(self, query: str, selection: str, model: str) -> None:
        """Background thread body. Updates UI via after() when done."""
        try:
            result = self._build_result(query, selection, model=model)
        except Exception as e:
            result = f"[ERROR]\n{e}"
        self.after(0, lambda: self._finish_query(query, selection, result))

    def _execute_query(self, query: str, selection: str, model: Optional[str]) -> None:
        """Synchronous query execution for extractive-only mode."""
        try:
            result = self._build_result(query, selection, model=model)
        except Exception as e:
            messagebox.showerror("Run error", str(e))
            _log_append(build_log_block(query, selection, f"[ERROR]\n{e}"))
            return
        self._finish_query(query, selection, result)

    def _build_result(self, query: str, selection: str, model: Optional[str]) -> str:
        """Dispatch to the correct query runner based on dropdown selection."""
        if selection.startswith("All Documents"):
            return run_query_all_docs(self.docs, query, model=model, top_docs=2)
        try:
            doc_id = int(selection.split(":")[0])
        except Exception:
            return "Could not parse the selected document.\n"
        return run_query_one_doc(self.docs, query, doc_id, model=model)

    def _finish_query(self, query: str, selection: str, result: str) -> None:
        """Write result to the output box and append to log. Always on main thread."""
        self.write_output(result)
        _log_append(build_log_block(query, selection, result))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = GlyphApp()
    app.mainloop()
