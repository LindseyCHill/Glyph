# glyph_doc_scoring.py
# Lindsey Hill - AIT626 Term Project - Glyph
#
# Document-level ranking: which documents are most relevant to a query?
#
# Two methods are provided so the writeup can compare them:
#
#   keyword_rank_documents() — BASELINE
#       Raw frequency sum. Simple, fast, easy to understand. Used as the
#       Checkpoint 1 / Lab 3 baseline. Suffers from "common word hijack":
#       a word like "art" appears everywhere in Art of War, so any query
#       containing "art" would rank it first regardless of topic.
#
#   tfidf_rank_documents() — IMPROVED (used by the main pipeline)
#       TF-IDF cosine similarity. IDF down-weights terms that appear in
#       many documents (like "art", "state", "man") so corpus-wide common
#       terms stop dominating. Specific terms like "communist", "poison",
#       "wallpaper" get higher weight because they only appear in one doc.
#       Matches the Lab 4 retrieval approach.

from __future__ import annotations

from collections import Counter
from math import log, sqrt
from typing import Dict, List, Tuple

from nltk.probability import FreqDist

from glyph_document import Document


# =============================================================================
# BASELINE: KEYWORD FREQUENCY SUM
# =============================================================================

def keyword_rank_documents(
    documents: List[Document],
    query_tokens: List[str],
) -> List[Tuple[Document, float]]:
    """
    Baseline document ranking: sum raw keyword frequencies.

    For each document, count how many times each query token appears.
    Score = total count across all query tokens.

    Kept for Checkpoint 2 comparison. Do NOT use for the main pipeline —
    it lets high-frequency common words dominate the ranking.
    """
    ranked: List[Tuple[Document, float]] = []
    for doc in documents:
        fdist = FreqDist(doc.tokens)
        score = sum(fdist[t] for t in query_tokens)  # 0 if token not present
        ranked.append((doc, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =============================================================================
# IMPROVED: TF-IDF + COSINE SIMILARITY
# =============================================================================

def _idf_map(documents: List[Document]) -> Dict[str, float]:
    """
    Compute smoothed IDF weights for the full document collection.

    Formula: idf(t) = log((N + 1) / (df(t) + 1)) + 1

    The +1 smoothing in numerator and denominator prevents division-by-zero
    and keeps IDF from exploding for very rare terms. The trailing +1 ensures
    IDF is always ≥ 1 even for terms that appear in every document.

    df(t) = number of documents containing t (set-based, not count-based)
    N     = total number of documents in the library
    """
    N  = len(documents)
    df: Counter[str] = Counter()

    for d in documents:
        # Use set() so repeated tokens in one document only count once
        df.update(set(d.tokens))

    return {
        term: log((N + 1) / (docfreq + 1)) + 1.0
        for term, docfreq in df.items()
    }


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Build a sparse TF-IDF vector for a token list.

    TF  = raw count of the term in this token list (not normalized here —
          cosine normalization in _cosine() handles the length effect)
    TFIDF = TF * IDF

    Returns a dict {term: weight} — sparse so only present terms are stored.
    Terms not in idf (unseen at index-build time) are silently skipped.
    """
    tf  = Counter(tokens)
    return {
        term: float(count) * idf[term]
        for term, count in tf.items()
        if term in idf
    }


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Cosine similarity between two sparse TF-IDF vectors.

    Iterates over the smaller dict for efficiency (only shared terms
    contribute to the dot product). Pre-computes norms from each dict's
    own values so no dense vector is ever materialized.

    Returns 0.0 if either vector is empty (no shared vocabulary → no similarity).
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a  # iterate over the smaller one

    dot = sum(av * b.get(k, 0.0) for k, av in a.items())
    na  = sqrt(sum(v * v for v in a.values()))
    nb  = sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def tfidf_rank_documents(
    documents: List[Document],
    query_tokens: List[str],
) -> List[Tuple[Document, float]]:
    """
    Improved document ranking: TF-IDF cosine similarity.

    Steps:
      1. Compute IDF weights over the full library (corpus-wide statistics).
      2. Build a TF-IDF vector for each document.
      3. Build a TF-IDF vector for the query.
      4. Rank documents by cosine similarity between their vector and the query.

    Why this is better than keyword_rank_documents():
        IDF penalizes terms that appear in many documents. So "state" appearing
        constantly in Art of War doesn't make it rank first for every query —
        its IDF weight is low because "state" appears in the Manifesto too.
        Meanwhile "communist" has high IDF because it only appears in one doc,
        so it strongly differentiates the Manifesto from other documents.

    Args:
        documents:    list of preprocessed Document objects
        query_tokens: preprocessed, expanded query token list

    Returns:
        List of (Document, cosine_score) sorted descending by score.
    """
    if not documents or not query_tokens:
        return [(d, 0.0) for d in documents]

    idf   = _idf_map(documents)
    q_vec = _tfidf_vector(query_tokens, idf)

    ranked = [
        (d, _cosine(_tfidf_vector(d.tokens, idf), q_vec))
        for d in documents
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
