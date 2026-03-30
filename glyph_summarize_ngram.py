# glyph_summarize_ngram.py
# Lindsey Hill - AIT626 Term Project - Glyph
#
# Extractive summarization — selects the most relevant sentences from the
# already-retrieved passage set and presents them as a summary answer.
#
# Two scoring components are blended:
#
#   Unigram salience (baseline, Lab 2 pattern):
#       Sentences containing high-frequency document terms are considered
#       "central" to the text. A word that appears often in the document
#       is probably important. Score = sum of normalized term frequencies
#       for each word in the sentence.
#
#   Query n-gram overlap (improvement):
#       Measures how many n-grams from the query appear in the sentence.
#       Ensures the summary stays relevant to the question, not just to the
#       document's general topic.
#
#   Final score = alpha * unigram_salience + (1 - alpha) * ngram_overlap
#
#   alpha=0.6 means 60% content-centrality, 40% query-relevance.
#   Tunable — higher alpha → more "central to the document"; lower → more
#   "directly answers the question".

from __future__ import annotations

from heapq import nlargest
from typing import Dict, List, Tuple

from nltk.probability import FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk import ngrams as nltk_ngrams


# =============================================================================
# COMPONENT SCORERS
# =============================================================================

def weighted_freq(tokens: List[str]) -> Dict[str, float]:
    """
    Build a normalized unigram frequency table from a token list.

    Normalizes counts to [0, 1] by dividing by the maximum count, so the
    most common term gets weight 1.0 and all others are proportional to it.
    This prevents very long documents from having inflated raw counts.

    Args:
        tokens: preprocessed document tokens (lowercase, lemmatized)

    Returns:
        Dict mapping token → normalized frequency weight in [0, 1].
    """
    fdist = FreqDist(tokens)
    if not fdist:
        return {}
    max_count = fdist.most_common(1)[0][1]
    return {w: c / max_count for w, c in fdist.items()}


def score_sentence_unigram(sentence: str, freq: Dict[str, float]) -> float:
    """
    Score a sentence by summing normalized unigram weights of its words.

    Uses surface tokenization (wordpunct_tokenize, not the full preprocessing
    pipeline) because we want to score the raw sentence text, and the freq
    table already maps to lowercase so lowercasing is sufficient.

    Higher score = sentence uses more of the document's important vocabulary
    = more "central" to the document's content.
    """
    score = 0.0
    for w in wordpunct_tokenize(sentence.lower()):
        score += freq.get(w, 0.0)
    return score


def _make_ngrams(text: str, n: int) -> set:
    """
    Produce the set of n-gram strings from a text.

    Uses wordpunct_tokenize on lowercase text, then generates n-grams
    as space-joined strings so they're easy to compare with intersection().

    Returns empty set if text is shorter than n tokens.
    """
    toks = wordpunct_tokenize(text.lower())
    if len(toks) < n:
        return set()
    return {" ".join(g) for g in nltk_ngrams(toks, n)}


def score_sentence_ngram(sentence: str, query_ngrams: set, n: int) -> float:
    """
    Count how many query n-grams appear in the sentence.

    Simple overlap count — works well for short, specific QA queries because
    the query phrases are distinctive and rarely appear by coincidence.

    Args:
        sentence:     raw sentence string
        query_ngrams: set of n-gram strings from the query
        n:            n-gram size (must match how query_ngrams was built)

    Returns:
        Float count of shared n-grams (0.0 if query_ngrams is empty).
    """
    if not query_ngrams:
        return 0.0
    return float(len(_make_ngrams(sentence, n) & query_ngrams))


# =============================================================================
# MAIN SUMMARIZER
# =============================================================================

def summarize_sentences(
    sentences:  List[str],
    doc_tokens: List[str],
    query_text: str,
    top_n:      int   = 5,
    ngram_n:    int   = 3,
    alpha:      float = 0.6,
) -> List[Tuple[int, str, float]]:
    """
    Score and rank sentences by a blend of content centrality and query relevance.

    This is called on the already-retrieved evidence passages (not the full
    document), so it re-ranks a small set of pre-filtered sentences rather
    than scanning the whole document.

    Scoring formula:
        final = alpha * unigram_salience + (1 - alpha) * query_ngram_overlap

    Args:
        sentences:  list of candidate sentence strings (from passage retrieval)
        doc_tokens: preprocessed token list for the full document (for IDF-like
                    frequency weighting — high-freq doc terms = important terms)
        query_text: raw or lightly preprocessed query string (for n-gram overlap)
        top_n:      number of sentences to return
        ngram_n:    n-gram size for query overlap scoring (3 = trigrams)
        alpha:      blend weight; 0.6 = 60% content, 40% query relevance

    Returns:
        List of (original_index, sentence, score) sorted in original reading
        order (not by score) so the output reads coherently.
    """
    freq         = weighted_freq(doc_tokens)
    query_ngrams = _make_ngrams(query_text, ngram_n)

    scored: List[Tuple[int, str, float]] = []
    for i, s in enumerate(sentences):
        uni   = score_sentence_unigram(s, freq)
        ng    = score_sentence_ngram(s, query_ngrams, ngram_n)
        final = alpha * uni + (1.0 - alpha) * ng
        scored.append((i, s, final))

    # Pick the top_n by score, then restore reading order for coherent display
    best = nlargest(top_n, scored, key=lambda x: x[2])
    best.sort(key=lambda x: x[0])
    return best
