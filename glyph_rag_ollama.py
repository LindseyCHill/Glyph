# glyph_rag_ollama.py
# Lindsey Hill - AIT626 Term Project - Glyph
#
# Ollama integration for true RAG-style generative answers.
#
# This module follows the same pattern established in Lab 4:
#   1. retrieve()      — TF-IDF + cosine similarity to rank passages
#   2. build_context() — format top passages with source metadata
#   3. build_prompt()  — wrap context + question in a grounded prompt
#   4. run_ollama()    — call Ollama via subprocess (same as lab)
#   5. rag_answer()    — orchestrate the full pipeline for one query
#
# Why subprocess instead of REST API?
#   The lab used subprocess("ollama run ...") which is more reliable
#   in a local dev environment — it uses the same CLI you interact with
#   directly, so model loading, path resolution, etc. all work the same way.
#
# Why TF-IDF + cosine here?
#   Lab 4 used TF-IDF vectors + cosine similarity for chunk retrieval.
#   Using the same method here means the RAG retrieval step is consistent
#   with what was taught in class and can be directly compared.

from __future__ import annotations

import subprocess
import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional


# Maximum characters of context to pass to the model.
# Keeps prompts from getting too long for smaller models.
MAX_CONTEXT_CHARS = 3000

# Timeout for subprocess call (seconds).
# Generation can take a while on CPU; 120s is generous but safe.
GENERATE_TIMEOUT = 120


# =============================================================================
# OLLAMA CLI INTERFACE
# =============================================================================

def run_ollama_cmd(args: List[str], timeout: int = 10) -> Tuple[bool, str]:
    """
    Run an ollama CLI command and return (success, output).

    Uses subprocess so behavior matches the lab exactly.
    Captures both stdout and stderr; stderr is used for error messages.

    Args:
        args:    list of command parts, e.g. ["ollama", "list"]
        timeout: seconds before giving up

    Returns:
        (True, stdout) on success
        (False, error_message) on failure
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            err = result.stderr.strip() or "Command failed with no error message."
            return False, err
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, f"Ollama timed out after {timeout}s."
    except FileNotFoundError:
        return False, "Ollama not found. Is it installed and on your PATH?"
    except Exception as e:
        return False, f"Unexpected error running Ollama: {e}"


def ping_ollama() -> bool:
    """
    Check whether Ollama is installed and responding.
    Uses 'ollama --version' as a lightweight probe.
    """
    ok, _ = run_ollama_cmd(["ollama", "--version"], timeout=5)
    return ok


def list_models() -> Tuple[bool, List[str], str]:
    """
    Get the list of locally available models via 'ollama list'.

    Parses the table output that looks like:
        NAME              ID        SIZE    MODIFIED
        llama3:latest     abc123    4.7GB   2 hours ago

    Returns:
        (success, model_name_list, error_message)
    """
    ok, output = run_ollama_cmd(["ollama", "list"], timeout=8)

    if not ok:
        return False, [], output  # output is the error message here

    lines  = output.strip().splitlines()
    models = []

    for line in lines[1:]:   # skip the header row
        line = line.strip()
        if not line:
            continue
        # First column is the model name
        name = line.split()[0]
        if name:
            models.append(name)

    if not models:
        return True, [], "Ollama is running but no models are pulled yet.\nRun: ollama pull llama3"

    return True, models, ""


# =============================================================================
# TF-IDF RETRIEVAL  (matches Lab 4 build_tfidf_index + retrieve pattern)
# =============================================================================

def _tokenize(text: str) -> List[str]:
    """Lowercase, keep alphanumeric tokens only. Matches lab tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_tfidf_index(passages: List[Tuple[int, str, float]]) -> Dict:
    """
    Build a TF-IDF index from a list of passage tuples.

    Input format matches top_passages_for_doc() output:
        [(sentence_index, sentence_text, keyword_score), ...]

    The keyword_score from passage pre-ranking is ignored here —
    we rerank using TF-IDF cosine similarity for the RAG step,
    which is a more principled retrieval method than raw keyword counts.

    Returns a dict with:
        chunks  — original passage list
        idf     — IDF weights per token
        vecs    — TF-IDF vector (sparse dict) per passage
        norms   — L2 norm per vector (pre-computed for cosine efficiency)
    """
    tokenized = [_tokenize(text) for _idx, text, _sc in passages]
    N = len(tokenized)

    # Document frequency: how many passages contain each token
    df: Counter = Counter()
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1

    # Smoothed IDF: log((N+1)/(df+1)) + 1  (avoids zero division, matches lab)
    idf = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

    # TF-IDF vectors (sparse dicts) + pre-computed norms
    vecs:  List[Dict[str, float]] = []
    norms: List[float]            = []

    for toks in tokenized:
        if not toks:
            vecs.append({})
            norms.append(1.0)
            continue
        tf  = Counter(toks)
        vec = {t: (tf[t] / len(toks)) * idf.get(t, 0.0) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vecs.append(vec)
        norms.append(norm)

    return {"chunks": passages, "idf": idf, "vecs": vecs, "norms": norms}


def _tfidf_query_vector(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    """Build a TF-IDF vector for the query using the index's IDF weights."""
    toks = _tokenize(query)
    if not toks:
        return {}
    tf = Counter(toks)
    return {t: (tf[t] / len(toks)) * idf.get(t, 0.0) for t in tf}


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Cosine similarity between two sparse TF-IDF vectors.
    Iterates over the smaller dict for efficiency (matches lab).
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(t, 0.0) for t, v in a.items())
    na  = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb  = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


def retrieve_tfidf(
    query:    str,
    passages: List[Tuple[int, str, float]],
    top_k:    int = 5,
) -> List[Tuple[float, int, str]]:
    """
    Rank passages by TF-IDF cosine similarity to the query.

    This is the Lab 4 retrieve() pattern applied to Glyph's passage list.
    Returns passages re-ranked by cosine score, not the original keyword score.

    Args:
        query:    raw query string
        passages: list of (sentence_index, text, keyword_score)
        top_k:    number of top passages to return

    Returns:
        List of (cosine_score, sentence_index, text) sorted descending.
    """
    if not passages:
        return []

    index   = build_tfidf_index(passages)
    q_vec   = _tfidf_query_vector(query, index["idf"])
    scored  = []

    for vec, (idx, text, _kw_score) in zip(index["vecs"], passages):
        score = _cosine_sparse(q_vec, vec)
        scored.append((score, idx, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# =============================================================================
# CONTEXT + PROMPT BUILDING  (matches Lab 4 build_context / build_prompt)
# =============================================================================

def build_context(
    top_passages: List[Tuple[float, int, str]],
    doc_title:    str,
    max_chars:    int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Format retrieved passages into a context block for the prompt.

    Follows the Lab 4 format:
        [Source: <title> | passage=<idx> | score=<score>]
        <passage text>

    Passages are included in order until max_chars is reached,
    so the context never exceeds what the model can handle well.

    Args:
        top_passages: list of (cosine_score, sentence_index, text)
        doc_title:    title of the source document
        max_chars:    hard character limit for total context
    """
    parts = []
    total = 0

    for score, idx, text in top_passages:
        block = (
            f"[Source: {doc_title} | passage={idx} | score={score:.3f}]\n"
            f"{text.strip()}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """
    Build a grounded RAG prompt.

    Matches the Lab 4 prompt template exactly — explicit grounding instruction
    and an "I don't know" fallback so the model doesn't hallucinate.
    """
    return (
        f"You are a helpful assistant. Answer the question using ONLY the context below.\n"
        f"If the answer is not in the context, say: "
        f"\"I don't know based on the provided context.\"\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n"
    )


# =============================================================================
# FULL RAG PIPELINE
# =============================================================================

def rag_answer(
    query:      str,
    passages:   List[Tuple[int, str, float]],
    doc_title:  str,
    model:      str,
    top_k:      int = 5,
) -> Tuple[bool, str]:
    """
    Full RAG pipeline for one document:
      1. Re-rank passages by TF-IDF cosine similarity
      2. Build context block from top passages
      3. Build grounded prompt
      4. Run Ollama via subprocess
      5. Return generated answer

    This matches the Lab 4 rag_answer() function pattern.

    Args:
        query:     raw query string from the user
        passages:  pre-retrieved passages from keyword scoring
                   (these are re-ranked here by TF-IDF cosine)
        doc_title: source document title (shown in context block)
        model:     Ollama model name string (e.g. "llama3:latest")
        top_k:     number of passages to include in context

    Returns:
        (success, answer_text)
        On failure, answer_text is a human-readable error message.
    """
    if not passages:
        return False, "(No passages retrieved — cannot generate an answer.)"

    # Step 1: Re-rank by TF-IDF cosine similarity
    top = retrieve_tfidf(query, passages, top_k=top_k)

    # Step 2: Build context block
    context = build_context(top, doc_title)

    # Step 3: Build prompt
    prompt = build_prompt(query, context)

    # Step 4: Run Ollama (subprocess, matching Lab 4)
    ok, output = run_ollama_cmd(
        ["ollama", "run", model, prompt],
        timeout=GENERATE_TIMEOUT,
    )

    if not ok:
        return False, f"(Ollama error: {output})"

    answer = output.strip()
    if not answer:
        return False, "(Ollama returned an empty response.)"

    return True, answer
