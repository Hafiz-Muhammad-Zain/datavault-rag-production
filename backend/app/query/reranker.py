"""
Reranker
--------
Takes the top N chunks from RRF and scores them more precisely
using BM25 against the full question text.

Why rerank after RRF?
- RRF uses rank position, not semantic similarity
- A chunk ranked #3 by both retrievers beats one ranked #1 by only one
- But RRF doesn't look at HOW relevant the text is to the exact question
- BM25 reranking scores each candidate against the full question
  and picks the most textually relevant top_n chunks

Why only rerank top 20 (not all chunks)?
- Running BM25 on 630 chunks every query = slow
- RRF already narrowed to the best candidates
- Reranking 20 candidates = fast (pure Python, no API call)
- This is the "candidate set reranking" pattern from production RAG

rank_bm25 library:
- Pure Python BM25 implementation
- No extra model to host — runs in-process
- Input: list of tokenized documents + a query
- Output: relevance scores for each document
"""

from rank_bm25 import BM25Okapi


def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_n: int
) -> list[dict]:
    """
    Rerank a list of chunks using BM25 against the question.

    Steps:
    1. Tokenize each chunk (split by whitespace — simple but effective)
    2. Build a BM25 index over just these candidates
    3. Score each candidate against the tokenized question
    4. Sort by score, return top_n

    BM25Okapi is the standard BM25 variant — same formula used by Elasticsearch.
    Okapi = the original implementation from Okapi BM25 paper (Robertson, 1994).

    Beginner example: like a teacher grading 20 pre-selected essay answers
    specifically for how well they answer THIS question,
    rather than just "are these good essays generally?"
    """
    if not chunks:
        return []

    # Tokenize each chunk — lowercase and split by whitespace
    # In production you'd use a proper tokenizer (NLTK, spaCy)
    # but for this use case simple splitting is sufficient
    tokenized_chunks = [chunk["chunk_text"].lower().split() for chunk in chunks]
    tokenized_question = question.lower().split()

    # Build BM25 index over these candidates only
    bm25 = BM25Okapi(tokenized_chunks)

    # Score each candidate against the question
    scores = bm25.get_scores(tokenized_question)

    # Attach score to each chunk and sort
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = float(score)
        scored_chunks.append(chunk_copy)

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_chunks[:top_n]
