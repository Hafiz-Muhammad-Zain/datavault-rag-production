"""
Retriever
---------
Runs two searches in parallel against PostgreSQL:
1. Semantic search via pgvector (cosine similarity on embeddings)
2. Keyword search via ts_vector (BM25-style full-text search)

Then merges results using Reciprocal Rank Fusion (RRF).

Why run both in parallel?
- asyncio.gather() runs both queries simultaneously
- Total time = max(semantic_time, keyword_time) instead of sum
- Typically saves 30-50ms per query

Why use engine.connect() instead of passing an AsyncSession?
- SQLAlchemy AsyncSession is NOT safe for concurrent use on the same connection
- asyncio.gather() runs both searches at the same time on the same session → crash
- engine.connect() gives each search its own connection from the pool → safe
"""

import asyncio
from openai import AsyncOpenAI
from sqlalchemy import text
from app.core.config import settings
from app.core.database import engine  # direct engine access — each search gets its own connection

client = AsyncOpenAI(api_key=settings.openai_api_key)

RRF_K = 60  # RRF constant — prevents top-ranked docs from dominating


async def rewrite_query(question: str, chat_history=None) -> str:
    """
    Expand a short or ambiguous question into a full natural language query
    before embedding. This improves semantic search significantly.

    Why this matters:
    - "what is gdpr" has a tiny embedding — almost no signal
    - "What is the General Data Protection Regulation (GDPR), its purpose and scope?" has rich signal
    - The vector for the expanded query lands much closer to GDPR document chunks

    The keyword (BM25) search also benefits because the expanded query contains
    more terms that appear verbatim in the documents.

    Beginner example: asking a librarian "book on flying" vs
    "book on aeronautical engineering, how aircraft generate lift" — same intent,
    dramatically better results.

    Cost: one tiny GPT-4o-mini call (~50 tokens) added per query.
    """
    if len(question.split()) >= 8 and not any(
        p in question.lower() for p in ["it", "this", "that", "they", "them", "its"]
    ):
        # Long enough and no pronouns — no need to expand
        return question

    messages = [
        {
            "role": "system",
            "content": (
                "You are a query expansion assistant for a GDPR and data protection compliance system. "
                "Rewrite the user's short or pronoun-heavy question into a detailed natural language search query "
                "that will find the right legal text in GDPR documents and company data protection policies. "
                "If the question contains pronouns (it, this, that, they), resolve them using the conversation history. "
                "Expand acronyms (GDPR = General Data Protection Regulation, BDSG = Bundesdatenschutzgesetz). "
                "Return ONLY the expanded query — no explanation, no quotes."
            )
        }
    ]

    # Include last 4 history messages so pronouns can be resolved
    if chat_history:
        for msg in chat_history[-4:]:
            if msg.content and msg.content.strip():
                messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": question})

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=120,
    )
    expanded = response.choices[0].message.content.strip()
    return expanded if expanded else question


async def embed_query(question: str) -> list[float]:
    """
    Convert the user's question into a 1536-dimensional vector.
    Same model used during ingestion — critical that they match.
    If ingestion used model A and query uses model B, vectors are incompatible.

    Beginner example: like translating a question and documents
    into the same language so they can be compared.
    """
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=question,
    )
    return response.data[0].embedding


async def semantic_search(
    query_vector: list[float],
    top_k: int
) -> list[dict]:
    """
    Vector similarity search using pgvector.
    Finds chunks whose meaning is closest to the question.

    The <=> operator in pgvector = cosine distance (1 - cosine_similarity)
    Lower distance = more similar. We convert to similarity: 1 - distance.

    SQL explanation:
        ORDER BY embedding <=> :vector   → sort by cosine distance ascending
        LIMIT :top_k                     → take only the top N closest

    Uses engine.connect() so it gets its own connection from the pool —
    safe to run concurrently with keyword_search via asyncio.gather().
    """
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT
                    id::text,
                    source_file,
                    source_url,
                    page_number,
                    section_title,
                    chunk_text,
                    1 - (embedding <=> CAST(:vector AS vector)) AS similarity_score
                FROM document_chunks
                ORDER BY embedding <=> CAST(:vector AS vector)
                LIMIT :top_k
            """),
            {
                "vector": str(query_vector),
                "top_k": top_k
            }
        )
        rows = result.mappings().all()
    return [
        {
            "id": row["id"],
            "source_file": row["source_file"],
            "source_url": row["source_url"],
            "page_number": row["page_number"],
            "section_title": row["section_title"],
            "chunk_text": row["chunk_text"],
            "similarity_score": float(row["similarity_score"]),
            "retriever": "semantic"
        }
        for row in rows
    ]


async def keyword_search(
    question: str,
    top_k: int
) -> list[dict]:
    """
    Full-text keyword search using PostgreSQL ts_vector (BM25-style).
    Finds chunks that contain the exact keywords from the question.

    plainto_tsquery converts natural language to a query:
        "What is Article 33?" → 'article' & '33'
    ts_rank_cd scores based on keyword frequency and position.

    When semantic search misses: "What is error code §26 BDSG?"
    BM25 finds "§26" exactly. Vector search might return "employee data law"
    which is semantically related but not the exact reference.

    Uses engine.connect() so it gets its own connection — safe to run
    concurrently with semantic_search via asyncio.gather().
    """
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT
                    id::text,
                    source_file,
                    source_url,
                    page_number,
                    section_title,
                    chunk_text,
                    ts_rank_cd(chunk_text_tsv, plainto_tsquery('english', :question)) AS bm25_score
                FROM document_chunks
                WHERE chunk_text_tsv @@ plainto_tsquery('english', :question)
                ORDER BY bm25_score DESC
                LIMIT :top_k
            """),
            {
                "question": question,
                "top_k": top_k
            }
        )
        rows = result.mappings().all()
    return [
        {
            "id": row["id"],
            "source_file": row["source_file"],
            "source_url": row["source_url"],
            "page_number": row["page_number"],
            "section_title": row["section_title"],
            "chunk_text": row["chunk_text"],
            "bm25_score": float(row["bm25_score"]),
            "retriever": "keyword"
        }
        for row in rows
    ]


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
) -> list[dict]:
    """
    Merge two ranked lists into one using Reciprocal Rank Fusion.

    Formula for each document:
        RRF_score = sum of 1 / (K + rank) across all retrievers

    Where K=60 (prevents position 1 from having infinite advantage).

    Why RRF instead of score averaging?
    - Semantic scores (cosine similarity) and keyword scores (ts_rank) live
      in completely different numerical spaces
    - You cannot add 0.87 cosine similarity + 0.003 ts_rank meaningfully
    - RRF ignores the scores entirely — only uses rank positions
    - A document ranked #1 by both retrievers wins over one ranked #1 by only one

    Example with K=60:
        Doc A: rank 1 semantic, rank 3 keyword
            score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        Doc B: rank 2 semantic, rank 1 keyword
            score = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        Doc B wins — ranked well in both retrievers.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Score from semantic retriever
    for rank, chunk in enumerate(semantic_results, start=1):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
        chunk_data[chunk_id] = chunk

    # Score from keyword retriever (additive — same doc gets both contributions)
    for rank, chunk in enumerate(keyword_results, start=1):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = chunk

    # Sort by RRF score descending — highest combined rank wins
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        chunk = chunk_data[chunk_id].copy()
        chunk["rrf_score"] = scores[chunk_id]
        merged.append(chunk)

    return merged


async def retrieve(
    question: str,
    db=None,  # kept for API compatibility but no longer used — each search opens its own connection
    chat_history=None,
) -> tuple[list[dict], float, str]:
    """
    Full retrieval pipeline:
    1. Embed the question
    2. Run semantic + keyword search in parallel (each with its own DB connection)
    3. Merge with RRF
    4. Return merged list + top RRF score

    Returns: (merged_chunks, top_rrf_score)
    top_rrf_score is used by the confidence gate in the query service.
    """
    # Expand short/ambiguous questions before searching — pass history for pronoun resolution
    expanded = await rewrite_query(question, chat_history)

    # Embed the expanded query for semantic search
    query_vector = await embed_query(expanded)

    # Run both searches simultaneously — each opens its own connection from the pool
    keyword_query = question if question == expanded else f"{question} {expanded}"
    semantic_results, keyword_results = await asyncio.gather(
        semantic_search(query_vector, settings.top_k_chunks),
        keyword_search(keyword_query, settings.top_k_chunks)
    )

    # Merge with RRF
    merged = reciprocal_rank_fusion(semantic_results, keyword_results)

    top_score = merged[0]["rrf_score"] if merged else 0.0

    # Return merged chunks, top RRF score, expanded query, AND top cosine similarity
    # top_cosine is used as the KB gate — it measures actual semantic distance
    # RRF score is kept for logging but is rank-based (not a good gate)
    top_cosine = semantic_results[0]["similarity_score"] if semantic_results else 0.0
    return merged, top_score, expanded, top_cosine
