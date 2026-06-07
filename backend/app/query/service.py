"""
Query Service
-------------
Two-stage pipeline with a hard cosine similarity gate:

  Stage 1 — KB Gate:
      Embed the question, check cosine similarity of top semantic hit.
      If cosine >= settings.cosine_threshold (0.48): go to RAG pipeline.
      If cosine <  settings.cosine_threshold:       go to GPT direct.

  Stage 2a — RAG pipeline (in-KB path):
      Retrieve (semantic + keyword + RRF) → rerank → GPT-4o with strict grounding.
      GPT must cite every claim. If answer=null (KB has no relevant chunk despite
      high cosine — e.g. "DataVault CEO"), fall back to GPT direct automatically.

  Stage 2b — GPT direct (out-of-KB path):
      Pass question + chat history to GPT-4o-mini with no rules.
      No citations, no grounding constraints, just a helpful answer.

Why cosine gate instead of intent classifier?
  The intent classifier was a prompt — fragile, edge cases never end.
  Cosine similarity is deterministic: it measures actual vector distance
  between the question and the closest document in the KB. If nothing in
  the KB is semantically close, the score is low and the gate opens to GPT.
  Tested on 30 questions: 29/30 correct at threshold=0.48.
"""

import time
import json
import uuid
import asyncio
import threading
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from openai import AsyncOpenAI

from app.query.retriever import retrieve, embed_query, semantic_search, rewrite_query
from app.query.reranker import rerank_chunks
from app.query.generator import generate_answer
from app.models.schemas import QueryRequest, QueryResponse, Citation
from app.core.config import settings
from app.core.database import engine

_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

GPT_DIRECT_SYSTEM = (
    "You are a helpful assistant for DataVault GmbH employees. "
    "Answer the user's question naturally and concisely. "
    "If the question is a greeting or social message, respond briefly and warmly. "
    "If the question is about compliance, GDPR, or data protection, let them know "
    "you can answer those questions with citations from the knowledge base — "
    "just ask them to be more specific about the compliance topic."
)


async def process_query(request: QueryRequest, db: AsyncSession, background_tasks=None) -> QueryResponse:
    """
    Full query pipeline: cosine gate → RAG or GPT direct.
    """
    t_total_start = time.time()

    # ── STAGE 1: COSINE GATE ──────────────────────────────────────────────
    # Embed the RAW question (no expansion) and check similarity against KB.
    # Why raw, not expanded? The expander is trained on GDPR context so it
    # rewrites even "tell me a joke" into GDPR language — inflating scores.
    # The raw question embedding is an honest signal: if it's semantically
    # distant from everything in the KB with zero coaching, it doesn't belong.
    t_retrieval_start = time.time()
    gate_vector = await embed_query(request.question)
    gate_results = await semantic_search(gate_vector, top_k=1)

    top_cosine = gate_results[0]["similarity_score"] if gate_results else 0.0

    if top_cosine < settings.cosine_threshold:
        # ── GPT DIRECT (out-of-KB) ────────────────────────────────────────
        reply = await _gpt_direct(request.question, request.chat_history)
        latency_total_ms = int((time.time() - t_total_start) * 1000)
        return QueryResponse(
            answered=True,
            answer_text=reply,
            citations=[],
            confidence_score=1.0,
            top_rrf_score=0.0,
            latency_total_ms=latency_total_ms,
            refusal_reason=None,
        )

    # ── STAGE 2: FULL RETRIEVAL ────────────────────────────────────────────
    # Cosine passed — run full hybrid search (semantic + keyword + RRF)
    merged_chunks, top_rrf_score, expanded_query, _ = await retrieve(
        request.question, db, request.chat_history
    )
    latency_retrieval_ms = int((time.time() - t_retrieval_start) * 1000)

    # ── STAGE 3: RERANK ───────────────────────────────────────────────────
    t_rerank_start = time.time()
    top_chunks = rerank_chunks(
        question=expanded_query,
        chunks=merged_chunks[:settings.top_k_chunks],
        top_n=settings.rerank_top_n,
    )
    latency_rerank_ms = int((time.time() - t_rerank_start) * 1000)

    # ── STAGE 4: GENERATE ─────────────────────────────────────────────────
    t_llm_start = time.time()
    generated = await generate_answer(
        question=request.question,
        chunks=top_chunks,
        chat_history=request.chat_history,
    )
    latency_llm_ms = int((time.time() - t_llm_start) * 1000)
    latency_total_ms = int((time.time() - t_total_start) * 1000)

    answered = generated["answer"] is not None

    if not answered:
        # ── GPT FALLBACK ──────────────────────────────────────────────────
        # Cosine was high (DataVault-related question) but KB has no relevant chunk.
        # Example: "Who is the DataVault CEO?" — cosine=0.62 but no chunk about personnel.
        # Fall back to GPT without citations. Log with answered=False.
        await _log_query(
            query_text=request.question,
            retrieved_chunk_ids=[c["id"] for c in top_chunks],
            retrieval_scores=[c.get("rrf_score", 0) for c in top_chunks],
            top_rrf_score=top_rrf_score,
            answered=False,
            answer_text=None,
            citations=[],
            confidence_score=0.0,
            latency_total_ms=latency_total_ms,
            latency_retrieval_ms=latency_retrieval_ms,
            latency_rerank_ms=latency_rerank_ms,
            latency_llm_ms=latency_llm_ms,
            tokens_input=generated["tokens_input"],
            tokens_output=generated["tokens_output"],
        )
        fallback_reply = await _gpt_direct(request.question, request.chat_history)
        return QueryResponse(
            answered=True,
            answer_text=fallback_reply,
            citations=[],
            confidence_score=1.0,
            top_rrf_score=top_rrf_score,
            latency_total_ms=latency_total_ms,
            refusal_reason=None,
        )

    # ── STAGE 5: LOG ──────────────────────────────────────────────────────
    await _log_query(
        query_text=request.question,
        retrieved_chunk_ids=[c["id"] for c in top_chunks],
        retrieval_scores=[c.get("rrf_score", 0) for c in top_chunks],
        top_rrf_score=top_rrf_score,
        answered=True,
        answer_text=generated["answer"],
        citations=generated["citations"],
        confidence_score=generated["confidence"],
        latency_total_ms=latency_total_ms,
        latency_retrieval_ms=latency_retrieval_ms,
        latency_rerank_ms=latency_rerank_ms,
        latency_llm_ms=latency_llm_ms,
        tokens_input=generated["tokens_input"],
        tokens_output=generated["tokens_output"],
    )

    # ── STAGE 6: RAGAS EVAL (background) ──────────────────────────────────────
    # Fire RAGAS with full chunk texts — NOT citation excerpts.
    # Citation excerpts are 1-2 sentences; RAGAS needs the full chunk to verify
    # every claim in the answer. Short contexts produce faithfulness=0 or NaN.
    if background_tasks and generated["answer"]:
        from app.query.evaluator import evaluate_and_store
        from app.logs.service import get_latest_log_id
        full_contexts = [c["chunk_text"] for c in top_chunks if c.get("chunk_text")]
        if full_contexts:
            log_id = await get_latest_log_id(db=db, question=request.question)
            if log_id:
                background_tasks.add_task(
                    evaluate_and_store,
                    query_log_id=log_id,
                    question=request.question,
                    answer=generated["answer"],
                    contexts=full_contexts,
                )

    return QueryResponse(
        answered=True,
        answer_text=generated["answer"],
        citations=generated["citations"],
        confidence_score=generated["confidence"],
        top_rrf_score=top_rrf_score,
        latency_total_ms=latency_total_ms,
        refusal_reason=None,
    )


async def _gpt_direct(question: str, chat_history) -> str:
    """
    GPT-4o-mini with no grounding rules. Used for out-of-KB questions
    and as a fallback when RAG finds no relevant answer.
    No citations, no constraints — just answer naturally.
    """
    messages = [{"role": "system", "content": GPT_DIRECT_SYSTEM}]
    if chat_history:
        for msg in chat_history[-4:]:
            if msg.content and msg.content.strip():
                messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": question})

    response = await _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


async def _log_query(
    query_text: str = "",
    retrieved_chunk_ids: list = [],
    retrieval_scores: list = [],
    top_rrf_score: float = 0.0,
    answered: bool = False,
    answer_text: str | None = None,
    citations: list = [],
    confidence_score: float = 0.0,
    latency_total_ms: int = 0,
    latency_retrieval_ms: int = 0,
    latency_rerank_ms: int = 0,
    latency_llm_ms: int = 0,
    tokens_input: int = 0,
    tokens_output: int = 0,
    error_message: str | None = None,
    db=None,  # unused — kept for call-site compatibility
):
    citations_json = json.dumps([
        {
            "chunk_id": c.chunk_id,
            "source_file": c.source_file,
            "source_url": c.source_url,
            "page_number": c.page_number,
            "section_title": c.section_title,
            "excerpt": c.excerpt,
        }
        for c in citations
    ])

    async with engine.begin() as conn:
        await conn.execute(
            text("""
                INSERT INTO query_logs (
                    id, query_text, retrieved_chunk_ids, retrieval_scores,
                    top_rrf_score, answered, answer_text, citations,
                    confidence_score, latency_total_ms, latency_retrieval_ms,
                    latency_rerank_ms, latency_llm_ms, llm_model,
                    tokens_input, tokens_output, error_message
                ) VALUES (
                    :id, :query_text, :retrieved_chunk_ids, :retrieval_scores,
                    :top_rrf_score, :answered, :answer_text, CAST(:citations AS jsonb),
                    :confidence_score, :latency_total_ms, :latency_retrieval_ms,
                    :latency_rerank_ms, :latency_llm_ms, :llm_model,
                    :tokens_input, :tokens_output, :error_message
                )
            """),
            {
                "id": str(uuid.uuid4()),
                "query_text": query_text,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieval_scores": retrieval_scores,
                "top_rrf_score": top_rrf_score,
                "answered": answered,
                "answer_text": answer_text,
                "citations": citations_json,
                "confidence_score": confidence_score,
                "latency_total_ms": latency_total_ms,
                "latency_retrieval_ms": latency_retrieval_ms,
                "latency_rerank_ms": latency_rerank_ms,
                "latency_llm_ms": latency_llm_ms,
                "llm_model": settings.llm_model,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "error_message": error_message,
            },
        )
