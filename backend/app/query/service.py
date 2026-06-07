"""
Query Service
-------------
Orchestrates the full query pipeline:
    embed → retrieve (semantic + keyword + RRF) → rerank → confidence gate → generate → log

This is the single function called by the FastAPI route handler.
Every step is timed individually so you can see in the logs
exactly where latency is coming from.
"""

import time
import json
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.query.retriever import retrieve
from app.query.reranker import rerank_chunks
from app.query.generator import generate_answer
from app.models.schemas import QueryRequest, QueryResponse, Citation
from app.core.config import settings
from app.core.database import engine


async def process_query(request: QueryRequest, db: AsyncSession) -> QueryResponse:
    """
    Full query pipeline with timing, confidence gate, and query logging.

    Pipeline stages:
    1. Retrieve — semantic + keyword search + RRF fusion
    2. Rerank — BM25 reranking on top candidates
    3. Confidence gate — refuse if top score below threshold
    4. Generate — call GPT-4o with top chunks + chat history
    5. Log — write everything to query_logs table
    """
    t_total_start = time.time()

    # ── INTENT CLASSIFICATION ──────────────────────────────────────────
    # Ask the LLM: does this need document lookup, or is it conversational?
    # This replaces hardcoded keyword lists — the LLM understands intent.
    needs_rag = await _needs_rag(request.question, request.chat_history)
    if not needs_rag:
        reply = await _conversational_reply(request.question, request.chat_history)
        latency_total_ms = int((time.time() - t_total_start) * 1000)
        return QueryResponse(
            answered=True,
            answer_text=reply,
            citations=[],
            confidence_score=1.0,
            top_rrf_score=1.0,
            latency_total_ms=latency_total_ms,
            refusal_reason=None
        )

    # ── STAGE 1: RETRIEVE ──────────────────────────────────────────────
    # Theory: embed the question, search by meaning AND keyword, merge with RRF
    t_retrieval_start = time.time()
    merged_chunks, top_rrf_score, expanded_query = await retrieve(request.question, db, request.chat_history)
    latency_retrieval_ms = int((time.time() - t_retrieval_start) * 1000)

    # ── STAGE 2: RERANK ────────────────────────────────────────────────
    # Theory: BM25 scores top 20 candidates against the search intent
    # Use expanded_query (not the original short question) so the reranker
    # scores against the full intent — "what is gdpr" → "What is the General
    # Data Protection Regulation..." — this selects the right top-5 chunks
    t_rerank_start = time.time()
    top_chunks = rerank_chunks(
        question=expanded_query,
        chunks=merged_chunks[:settings.top_k_chunks],
        top_n=settings.rerank_top_n
    )
    latency_rerank_ms = int((time.time() - t_rerank_start) * 1000)

    # ── STAGE 3: CONFIDENCE GATE ───────────────────────────────────────
    # Theory: if the best chunk isn't relevant enough, refuse to answer
    # This prevents hallucination on out-of-scope questions
    # A question about football asked to a GDPR chatbot → score will be low → refused
    if not merged_chunks or top_rrf_score < settings.confidence_threshold:
        latency_total_ms = int((time.time() - t_total_start) * 1000)

        # Log the refused query
        await _log_query(
            db=db,
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
            latency_llm_ms=0,
            tokens_input=0,
            tokens_output=0,
        )

        return QueryResponse(
            answered=False,
            answer_text=None,
            citations=[],
            confidence_score=0.0,
            top_rrf_score=top_rrf_score,
            latency_total_ms=latency_total_ms,
            refusal_reason=(
                f"No sufficiently relevant content found in the knowledge base. "
                f"Top relevance score: {top_rrf_score:.3f} "
                f"(minimum required: {settings.confidence_threshold})"
            )
        )

    # ── STAGE 4: GENERATE ──────────────────────────────────────────────
    # Theory: pass top 5 chunks + chat history to GPT-4o
    # GPT-4o returns structured JSON: answer + citations + confidence
    t_llm_start = time.time()
    generated = await generate_answer(
        question=request.question,
        chunks=top_chunks,
        chat_history=request.chat_history
    )
    latency_llm_ms = int((time.time() - t_llm_start) * 1000)
    latency_total_ms = int((time.time() - t_total_start) * 1000)

    answered = generated["answer"] is not None

    # ── STAGE 5: LOG ───────────────────────────────────────────────────
    await _log_query(
        db=db,
        query_text=request.question,
        retrieved_chunk_ids=[c["id"] for c in top_chunks],
        retrieval_scores=[c.get("rrf_score", 0) for c in top_chunks],
        top_rrf_score=top_rrf_score,
        answered=answered,
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

    return QueryResponse(
        answered=answered,
        answer_text=generated["answer"],
        citations=generated["citations"],
        confidence_score=generated["confidence"],
        top_rrf_score=top_rrf_score,
        latency_total_ms=latency_total_ms,
        refusal_reason=generated.get("refusal_reason") if not answered else None
    )


async def _log_query(
    db=None,  # kept for API compatibility — logging uses its own engine connection
    query_text: str = "",
    retrieved_chunk_ids: list[str] = [],
    retrieval_scores: list[float] = [],
    top_rrf_score: float = 0.0,
    answered: bool = False,
    answer_text: str | None = None,
    citations: list[Citation] = [],
    confidence_score: float = 0.0,
    latency_total_ms: int = 0,
    latency_retrieval_ms: int = 0,
    latency_rerank_ms: int = 0,
    latency_llm_ms: int = 0,
    tokens_input: int = 0,
    tokens_output: int = 0,
    error_message: str | None = None,
):
    """
    Write a complete query record to the query_logs table.
    Uses its own engine connection (not the request session) so the INSERT + COMMIT
    don't interfere with the session state managed by FastAPI's Depends(get_db).
    """
    citations_json = json.dumps([
        {
            "chunk_id": c.chunk_id,
            "source_file": c.source_file,
            "source_url": c.source_url,
            "page_number": c.page_number,
            "section_title": c.section_title,
            "excerpt": c.excerpt
        }
        for c in citations
    ])

    async with engine.begin() as conn:  # engine.begin() = auto-commit on exit
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
            }
        )


from openai import AsyncOpenAI as _AsyncOpenAI
_openai_client = _AsyncOpenAI(api_key=settings.openai_api_key)

CONVERSATIONAL_SYSTEM = (
    "You are a compliance assistant for DataVault GmbH, specializing in GDPR, "
    "German data protection law (BDSG), and DataVault's internal data protection policies. "
    "For greetings, respond briefly and direct the user to ask a compliance question. "
    "For gibberish, single words, or clearly off-topic input, respond in one sentence: "
    "acknowledge you didn't understand and invite them to ask a compliance question. "
    "Never engage playfully or warmly with nonsense input. Be concise."
)


async def _needs_rag(question: str, chat_history=None) -> bool:
    """
    Ask GPT-4o-mini to classify: does this question need document lookup?
    Returns True if RAG is needed, False if it's conversational.

    Chat history is included so follow-up questions like "find a summary"
    or "explain more" are classified correctly based on the conversation context.
    Without history, "find a summary" looks conversational — with history showing
    a GDPR discussion, it's clearly a COMPLIANCE follow-up.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a router for a GDPR compliance assistant. "
                "Classify the LATEST user message as either COMPLIANCE or CONVERSATIONAL.\n"
                "COMPLIANCE: any question about data protection, privacy, GDPR, BDSG, AI tools at work, "
                "employee data, customer data, data sharing, data breaches, retention periods, consent, "
                "legal basis, company policies, workplace rules, software usage at work, or anything "
                "that could have a compliance or legal implication.\n"
                "Also COMPLIANCE: pronoun follow-ups when the prior topic was compliance "
                "('can i use it', 'is it allowed', 'what about it', 'can we do that', 'is that legal', "
                "'what does it say', 'tell me more', 'explain more', 'give an example').\n"
                "Also COMPLIANCE: questions about any software tool or service in a work context "
                "('what is ChatGPT', 'what is Slack', 'what is Notion') when compliance topics appear "
                "in history — treat as a request to understand the tool's compliance implications.\n"
                "Also COMPLIANCE: any factual question about DataVault GmbH specifically "
                "(prices, salaries, financials, personnel, internal details).\n"
                "CONVERSATIONAL: ONLY pure greetings (hi, hey, hello, hy, hiya, good morning, good afternoon), "
                "thank you, farewells, emotional reactions and exclamations (ok, cool, nice, lol, wtf, wtf is going on, "
                "omg, wow, damn, seriously, really, huh, what the hell, that's crazy, interesting), "
                "or questions about what the assistant can do. "
                "These are ALWAYS CONVERSATIONAL regardless of history — even if compliance topics appeared before.\n"
                "Reply with exactly one word: COMPLIANCE or CONVERSATIONAL."
            )
        }
    ]

    # Include last 4 messages of history so the classifier has context for follow-ups
    if chat_history:
        for msg in chat_history[-4:]:
            if msg.content and msg.content.strip():
                messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": question})

    response = await _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=5,
    )
    result = response.choices[0].message.content.strip().upper()
    return result == "COMPLIANCE"


async def _conversational_reply(question: str, chat_history) -> str:
    """
    Let GPT-4o-mini respond naturally — no RAG, just the LLM's own knowledge.
    """
    messages = [{"role": "system", "content": CONVERSATIONAL_SYSTEM}]

    for msg in chat_history[-4:]:
        if msg.content and msg.content.strip():
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": question})

    response = await _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()
