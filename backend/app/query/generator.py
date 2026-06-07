"""
Answer Generator
----------------
Takes the top reranked chunks and calls GPT-4o to generate
a grounded answer with structured citations.

Key design decisions:
1. Strict grounding prompt — LLM is instructed to ONLY answer from context
2. Structured JSON output — forces citation format, no free-form hallucination
3. Conversation history — last N messages passed so follow-ups work in context
4. System prompt enforces refusal — if answer not in context, say so explicitly

Why JSON output?
- We parse the LLM response to extract: answer, citations, confidence
- Free-form text responses are unparseable — we'd have to guess where citations are
- JSON forces the LLM to be explicit about what it knows and where it came from
"""

import json
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.schemas import ChatMessage, Citation

client = AsyncOpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are a compliance assistant for DataVault GmbH, a German SaaS company.
You answer questions about GDPR, the German BDSG, and DataVault's internal data protection policies.

STRICT RULES:
1. Answer ONLY from the context provided below. Do not use any outside knowledge.
2. Every sentence in your answer must be supported by at least one chunk in the context.
3. If a claim cannot be directly supported by a chunk, do not write it.
4. If the context contains PARTIAL information, answer what you can from the chunks — do not refuse just because the answer is incomplete.
5. Only set "answer" to null if the context has NO relevant information at all for the question.
6. Do NOT give general advice not sourced from the chunks. But synthesizing a clear explanation from chunk content is allowed — cite the chunks you used.
7. If "answer" is not null, "citations" MUST be non-empty — you must cite every chunk you used. An answer without citations is a policy violation.
8. Return ONLY valid JSON matching the exact schema below. No markdown, no explanation outside JSON.

RESPONSE SCHEMA:
{
  "answer": "Your answer here, citing only what the chunks say — or null if not found in context",
  "refusal_reason": "Explanation if answer is null, otherwise null",
  "confidence": 0.95,
  "citations": [
    {
      "chunk_id": "the chunk_id from context",
      "source_file": "filename.pdf",
      "source_url": "full path",
      "page_number": 33,
      "section_title": "Article 33 — Notification",
      "excerpt": "exact quote from the chunk that supports this citation"
    }
  ]
}

confidence is a float between 0 and 1 reflecting how completely the context answers the question.
citations must list every chunk you used. Do not cite chunks you did not use.
For the excerpt field: write a clean, readable version of the relevant sentence from the chunk. Fix any obvious PDF extraction artifacts (missing spaces like "appliestothe" → "applies to the", "whollyor" → "wholly or"). The excerpt must convey the meaning accurately."""


def build_context_block(chunks: list[dict]) -> str:
    """
    Format the top chunks into a readable context block for the prompt.
    Each chunk is clearly labelled with its ID, source, and page number
    so the LLM can reference them accurately in citations.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[CHUNK {i}]\n"
            f"chunk_id: {chunk['id']}\n"
            f"source_file: {chunk['source_file']}\n"
            f"source_url: {chunk['source_url']}\n"
            f"page_number: {chunk.get('page_number', 'N/A')}\n"
            f"section_title: {chunk.get('section_title', 'N/A')}\n"
            f"text:\n{chunk['chunk_text']}\n"
        )
    return "\n---\n".join(context_parts)


def build_messages(
    question: str,
    chunks: list[dict],
    chat_history: list[ChatMessage]
) -> list[dict]:
    """
    Build the full message list to send to GPT-4o.

    Structure:
    1. System prompt (grounding rules + JSON schema)
    2. Past conversation messages (so follow-up questions work)
    3. Context block (the retrieved chunks)
    4. Current user question

    Why put context in the user message (not system)?
    - GPT-4o attends better to context placed close to the question
    - System prompt is for instructions, user turn is for data
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Only include messages with actual content — refused answers have empty/null
    # content and confuse the LLM when included as history
    clean_history = [m for m in chat_history[-6:] if m.content and m.content.strip()]
    for msg in clean_history:
        messages.append({"role": msg.role, "content": msg.content})

    # Add context + question as the current user turn
    context_block = build_context_block(chunks)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"
    })

    return messages


async def generate_answer(
    question: str,
    chunks: list[dict],
    chat_history: list[ChatMessage]
) -> dict:
    """
    Call GPT-4o and parse the structured JSON response.

    Returns a dict with:
        answer: str | None
        refusal_reason: str | None
        confidence: float
        citations: list of Citation objects
        raw_response: the full LLM response text (for debugging)
    """
    messages = build_messages(question, chunks, chat_history)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.0,        # zero temperature = deterministic, no creativity
        response_format={"type": "json_object"},  # force JSON output
        max_tokens=1500,
    )

    raw = response.choices[0].message.content
    tokens_input = response.usage.prompt_tokens
    tokens_output = response.usage.completion_tokens

    # Parse the JSON response
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "answer": None,
            "refusal_reason": "Failed to parse LLM response as JSON",
            "confidence": 0.0,
            "citations": [],
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "raw_response": raw
        }

    # Build Citation objects from the parsed citations list
    citations = []
    for c in parsed.get("citations", []):
        try:
            citations.append(Citation(
                chunk_id=c.get("chunk_id", ""),
                source_file=c.get("source_file", ""),
                source_url=c.get("source_url", ""),
                page_number=c.get("page_number"),
                section_title=c.get("section_title"),
                excerpt=c.get("excerpt", "")
            ))
        except Exception:
            continue

    return {
        "answer": parsed.get("answer"),
        "refusal_reason": parsed.get("refusal_reason"),
        "confidence": float(parsed.get("confidence", 0.0)),
        "citations": citations,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "raw_response": raw
    }
