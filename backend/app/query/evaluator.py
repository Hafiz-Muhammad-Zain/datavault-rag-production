"""
RAGAS Evaluator
---------------
Runs reference-free RAGAS evaluation on answered queries.
Fires as a FastAPI background task — does not block the response to the user.

Two metrics measured:
1. Faithfulness — does every claim in the answer come from the retrieved chunks?
   How: RAGAS breaks the answer into atomic claims, checks each against context.
   Score 1.0 = fully grounded. Score 0.0 = answer contains hallucinated claims.

2. Answer Relevancy — does the answer actually address the question?
   How: embeds the answer, generates synthetic questions from it, measures
   cosine similarity between generated questions and the original.
   Score 1.0 = answer directly addresses the question.
   Score 0.0 = answer is on-topic but doesn't answer what was asked.

Why reference-free?
   Standard RAGAS needs ground-truth answers to measure recall.
   Reference-free mode skips that — it only needs (question, answer, context).
   This is the only practical mode for live production where you don't have labels.

Why background task?
   RAGAS makes multiple LLM calls internally (~3-5 per query).
   Running it inline would double response latency for the user.
   Instead: respond immediately, evaluate async, write score to DB.

Why ProcessPoolExecutor?
   RAGAS calls nest_asyncio.apply() at import time. nest_asyncio cannot patch
   uvloop (used by uvicorn in production) — it throws "Can't patch loop of type
   uvloop.Loop". A ProcessPoolExecutor worker is a fresh Python process with no
   existing event loop, so RAGAS can create and patch its own loop freely.
   ThreadPoolExecutor shares the parent process's loop state — that's why it fails.
"""

import uuid
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import text
from app.core.database import engine
from app.core.config import settings

logger = logging.getLogger(__name__)

# Process pool — each worker is a fresh Python process, no uvloop conflict
_executor = ProcessPoolExecutor(max_workers=1)


def _run_ragas_sync(question: str, answer: str, contexts: list[str], openai_api_key: str, embedding_model: str) -> dict:
    """
    Synchronous RAGAS evaluation — runs in a subprocess worker.

    Receives openai_api_key and embedding_model as plain arguments because
    subprocess workers cannot access the parent process's settings object.

    Beginner example:
        question  = "What is the deadline after a data breach?"
        answer    = "72 hours"
        contexts  = ["Article 33: notify within 72 hours of becoming aware..."]
        → faithfulness=1.0 (answer claim is in context)
        → answer_relevancy=0.95 (answer directly addresses the deadline question)
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.0,
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=embedding_model,
        api_key=openai_api_key,
    ))

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    scores = result.to_pandas()
    return {
        "faithfulness": float(scores["faithfulness"].iloc[0]) if "faithfulness" in scores.columns else None,
        "answer_relevancy": float(scores["answer_relevancy"].iloc[0]) if "answer_relevancy" in scores.columns else None,
    }


async def evaluate_and_store(
    query_log_id: str,
    question: str,
    answer: str,
    contexts: list[str],
) -> None:
    """
    Run RAGAS faithfulness + answer_relevancy on one answered query.
    Writes scores to eval_scores table. Called as a background task.
    """
    if not answer or not contexts:
        return

    try:
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            _executor,
            _run_ragas_sync,
            question,
            answer,
            contexts,
            settings.openai_api_key,
            settings.embedding_model,
        )

        async with engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO eval_scores (id, query_log_id, faithfulness, answer_relevancy)
                    VALUES (:id, :query_log_id, :faithfulness, :answer_relevancy)
                """),
                {
                    "id": str(uuid.uuid4()),
                    "query_log_id": query_log_id,
                    "faithfulness": scores["faithfulness"],
                    "answer_relevancy": scores["answer_relevancy"],
                }
            )

        logger.info(
            f"RAGAS eval complete — log_id={query_log_id} "
            f"faithfulness={scores['faithfulness']} relevancy={scores['answer_relevancy']}"
        )

    except Exception as e:
        logger.warning(f"RAGAS eval failed for query_log_id={query_log_id}: {e}")
