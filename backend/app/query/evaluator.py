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

Why lazy imports?
   RAGAS calls nest_asyncio.apply() at module load time, which crashes inside
   uvicorn's event loop with "Can't patch loop of type". Importing inside the
   function body delays that call until the background thread runs — outside
   the main event loop — so it applies cleanly.
"""

import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text
from app.core.database import engine
from app.core.config import settings

logger = logging.getLogger(__name__)

# Single thread pool for all RAGAS evaluations — RAGAS is sync and CPU-light
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ragas")


def _run_ragas_sync(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Synchronous RAGAS evaluation — runs in a thread pool worker.

    All RAGAS imports are inside this function so nest_asyncio.apply() fires
    in the worker thread, not in uvicorn's main event loop.

    Beginner example:
        question  = "What is the deadline after a data breach?"
        answer    = "72 hours"
        contexts  = ["Article 33: notify within 72 hours of becoming aware..."]
        → faithfulness=1.0 (answer claim is in context)
        → answer_relevancy=0.95 (answer directly addresses the deadline question)
    """
    # Lazy imports — must stay inside this function
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
        temperature=0.0,
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
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
