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

Why new event loop in thread?
   RAGAS calls nest_asyncio.apply() at import time. nest_asyncio cannot patch
   uvloop (used by uvicorn in production) — it throws "Can't patch loop of type
   uvloop.Loop". The fix: run RAGAS in a plain thread but give it a FRESH asyncio
   event loop (not uvloop) via asyncio.new_event_loop(). This loop has no existing
   type so nest_asyncio patches it cleanly.
"""

import uuid
import logging
import asyncio
import threading
from sqlalchemy import text
from app.core.database import engine
from app.core.config import settings

logger = logging.getLogger(__name__)


def _run_ragas_in_thread(question: str, answer: str, contexts: list[str], openai_api_key: str, embedding_model: str) -> dict:
    """
    Run RAGAS in a thread with a fresh asyncio event loop.

    Fresh loop = no uvloop = nest_asyncio can patch it cleanly.

    Beginner example:
        question  = "What is the deadline after a data breach?"
        answer    = "72 hours"
        contexts  = ["Article 33: notify within 72 hours of becoming aware..."]
        → faithfulness=1.0 (answer claim is in context)
        → answer_relevancy=0.95 (answer directly addresses the deadline question)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        from ragas import evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy
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

        # Instantiate fresh metric objects — RAGAS 0.2.x module-level singletons
        # don't reliably pick up llm/embeddings set after import in a thread context
        faithfulness_metric = Faithfulness(llm=llm)
        answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        scores = result.to_pandas()
        return {
            "faithfulness": float(scores["faithfulness"].iloc[0]) if "faithfulness" in scores.columns else None,
            "answer_relevancy": float(scores["answer_relevancy"].iloc[0]) if "answer_relevancy" in scores.columns else None,
        }
    finally:
        loop.close()


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
        result_holder = {}
        error_holder = {}

        def thread_target():
            try:
                result_holder["scores"] = _run_ragas_in_thread(
                    question, answer, contexts,
                    settings.openai_api_key, settings.embedding_model
                )
            except Exception as e:
                error_holder["error"] = e

        t = threading.Thread(target=thread_target, daemon=True)
        t.start()
        await asyncio.get_event_loop().run_in_executor(None, t.join)

        if "error" in error_holder:
            raise error_holder["error"]

        scores = result_holder["scores"]

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
