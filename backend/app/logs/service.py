"""
Logs Service
------------
Fetches query logs and system health summary from PostgreSQL.
Called by the /logs and /health endpoints.
Used by the Next.js dashboard to populate the live query table.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.schemas import QueryLogEntry, QueryLogsResponse, SystemHealth
import json


async def get_recent_logs(db: AsyncSession, limit: int = 50) -> QueryLogsResponse:
    """
    Fetch the most recent query logs ordered by time descending.
    The dashboard calls this every 5 seconds to refresh the live table.
    """
    result = await db.execute(
        text("""
            SELECT
                id::text,
                queried_at,
                query_text,
                answered,
                answer_text,
                top_rrf_score,
                confidence_score,
                latency_total_ms,
                citations,
                error_message
            FROM query_logs
            ORDER BY queried_at DESC
            LIMIT :limit
        """),
        {"limit": limit}
    )
    rows = result.mappings().all()

    logs = []
    for row in rows:
        citations = row["citations"]
        if isinstance(citations, str):
            try:
                citations = json.loads(citations)
            except Exception:
                citations = []

        logs.append(QueryLogEntry(
            id=str(row["id"]),
            queried_at=row["queried_at"],
            query_text=row["query_text"],
            answered=row["answered"],
            answer_text=row["answer_text"],
            top_rrf_score=float(row["top_rrf_score"]) if row["top_rrf_score"] else None,
            confidence_score=float(row["confidence_score"]) if row["confidence_score"] else None,
            latency_total_ms=row["latency_total_ms"],
            citations=citations,
            error_message=row["error_message"],
        ))

    count_result = await db.execute(text("SELECT COUNT(*) FROM query_logs"))
    total_count = count_result.scalar()

    return QueryLogsResponse(logs=logs, total_count=total_count)


async def get_latest_log_id(db: AsyncSession, question: str) -> str | None:
    """
    Fetch the UUID of the most recently written query_log row for a question.
    Called right after process_query() to get the log ID for RAGAS eval linking.
    """
    result = await db.execute(
        text("""
            SELECT id::text FROM query_logs
            WHERE query_text = :question
            ORDER BY queried_at DESC
            LIMIT 1
        """),
        {"question": question}
    )
    row = result.fetchone()
    return row[0] if row else None


async def get_eval_health(db: AsyncSession) -> dict:
    """
    RAGAS eval summary for the last 24 hours.
    Returns avg faithfulness and answer relevancy, or None if no evals yet.
    """
    result = await db.execute(
        text("""
            SELECT
                COUNT(*) AS total_evaluated,
                ROUND(AVG(faithfulness)::NUMERIC, 3) AS avg_faithfulness,
                ROUND(AVG(answer_relevancy)::NUMERIC, 3) AS avg_answer_relevancy
            FROM eval_scores
            WHERE evaluated_at >= NOW() - INTERVAL '24 hours'
        """)
    )
    row = result.mappings().one()
    return {
        "total_evaluated": row["total_evaluated"] or 0,
        "avg_faithfulness": float(row["avg_faithfulness"]) if row["avg_faithfulness"] else None,
        "avg_answer_relevancy": float(row["avg_answer_relevancy"]) if row["avg_answer_relevancy"] else None,
    }


async def get_system_health(db: AsyncSession) -> SystemHealth:
    """
    Compute system health metrics for the last 24 hours.
    Shown at the top of the dashboard as a summary panel.
    """
    result = await db.execute(
        text("""
            SELECT
                COUNT(*) AS total_queries,
                COUNT(*) FILTER (WHERE answered = TRUE) AS total_answered,
                COUNT(*) FILTER (WHERE answered = FALSE) AS total_refused,
                ROUND(
                    COUNT(*) FILTER (WHERE answered = TRUE)::NUMERIC
                    / NULLIF(COUNT(*), 0) * 100, 1
                ) AS answer_rate_pct,
                ROUND(AVG(latency_total_ms)::NUMERIC, 0) AS avg_latency_ms,
                ROUND(
                    AVG(confidence_score) FILTER (WHERE answered = TRUE)::NUMERIC, 3
                ) AS avg_confidence
            FROM query_logs
            WHERE queried_at >= NOW() - INTERVAL '24 hours'
        """)
    )
    row = result.mappings().one()

    return SystemHealth(
        total_queries=row["total_queries"] or 0,
        total_answered=row["total_answered"] or 0,
        total_refused=row["total_refused"] or 0,
        answer_rate_pct=float(row["answer_rate_pct"] or 0),
        avg_latency_ms=float(row["avg_latency_ms"]) if row["avg_latency_ms"] else None,
        avg_confidence=float(row["avg_confidence"]) if row["avg_confidence"] else None,
    )
