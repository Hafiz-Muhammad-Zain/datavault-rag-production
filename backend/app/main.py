"""
FastAPI Application
-------------------
Entry point for the RAG backend.
Defines all HTTP routes and wires them to the pipeline services.
"""

import os
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, check_db_connection
from app.core.config import settings
from app.models.schemas import (
    QueryRequest, QueryResponse,
    IngestResponse, HealthResponse,
    QueryLogsResponse, SystemHealth
)
from app.ingestion.service import ingest_document
from app.query.service import process_query
from app.logs.service import get_recent_logs, get_system_health, get_eval_health

# ── APP SETUP ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataVault Compliance RAG API",
    description="Production RAG system with hybrid search, hallucination prevention, and live observability.",
    version="1.0.1",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc UI at /redoc
)

# CORS — allows the Next.js dashboard (on Vercel) to call this API
# In production, replace "*" with your actual Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HEALTH ENDPOINT ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns system status and whether PostgreSQL is reachable.
    Used by Docker healthcheck, Coolify, and uptime monitors.
    """
    db_ok = await check_db_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database=db_ok,
        version="1.0.1"
    )


# ── INGESTION ENDPOINT ─────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document (PDF, MD, TXT) to ingest into the knowledge base.

    Process:
    1. Save uploaded file to /tmp
    2. Run ingestion pipeline (load → chunk → embed → store)
    3. Return chunk count and status

    Supported formats: .pdf, .md, .txt
    """
    allowed_extensions = {".pdf", ".md", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    # Save to temp file — ingestion pipeline reads from disk
    tmp_path = f"/tmp/{file.filename}"
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        result = await ingest_document(file_path=tmp_path, db=db)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class LocalIngestRequest(BaseModel):
    file_path: str

@app.post("/ingest/local", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_local_file(
    request: LocalIngestRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest a file that already exists on the server's filesystem.
    Used for the initial bulk ingestion of the knowledge base documents
    (gdpr_full_regulation.pdf, datavault policy, FAQ).

    Call this once per document to populate the knowledge base.
    """
    if not Path(request.file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    try:
        result = await ingest_document(file_path=request.file_path, db=db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── QUERY ENDPOINT ─────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Ask a question against the ingested knowledge base.

    Pipeline:
    1. Intent classify — COMPLIANCE or CONVERSATIONAL
    2. Embed + expand query
    3. Semantic search (pgvector) + keyword search (BM25) in parallel
    4. Merge results with Reciprocal Rank Fusion
    5. Rerank top candidates
    6. Confidence gate — refuse if score below threshold
    7. Generate answer with GPT-4o, enforce citations
    8. Log everything to query_logs table
    9. Fire RAGAS evaluation as background task (non-blocking)

    Returns: answer + citations + confidence score + latency breakdown
    """
    try:
        # background_tasks passed in so process_query can fire RAGAS
        # with full chunk texts (not citation excerpts — those are too short for RAGAS)
        result = await process_query(request=request, db=db, background_tasks=background_tasks)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── LOGS ENDPOINTS ─────────────────────────────────────────────────────────
@app.get("/logs", response_model=QueryLogsResponse, tags=["Observability"])
async def get_logs(
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch recent query logs for the live observability dashboard.
    Called by the Next.js frontend every 5 seconds.

    Returns last N queries with: question, answer, citations, scores, latency.
    """
    return await get_recent_logs(db=db, limit=limit)


@app.get("/logs/health", response_model=SystemHealth, tags=["Observability"])
async def get_health_metrics(db: AsyncSession = Depends(get_db)):
    """
    System health summary for the last 24 hours.
    Shown at the top of the dashboard.

    Returns: answer rate %, avg latency, avg confidence, total queries.
    """
    return await get_system_health(db=db)


@app.get("/logs/eval", tags=["Observability"])
async def get_eval_metrics(db: AsyncSession = Depends(get_db)):
    """
    RAGAS evaluation health for the last 24 hours.
    Returns avg faithfulness and answer relevancy scores.
    None if no evaluations have run yet.
    """
    return await get_eval_health(db=db)
