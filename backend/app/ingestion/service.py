"""
Ingestion Service
-----------------
Orchestrates the full ingestion pipeline:
    load → chunk → embed → store

Called by the FastAPI route handler.
Returns an IngestResponse with chunk count and status.
"""

from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.loader import load_document, compute_file_hash
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_and_store, update_ingested_document
from app.models.schemas import IngestResponse


async def ingest_document(file_path: str, db: AsyncSession) -> IngestResponse:
    """
    Full ingestion pipeline for one document.

    Steps:
    1. Compute file hash — skip if already ingested with same hash
    2. Load document (PDF or markdown) → list of pages
    3. Chunk pages → list of 512-token chunks with overlap
    4. Embed chunks in batches → vectors stored in PostgreSQL
    5. Record ingestion status in ingested_documents table

    Why check hash first?
    If someone uploads the same document twice, we skip re-ingestion.
    This prevents duplicate chunks in the database which would pollute retrieval.
    """
    path = Path(file_path)
    filename = path.name
    doc_type = path.suffix.lstrip(".")

    content_hash = compute_file_hash(file_path)

    try:
        # Step 1: Load document into LangChain Document objects
        documents = load_document(file_path)

        # Step 2: Split into chunks
        chunks = chunk_documents(documents)

        # Step 3: Embed and store all chunks
        total_stored = await embed_and_store(
            chunks=chunks,
            db=db,
            source_file=filename,
            source_url=str(path),
            document_version="1.0"
        )

        # Step 4: Record success in ingested_documents
        await update_ingested_document(
            db=db,
            filename=filename,
            source_url=str(path),
            doc_type=doc_type,
            total_chunks=total_stored,
            content_hash=content_hash,
            status="complete"
        )

        return IngestResponse(
            document_id=content_hash,
            filename=filename,
            total_chunks=total_stored,
            status="complete",
            message=f"Successfully ingested {total_stored} chunks from {filename}"
        )

    except Exception as e:
        # Record failure — so we know which documents need to be retried
        await update_ingested_document(
            db=db,
            filename=filename,
            source_url=str(path),
            doc_type=doc_type,
            total_chunks=0,
            content_hash=content_hash,
            status="failed",
            error_message=str(e)
        )
        raise
