"""
Embedder + Database Writer
--------------------------
Takes document chunks, converts them to vectors via OpenAI,
and writes everything to PostgreSQL.

Why batch embedding?
- OpenAI API has rate limits (tokens per minute)
- Sending 500 chunks one-by-one = 500 API calls = slow + rate-limited
- Batching sends 50 chunks per call = 10 calls total = 10x faster
- Each batch call returns a list of 1536-dimensional vectors in the same order

The embedding model: text-embedding-3-small
- 1536 dimensions
- Cost: $0.02 per 1M tokens — essentially free for this project
- Better than ada-002 at the same price point
"""

import uuid
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain.schema import Document
from app.core.config import settings
import asyncio

# Async OpenAI client — doesn't block the server while waiting for API response
client = AsyncOpenAI(api_key=settings.openai_api_key)

BATCH_SIZE = 50  # number of chunks to embed per API call


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Send a batch of texts to OpenAI and return their embedding vectors.

    Returns: list of vectors, one per input text
    Each vector is a list of 1536 floats.

    Example (simplified):
        embed_texts(["What is GDPR?"]) -> [[0.23, -0.87, 0.44, ...]]  # 1536 numbers
    """
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    # response.data is a list of Embedding objects, sorted by index
    # We extract just the float list from each
    return [item.embedding for item in response.data]


async def embed_and_store(
    chunks: list[Document],
    db: AsyncSession,
    source_file: str,
    source_url: str,
    document_version: str = "1.0"
) -> int:
    """
    Main ingestion function:
    1. Split chunks into batches of BATCH_SIZE
    2. Embed each batch via OpenAI
    3. Write each chunk + its vector to PostgreSQL

    Returns: total number of chunks stored
    """
    total_stored = 0

    # Process in batches to respect OpenAI rate limits
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]

        # Call OpenAI — get back one vector per chunk
        vectors = await embed_texts(texts)

        # Write each chunk + vector to PostgreSQL
        for chunk, vector in zip(batch, vectors):
            chunk_id = str(uuid.uuid4())

            # pgvector expects the vector as a Python list of floats
            # SQLAlchemy + pgvector handles the serialisation automatically
            await db.execute(
                text("""
                    INSERT INTO document_chunks
                        (id, source_file, source_url, page_number, section_title,
                         chunk_text, embedding, token_count, document_version)
                    VALUES
                        (:id, :source_file, :source_url, :page_number, :section_title,
                         :chunk_text, :embedding, :token_count, :document_version)
                """),
                {
                    "id": chunk_id,
                    "source_file": source_file,
                    "source_url": source_url,
                    "page_number": chunk.metadata.get("page_number"),
                    "section_title": chunk.metadata.get("section_title"),
                    "chunk_text": chunk.page_content,
                    "embedding": str(vector),  # pgvector accepts "[0.1, 0.2, ...]" string format
                    "token_count": chunk.metadata.get("token_count"),
                    "document_version": document_version,
                }
            )
            total_stored += 1

        await db.commit()

        # Brief pause between batches to stay within OpenAI rate limits
        # 0.5 seconds between batches of 50 = safe for tier-1 accounts
        if batch_start + BATCH_SIZE < len(chunks):
            await asyncio.sleep(0.5)

    return total_stored


async def update_ingested_document(
    db: AsyncSession,
    filename: str,
    source_url: str,
    doc_type: str,
    total_chunks: int,
    content_hash: str,
    status: str = "complete",
    error_message: str = None
):
    """
    Record or update the ingestion status in the ingested_documents table.
    Called after embed_and_store completes (or fails).
    """
    await db.execute(
        text("""
            INSERT INTO ingested_documents
                (filename, source_url, doc_type, total_chunks, status,
                 error_message, content_hash, ingested_at)
            VALUES
                (:filename, :source_url, :doc_type, :total_chunks, :status,
                 :error_message, :content_hash, NOW())
            ON CONFLICT (filename) DO UPDATE SET
                total_chunks = EXCLUDED.total_chunks,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                content_hash = EXCLUDED.content_hash,
                ingested_at = NOW()
        """),
        {
            "filename": filename,
            "source_url": source_url,
            "doc_type": doc_type,
            "total_chunks": total_chunks,
            "status": status,
            "error_message": error_message,
            "content_hash": content_hash,
        }
    )
    await db.commit()
