-- DataVault RAG System — PostgreSQL Schema
-- Run this once on a fresh database before starting the application

-- ============================================================
-- EXTENSIONS
-- ============================================================

-- pgvector: adds the vector data type and cosine similarity search
-- Without this, PostgreSQL cannot store or query embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- uuid-ossp: generates universally unique IDs for each chunk
-- Better than auto-increment integers — safe across distributed systems
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ============================================================
-- TABLE 1: document_chunks
-- Stores every chunk from every ingested document
-- This is the core of the RAG system — everything retrieval depends on this
-- ============================================================

CREATE TABLE IF NOT EXISTS document_chunks (

    -- Unique ID for this chunk — used in citations
    -- uuid_generate_v4() creates a random UUID like: a3f2c1d0-...
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Which document this chunk came from
    -- Example: "gdpr_full_regulation.pdf" or "datavault_datenschutzrichtlinie.md"
    source_file TEXT NOT NULL,

    -- Full path or URL to the source document
    -- Used in citations so the user can navigate to the original
    source_url TEXT NOT NULL,

    -- Page number in the original document
    -- For PDFs: actual page number. For markdown: estimated section number.
    page_number INTEGER,

    -- Which section or heading this chunk falls under
    -- Example: "Article 33 — Notification of Data Breach"
    section_title TEXT,

    -- The actual text content of this chunk
    -- This is what gets passed to the LLM as context
    chunk_text TEXT NOT NULL,

    -- The embedding vector — 1536 dimensions for OpenAI text-embedding-3-small
    -- This is what pgvector uses for semantic (cosine similarity) search
    embedding vector(1536) NOT NULL,

    -- ts_vector column for BM25 full-text keyword search
    -- PostgreSQL automatically computes this from chunk_text
    -- 'english' means it strips stopwords and stems words in English
    -- Example: "processing" and "processed" both map to "process"
    chunk_text_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', chunk_text)
    ) STORED,

    -- Token count of this chunk — useful for monitoring chunk sizes
    token_count INTEGER,

    -- When this chunk was ingested into the system
    ingested_at TIMESTAMPTZ DEFAULT NOW(),

    -- Which version of the source document this came from
    -- Important for compliance: if GDPR is updated, old chunks are marked stale
    document_version TEXT DEFAULT '1.0'
);


-- ============================================================
-- INDEXES — this is what makes search fast
-- ============================================================

-- Index 1: HNSW index for pgvector cosine similarity search
-- HNSW = Hierarchical Navigable Small World graph
-- How it works: builds a multi-layer graph of vector neighbors at index time
--   At query time, navigates the graph in O(log n) — much faster than IVFFlat
-- m=16: connections per node (higher = better recall, more memory)
-- ef_construction=64: candidates considered during build (higher = better quality index)
-- 5,250x faster than sequential scan, 1.5x faster than IVFFlat at same recall
-- Does NOT require knowing dataset size upfront — safe default for production
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index 2: GIN index for BM25 full-text search
-- GIN = Generalised Inverted Index
-- How it works: maps every word to the rows that contain it
--   Like the index at the back of a textbook — word → page numbers
-- Without this, BM25 search does a full table scan (slow)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON document_chunks
    USING GIN (chunk_text_tsv);

-- Index 3: B-tree index on source_file
-- Allows fast filtering by document — useful for multi-tenant or filtered search
CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON document_chunks (source_file);


-- ============================================================
-- TABLE 2: query_logs
-- Stores every query made to the system — this is your observability layer
-- Every row = one user question + everything that happened to answer it
-- ============================================================

CREATE TABLE IF NOT EXISTS query_logs (

    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- The exact question the user asked
    query_text TEXT NOT NULL,

    -- Top chunk IDs retrieved after RRF fusion and reranking
    -- Stored as an array of UUIDs — links back to document_chunks.id
    retrieved_chunk_ids UUID[],

    -- Cosine similarity scores for each retrieved chunk (same order as chunk_ids)
    -- Example: [0.91, 0.87, 0.83, 0.79, 0.74]
    retrieval_scores FLOAT[],

    -- The fused RRF score for the top chunk (used for confidence gate decision)
    top_rrf_score FLOAT,

    -- Did the system answer or refuse?
    -- TRUE = answered (score >= 0.75)
    -- FALSE = refused ("Insufficient data" returned, no LLM call made)
    answered BOOLEAN NOT NULL,

    -- The final answer returned to the user
    -- NULL if answered = FALSE
    answer_text TEXT,

    -- Citations returned with the answer
    -- Stored as JSONB — flexible structure
    -- Example: [{"chunk_id": "uuid", "source_file": "gdpr.pdf", "page_number": 33}]
    citations JSONB,

    -- Confidence score returned by the LLM in its structured response
    -- Between 0 and 1 — reflects how well the answer is grounded in context
    confidence_score FLOAT,

    -- Total time from query received to response sent — in milliseconds
    -- Broken down into stages for debugging
    latency_total_ms INTEGER,
    latency_embedding_ms INTEGER,   -- time to embed the query
    latency_retrieval_ms INTEGER,   -- time for vector + BM25 search + RRF
    latency_rerank_ms INTEGER,      -- time for reranking
    latency_llm_ms INTEGER,         -- time for OpenAI GPT-4o response

    -- Model used for this query — useful if you ever A/B test models
    llm_model TEXT DEFAULT 'gpt-4o',

    -- Number of tokens sent to and received from the LLM
    -- Used to track API costs
    tokens_input INTEGER,
    tokens_output INTEGER,

    -- Any error that occurred — NULL if successful
    -- Example: "OpenAI API timeout after 30s"
    error_message TEXT,

    -- Timestamp of the query
    queried_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- INDEXES on query_logs
-- ============================================================

-- Index on queried_at — for fetching recent logs in the dashboard
-- The dashboard fetches "last 50 queries" — this makes it instant
CREATE INDEX IF NOT EXISTS idx_logs_queried_at
    ON query_logs (queried_at DESC);

-- Index on answered — for filtering refused vs answered queries
-- Useful for monitoring: "how many queries were refused today?"
CREATE INDEX IF NOT EXISTS idx_logs_answered
    ON query_logs (answered);


-- ============================================================
-- TABLE 3: eval_scores
-- RAGAS evaluation scores written async after each answered query
-- Measures answer quality from the outside (not LLM self-grading)
-- Must come AFTER query_logs because it references query_logs(id)
-- ============================================================

CREATE TABLE IF NOT EXISTS eval_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_log_id UUID NOT NULL REFERENCES query_logs(id) ON DELETE CASCADE,

    -- RAGAS faithfulness: does the answer stay within the retrieved chunks?
    -- 1.0 = every claim in the answer is grounded in context
    -- 0.0 = answer contains claims not supported by any retrieved chunk (hallucination)
    faithfulness FLOAT,

    -- RAGAS answer relevancy: does the answer actually address the question?
    -- 1.0 = answer directly answers what was asked
    -- 0.0 = answer is technically grounded but doesn't address the question
    answer_relevancy FLOAT,

    ragas_version TEXT DEFAULT '0.2',
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_query_log_id
    ON eval_scores (query_log_id);

CREATE INDEX IF NOT EXISTS idx_eval_evaluated_at
    ON eval_scores (evaluated_at DESC);


-- ============================================================
-- TABLE 4: ingested_documents
-- Tracks which documents have been ingested and their status
-- Prevents duplicate ingestion and tracks document versions
-- ============================================================

CREATE TABLE IF NOT EXISTS ingested_documents (

    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Filename of the document
    filename TEXT NOT NULL UNIQUE,

    -- Full path or URL
    source_url TEXT NOT NULL,

    -- Document type — determines which loader to use
    -- Values: 'pdf', 'markdown', 'txt'
    doc_type TEXT NOT NULL,

    -- Total chunks created from this document
    total_chunks INTEGER,

    -- Ingestion status
    -- Values: 'pending', 'processing', 'complete', 'failed'
    status TEXT DEFAULT 'pending',

    -- Error message if ingestion failed
    error_message TEXT,

    -- Document version — bump this when document is updated
    document_version TEXT DEFAULT '1.0',

    -- Hash of the file content — used to detect if document has changed
    -- If hash changes, re-ingestion is triggered
    content_hash TEXT,

    ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- USEFUL VIEWS — pre-built queries for the dashboard
-- ============================================================

-- View 1: Recent queries with chunk details joined in
-- The dashboard uses this to show the live query log table
CREATE OR REPLACE VIEW recent_query_logs AS
SELECT
    ql.id,
    ql.queried_at,
    ql.query_text,
    ql.answered,
    ql.answer_text,
    ql.top_rrf_score,
    ql.confidence_score,
    ql.latency_total_ms,
    ql.citations,
    ql.error_message
FROM query_logs ql
ORDER BY ql.queried_at DESC
LIMIT 100;

-- View 2: System health summary
-- Shows at a glance: total queries, answer rate, avg latency, avg confidence
CREATE OR REPLACE VIEW system_health AS
SELECT
    COUNT(*) AS total_queries,
    COUNT(*) FILTER (WHERE answered = TRUE) AS total_answered,
    COUNT(*) FILTER (WHERE answered = FALSE) AS total_refused,
    ROUND(
        COUNT(*) FILTER (WHERE answered = TRUE)::NUMERIC / NULLIF(COUNT(*), 0) * 100,
        1
    ) AS answer_rate_pct,
    ROUND(AVG(latency_total_ms)::NUMERIC, 0) AS avg_latency_ms,
    ROUND(AVG(confidence_score) FILTER (WHERE answered = TRUE)::NUMERIC, 3) AS avg_confidence,
    DATE_TRUNC('day', NOW()) AS as_of_date
FROM query_logs
WHERE queried_at >= NOW() - INTERVAL '24 hours';

-- View 3: RAGAS eval health — avg faithfulness + answer relevancy over 24h
-- Shows whether answer quality is degrading over time
CREATE OR REPLACE VIEW eval_health AS
SELECT
    COUNT(*) AS total_evaluated,
    ROUND(AVG(faithfulness)::NUMERIC, 3) AS avg_faithfulness,
    ROUND(AVG(answer_relevancy)::NUMERIC, 3) AS avg_answer_relevancy,
    DATE_TRUNC('day', NOW()) AS as_of_date
FROM eval_scores
WHERE evaluated_at >= NOW() - INTERVAL '24 hours';
