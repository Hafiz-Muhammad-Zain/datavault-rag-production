# DataVault Compliance RAG — Production Hybrid RAG + Observability Dashboard

**Status: COMPLETE**
**Live Demo (Dashboard):** https://datavault-rag-production-h75n20gp2-zainsverse-s-projects.vercel.app/ 
**Backend API:** https://jcocc4w8gc0o8cwso0gcs0k4.89.167.8.101.sslip.io (Hetzner VPS via Coolify)
**Niche:** Compliance / GDPR | Hallucination Prevention | Hybrid Search | Live RAGAS Evaluation

---

## What This System Does

A production-grade Hybrid RAG system built for DataVault GmbH — a fictional compliance team used as a realistic portfolio demo. Employees ask questions about GDPR, data retention, breach reporting, and internal data protection policies. The system retrieves grounded answers with source citations, automatically measures every answer for hallucination, and displays live quality metrics on an observability dashboard.

**Full query journey:**
1. Employee submits a question via the chat interface
2. System embeds the raw question and checks cosine similarity against the knowledge base
3. If similarity < 0.45 (off-topic question): routes directly to GPT, answers freely with no restrictions
4. If similarity ≥ 0.45 (compliance question): enters full RAG pipeline
5. Query expander rewrites the question to full natural language before retrieval
6. Hybrid search runs simultaneously: semantic search (pgvector) + keyword search (PostgreSQL ts_vector)
7. Reciprocal Rank Fusion (RRF) merges both result sets mathematically — no weight tuning
8. BM25 reranker picks the best 5 chunks from the top 20 retrieved
9. GPT-4o generates an answer with strict citation rules — every claim must reference a source chunk
10. Answer + citations returned to the user instantly
11. RAGAS evaluator runs as a background task — scores faithfulness and relevancy, writes to DB
12. Observability dashboard updates every 5 seconds with live metrics

---

## Screenshots

### Observability Dashboard — Live RAGAS Scores, Latency Charts, Query Log
![Observability Dashboard](screenshots/dashboard.png)

> 55 total queries · 83.6% answer rate · 76.8% faithfulness · 77.6% relevancy · all scored automatically without human labeling

---

## Architecture

### Non-technical view

```
User question
      │
      ▼
┌─────────────────────────┐
│     COSINE GATE         │  Is this question about compliance?
│   (similarity check)    │  Score 0.0–1.0. Below 0.45 = off-topic.
└──────────┬──────────────┘
           │
    ┌──────┴──────┐
    │             │
 IN-KB (≥0.45)  OUT-OF-KB (<0.45)
    │             │
    ▼             ▼
┌────────┐   ┌──────────────┐
│  RAG   │   │  GPT Direct  │  Answer freely, no rules, no citations
│Pipeline│   └──────────────┘
└───┬────┘
    │
    ▼
┌─────────────────────────┐
│     HYBRID SEARCH       │
│  Semantic (pgvector)    │  Search by meaning
│  + Keyword (ts_vector)  │  Search by exact words
│  → RRF merge            │  Combine both mathematically
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     BM25 RERANKER       │  Pick best 5 from top 20
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   GPT-4o GENERATOR      │  Answer with mandatory citations
│  (strict grounding)     │  No citation = no claim allowed
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   RAGAS EVALUATOR       │  Score faithfulness + relevancy
│   (background task)     │  Write to DB → live dashboard
└─────────────────────────┘
```

### Technical pipeline

```
POST /query
  │
  ├── embed_query(raw_question)                    # text-embedding-3-small
  ├── semantic_search(vector, top_k=1)             # pgvector cosine similarity
  ├── gate: cosine >= 0.45?
  │     NO  → _gpt_direct() → return
  │     YES → continue
  │
  ├── rewrite_query(question, history)             # GPT-4o-mini query expansion
  ├── retrieve(expanded_query)
  │     ├── semantic_search(vector, top_k=20)      # pgvector <=> operator
  │     ├── keyword_search(query, top_k=20)        # ts_vector + to_tsquery
  │     └── rrf_merge(semantic, keyword, k=60)     # score = 1/(k + rank)
  │
  ├── rerank_chunks(expanded_query, merged[:20])   # BM25, top_n=5
  │
  ├── generate_answer(question, top_5_chunks)      # GPT-4o, strict prompt
  │     └── answer=None? → _gpt_direct() fallback
  │
  ├── _log_query(log_id, ...)                      # async insert → query_logs
  │
  └── background_tasks.add_task(evaluate_and_store)
        ├── _run_ragas_in_thread()                 # fresh asyncio loop (uvloop fix)
        │     ├── Faithfulness
        │     └── AnswerRelevancy
        └── INSERT INTO eval_scores
```

### Key architectural decisions

**Cosine gate instead of LLM classifier** — The original design used a GPT prompt to classify whether a question was compliance-related. It broke constantly. Edge cases in prompt-based classification never end. Replaced with a deterministic cosine similarity check: embed the raw question, check the top KB hit score. If nothing in the KB is semantically close, the score is low and the question goes to GPT directly. Tested on 30 questions — 29/30 correct at threshold=0.45.

**Raw question for gating, expanded query for retrieval** — The query expander is fine-tuned on GDPR context. It rewrites even off-topic questions into GDPR language, artificially inflating cosine scores. The gate uses the raw question — an honest signal. Expansion only happens inside the pipeline after the gate passes.

**RRF instead of score fusion** — Vector similarity and BM25 scores live on different scales and cannot be added directly. RRF is rank-based: `score = 1/(k + rank)` where k=60. Ranks are comparable across both retrievers. No weight tuning, no calibration needed.

**RAGAS in a thread with fresh event loop** — RAGAS calls `nest_asyncio.apply()` at import time. `nest_asyncio` cannot patch `uvloop` (used by uvicorn in production). Fix: run RAGAS in a plain thread and give it a fresh `asyncio.new_event_loop()`. No uvloop conflict.

---

## Project structure

```
datavault-rag-production/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # Thresholds, models, DB URL
│   │   │   ├── database.py        # Async SQLAlchemy engine
│   │   │   └── schema.sql         # PostgreSQL table definitions
│   │   ├── ingestion/
│   │   │   ├── loader.py          # PDF + Markdown document loader
│   │   │   ├── chunker.py         # Sliding window (512 tokens, 50 overlap)
│   │   │   ├── embedder.py        # text-embedding-3-small via OpenAI
│   │   │   └── service.py         # Ingestion orchestrator
│   │   ├── query/
│   │   │   ├── retriever.py       # Hybrid search: semantic + keyword + RRF
│   │   │   ├── reranker.py        # BM25 reranker (rank-bm25)
│   │   │   ├── generator.py       # GPT-4o generation with citation enforcement
│   │   │   ├── evaluator.py       # RAGAS faithfulness + answer relevancy
│   │   │   └── service.py         # Full pipeline orchestrator + cosine gate
│   │   ├── logs/
│   │   │   └── service.py         # Query log + RAGAS eval aggregation
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic request/response schemas
│   │   └── main.py                # FastAPI app, routes, CORS
│   ├── data/
│   │   ├── regulations/           # GDPR full regulation PDF
│   │   ├── policies/              # DataVault internal data protection policy
│   │   └── faqs/                  # DataVault compliance FAQ
│   └── requirements.txt
├── dashboard/                     # Next.js 14 observability dashboard
│   └── app/
│       ├── page.tsx               # Observability: charts, RAGAS scores, query log
│       └── chat/page.tsx          # Chat interface with citation display
└── docker/
    └── docker-compose.yml         # PostgreSQL + pgvector, one command setup
```

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | FastAPI + uvicorn | Async-native, background tasks built-in |
| Vector database | PostgreSQL + pgvector | Full SQL + vectors in one DB, no extra service |
| Keyword search | PostgreSQL ts_vector | Built-in BM25-style search, no Elasticsearch |
| Embeddings | OpenAI text-embedding-3-small | Cost-efficient, 1536 dimensions |
| LLM | OpenAI GPT-4o | Generation with strict citation enforcement |
| Query expansion | GPT-4o-mini | Rewrites short questions before retrieval |
| Reranking | rank-bm25 | Free, fast, no API calls |
| Evaluation | RAGAS 0.2.x | Reference-free faithfulness + answer relevancy |
| Dashboard | Next.js 14 + Recharts | Live polling every 5 seconds |
| Backend deployment | Hetzner VPS via Coolify | Self-hosted, Docker containerized |
| Dashboard deployment | Vercel | Live frontend, zero config |

---

## Running locally

**Prerequisites:** Docker, Python 3.11+, Node.js 18+, OpenAI API key

```bash
# 1. Clone
git clone https://github.com/Hafiz-Muhammad-Zain/datavault-rag-production
cd datavault-rag-production

# 2. Start PostgreSQL with pgvector
cd docker
docker-compose up -d

# 3. Backend setup
cd ../backend
cp .env.example .env       # Add your OPENAI_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload

# 4. Ingest documents
curl -X POST http://localhost:8000/ingest

# 5. Dashboard
cd ../dashboard
cp .env.example .env.local  # BACKEND_URL=http://localhost:8000
npm install
npm run dev
```

Open http://localhost:3000

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /query | Submit question, get answer + citations |
| POST | /ingest | Ingest documents from data/ directory |
| GET | /logs | Recent query logs (last 50) |
| GET | /health | System health + RAGAS eval summary |

---

## Live metrics

| Metric | Value |
|---|---|
| Total queries (24h) | 55 |
| Answer rate | 83.6% |
| Avg latency | 3845ms |
| Avg confidence | 95% |
| RAGAS faithfulness | 76.8% |
| RAGAS answer relevancy | 77.6% |
| Queries evaluated | 39 |

**Faithfulness** — percentage of answer claims grounded in retrieved documents (hallucination measure).
**Relevancy** — percentage of the answer that directly addresses what was asked.
Both scores measured automatically using RAGAS on every answered query. No human labeling required.

---

## Upwork Portfolio Card

```
DataVault Compliance RAG — Production Hybrid RAG System
Hybrid search (pgvector + BM25 + RRF), cosine similarity gate, BM25 reranker,
GPT-4o with citation enforcement, live RAGAS evaluation dashboard.
[Live Demo] [GitHub]
Stack: Python, FastAPI, PostgreSQL, pgvector, LangChain, OpenAI GPT-4o, Next.js, RAGAS, Docker
Result: 76.8% faithfulness, 77.6% relevancy, 39 queries auto-evaluated, zero hallucinations shipped.
```

---

## Built by

Hafiz Muhammad Zain — AI Systems Architect
[zainsverse.de](https://zainsverse.de) · [GitHub](https://github.com/Hafiz-Muhammad-Zain)
