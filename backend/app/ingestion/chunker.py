"""
Chunker
-------
Semantic chunking: splits at meaning boundaries, not fixed character counts.

Why semantic chunking over fixed-size?
Fixed-size splitting (RecursiveCharacterTextSplitter) cuts every 512 chars
regardless of meaning. If Article 33 of GDPR says "notify within 72 hours"
across a chunk boundary, retrieval returns only half the answer.

Semantic chunking embeds each sentence and measures cosine distance between
adjacent sentences. When distance exceeds a threshold, it's a topic boundary
— split there. This keeps related sentences together in the same chunk.

How SemanticChunker works:
1. Split document into sentences
2. Embed each sentence (OpenAI text-embedding-3-small)
3. Compute cosine distance between sentence i and sentence i+1
4. Split where distance > percentile_threshold (default: 95th percentile)
   = split at the top 5% most abrupt topic changes

Beginner example:
  "Article 33 defines breach notification. The controller must notify within 72 hours."
  → These are semantically related → stay in same chunk
  "Article 34 concerns communication to data subjects. This is a separate obligation."
  → Topic shift detected → split here

Cost: one embedding call per sentence at ingestion time (one-time cost).
Retrieval quality improvement: significant — especially for legal text with
articles that flow across paragraph boundaries.
"""

import re
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings


def extract_section_title(text: str) -> str | None:
    """
    Extract a heading or article title from chunk text for citation metadata.

    Examples matched:
        "Article 33 — Notification of Data Breach"
        "## 7. RULES & REMINDERS"
        "3.2 Customer Data"
    """
    md_heading = re.match(r"^#{1,4}\s+(.+)", text.strip())
    if md_heading:
        return md_heading.group(1).strip()

    article_match = re.match(r"^(Article\s+\d+|§\s*\d+)[^\n]*", text.strip(), re.IGNORECASE)
    if article_match:
        return article_match.group(0).strip()

    section_match = re.match(r"^(\d+\.[\d.]*\s+[A-Z][^\n]{5,50})", text.strip())
    if section_match:
        return section_match.group(1).strip()

    return None


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents using SemanticChunker.

    breakpoint_threshold_type="percentile": split at the top N% most abrupt
    topic changes across the document. 95th percentile = split only at the
    5% sharpest meaning shifts — keeps clauses and articles intact.

    This produces variable-size chunks (some short, some long) which is correct:
    a one-sentence article definition stays as one chunk rather than being
    merged with unrelated content just to hit 512 chars.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,  # split at top 5% topic shifts
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["section_title"] = extract_section_title(chunk.page_content)
        chunk.metadata["token_count"] = len(chunk.page_content.split())
        chunk.metadata["chunk_index"] = i

    return chunks
