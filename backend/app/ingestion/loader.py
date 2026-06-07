"""
Document Loader
---------------
Reads raw files and returns a list of (text, metadata) tuples.
Each tuple = one page or section of the document.

Supports: PDF, Markdown, plain text.
"""

import hashlib
import re
from pathlib import Path
import pdfplumber
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document


def compute_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _clean_pdf_text(text: str) -> str:
    """
    Fix common PDF extraction artifacts:
    - Remove mid-word spaces: "pr otection" → "protection"
    - Collapse multiple spaces
    - Remove soft hyphens at line breaks: "regu-\nlation" → "regulation"

    Why pdfplumber over PyPDFLoader?
    EU legislation PDFs (like GDPR) store text as positioned glyphs — PyPDF
    inserts spaces between every glyph group, producing "Gener al Data Pr otection".
    pdfplumber uses character bounding-box proximity to merge glyphs correctly.
    This cleanup pass catches any remaining artifacts.

    Beginner example: "H e llo W or ld" → "Hello World"
    """
    # Remove soft hyphens at line breaks
    text = re.sub(r'-\n(\S)', r'\1', text)
    # Fix "wo rd" style mid-word spaces (letter, space, 1-3 letters, space or end)
    text = re.sub(r'(?<=[a-zA-Z]) ([a-zA-Z]{1,3})(?=[^a-zA-Z])', r'\1', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def load_document(file_path: str) -> list[Document]:
    """
    Load a document from disk and return LangChain Document objects.

    For PDFs: uses pdfplumber (cleaner text extraction than PyPDF — handles
    EU legislation format without mid-word spaces).
    For MD/TXT: standard TextLoader.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                clean_text = _clean_pdf_text(raw_text)
                if len(clean_text.strip()) < 20:
                    continue  # skip blank/header-only pages
                documents.append(Document(
                    page_content=clean_text,
                    metadata={
                        "source_file": path.name,
                        "source_url": str(path),
                        "page_number": page_num,
                    }
                ))

    elif extension in (".md", ".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = path.name
            doc.metadata["source_url"] = str(path)
            doc.metadata["page_number"] = None

    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported: .pdf, .md, .txt")

    return documents
