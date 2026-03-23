"""
Unified ingestion pipeline – cloned from the chatbot gold standard.

Key improvements over the original document_ingest.py:
  1. Markdown-header-aware splitting BEFORE recursive chunking (preserves H1>H2>H3).
  2. Rich context injection into every chunk's page_content so the LLM can cite accurately.
  3. Full metadata schema: source, page, page_title, context_header, Header 1/2/3, chunk index.
  4. Works with the existing VectorStoreFactory (Chroma).
"""

import os
import gc
import glob
import json
import time
import sqlite3
import traceback
from typing import List, Generator

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

from app.config import Config
from app.vectorstore.factory import VectorStoreFactory

# Tracking / status files (kept compatible with the rest of the RAG app)
TRACKING_FILE = "/tmp/ingested_files_v2.json"
STATUS_FILE = "/tmp/ingestion_status_v2.json"


# ======================================================================
# 1. LOADING
# ======================================================================

def get_loader(file_path: str):
    """Return the appropriate LangChain loader for a given file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".pdf": lambda: PyMuPDFLoader(file_path),
        ".docx": lambda: Docx2txtLoader(file_path),
        ".txt": lambda: TextLoader(file_path, encoding="utf-8"),
        ".md": lambda: TextLoader(file_path, encoding="utf-8"),
        ".csv": lambda: CSVLoader(file_path),
        ".xlsx": lambda: UnstructuredExcelLoader(file_path),
        ".xls": lambda: UnstructuredExcelLoader(file_path),
    }
    factory = loaders.get(ext)
    return factory() if factory else None


def get_url_from_pages(file_path: str) -> str:
    """Resolve original URL for a file from the crawler's pages table."""
    if not os.path.exists(Config.CRAWLER_DB):
        return file_path
    try:
        conn = sqlite3.connect(Config.CRAWLER_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        basename = os.path.basename(file_path)
        # Search by full path or just filename
        cursor.execute(
            "SELECT url FROM pages WHERE filename = ? OR filename = ? LIMIT 1",
            (file_path, basename)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else file_path
    except Exception:
        return file_path


# ======================================================================
# 2. SPLITTING  (Gold-standard: markdown headers → recursive)
# ======================================================================

_HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def split_markdown_preserves_headers(documents: List[Document]) -> List[Document]:
    """
    For markdown-compatible documents, first split by H1/H2/H3 headers so
    that the section hierarchy is captured in metadata. Non-markdown docs
    pass through unchanged.
    Cloned from chatbot/rag.py → _split_markdown_preserves_headers.
    """
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT)
    result: List[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "").lower()
        is_md_compat = any(
            ext in source for ext in (".md", ".markdown", ".html", ".htm")
        )
        if is_md_compat:
            header_splits = md_splitter.split_text(doc.page_content)
            for split in header_splits:
                split.metadata.update(doc.metadata)
                result.append(split)
        else:
            result.append(doc)

    return result


def recursive_split(documents: List[Document]) -> List[Document]:
    """Apply the recursive character text splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


# ======================================================================
# 3. METADATA ENRICHMENT  (Gold-standard context injection)
# ======================================================================

def inject_context_to_chunks(chunks: List[Document]) -> List[Document]:
    """
    Prepend a rich context header into every chunk's page_content so the LLM
    has grounding information (title, section path, source) even if the
    original text doesn't mention it.

    Cloned from chatbot/rag.py → _inject_context_to_chunks.
    """
    for doc in chunks:
        page_title = doc.metadata.get("page_title", "").strip()
        source = doc.metadata.get("source", "Unknown").strip()
        context_header = doc.metadata.get("context_header", "").strip()
        h1 = doc.metadata.get("Header 1", "")
        h2 = doc.metadata.get("Header 2", "")
        h3 = doc.metadata.get("Header 3", "")
        section_path = " > ".join(filter(None, [h1, h2, h3]))

        header_parts: list[str] = []
        if page_title:
            header_parts.append(f"# {page_title}")
        if section_path:
            header_parts.append(f"**Current Section:** {section_path}")
        if source:
            source_display = source if source.startswith("http") else os.path.basename(source)
            header_parts.append(f"**Source:** {source_display}")
        if header_parts:
            header_parts.append("---")
        if context_header:
            header_parts.append(context_header)

        if header_parts:
            header_str = "\n".join(header_parts)
            doc.page_content = f"{header_str}\n\n{doc.page_content}"

    return chunks


def enrich_chunk_indices(chunks: List[Document]) -> List[Document]:
    """
    Add sequential chunk index per source + ensure every chunk has a `page` field.
    """
    by_source: dict[str, int] = {}
    for doc in chunks:
        src = doc.metadata.get("source", "")
        idx = by_source.get(src, 0)
        doc.metadata["chunk"] = idx
        by_source[src] = idx + 1
        if "page" not in doc.metadata or doc.metadata["page"] is None:
            doc.metadata["page"] = 0
    return chunks


# ======================================================================
# 4. FULL PIPELINE
# ======================================================================

def process_documents(raw_docs: List[Document]) -> List[Document]:
    """
    Run the complete gold-standard pipeline on a list of raw documents:
      1. Markdown header split (preserves H1>H2>H3 in metadata)
      2. Recursive character split
      3. Chunk index assignment
      4. Rich context injection
    """
    docs = split_markdown_preserves_headers(raw_docs)
    docs = recursive_split(docs)
    docs = enrich_chunk_indices(docs)
    docs = inject_context_to_chunks(docs)
    return docs


# ======================================================================
# 5. TRACKING HELPERS  (unchanged from original for compatibility)
# ======================================================================

def load_tracking_data() -> dict:
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_tracking_data(data: dict):
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=4)


def update_status(status, current=0, total=0, message="", start_time=None):
    data = {
        "status": status,
        "current_batch": current,
        "total_batches": total,
        "message": message,
        "timestamp": time.time(),
    }
    if start_time and current > 0:
        elapsed = time.time() - start_time
        remaining = total - current
        data["eta_seconds"] = int(remaining * (elapsed / current))
    else:
        data["eta_seconds"] = None

    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)


# ======================================================================
# 6. MAIN INGESTION GENERATOR  (stream progress back to caller)
# ======================================================================

def ingest_documents(force_fresh: bool = False) -> Generator[str, None, None]:
    """
    Scan RESOURCE_DIR, load supported files, process through the gold-standard
    pipeline, and upsert into the vector store.  Yields progress strings.
    """
    if force_fresh:
        yield "Fresh start requested. Clearing document tracking data...\n"
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)

    yield "Scanning documents...\n"
    update_status("running", message="Scanning files...")

    resource_dir = Config.RESOURCE_DIR
    if not os.path.exists(resource_dir):
        msg = f"Resource directory {resource_dir} does not exist."
        update_status("error", message=msg)
        yield f"ERROR: {msg}\n"
        return

    extensions = ["*.pdf", "*.docx", "*.txt", "*.md", "*.csv", "*.xlsx", "*.xls"]
    all_files: list[str] = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(resource_dir, ext)))

    if not all_files:
        update_status("completed", message="No documents found.")
        yield "WARNING: No documents found.\n"
        return

    tracking_data = load_tracking_data()
    new_documents: List[Document] = []
    processed_meta: list[tuple[str, str]] = []

    yield f"Found {len(all_files)} documents. Checking for updates...\n"

    store = VectorStoreFactory.get_instance()

    for file_path in all_files:
        try:
            stats = os.stat(file_path)
            file_id = f"{file_path}_{stats.st_size}_{stats.st_mtime}"

            if not force_fresh and file_path in tracking_data and tracking_data[file_path] == file_id:
                continue

            yield f"Reading: {os.path.basename(file_path)}\n"

            try:
                store.delete_source(file_path)
            except Exception:
                pass

            loader = get_loader(file_path)
            if loader:
                docs = loader.load()
                # Resolve URL from pages table if possible
                resolved_source = get_url_from_pages(file_path)
                for d in docs:
                    d.metadata["source"] = resolved_source
                
                new_documents.extend(docs)
                processed_meta.append((file_path, file_id))
            else:
                yield f"Skipping unsupported: {os.path.basename(file_path)}\n"
        except Exception as e:
            yield f"FAILED to load {os.path.basename(file_path)}: {e}\n"

    if not new_documents:
        yield "All documents are already up to date.\n"
        update_status("completed", message="All documents are up to date.")
        return

    yield f"Processing {len(new_documents)} items through gold-standard pipeline...\n"
    update_status("running", message="Splitting & enriching documents...")

    chunks = process_documents(new_documents)
    total_chunks = len(chunks)
    yield f"Generated {total_chunks} enriched chunks. Starting vectorization...\n"

    if force_fresh:
        yield "Wiping existing vector store for fresh build...\n"
        store.clear_all()

    batch_size = 100
    total_batches = (total_chunks + batch_size - 1) // batch_size
    start_time = time.time()

    try:
        for i in range(0, total_chunks, batch_size):
            batch_num = i // batch_size + 1
            batch_msg = f"Ingesting batch {batch_num}/{total_batches} (Chunks {i+1}-{min(i+batch_size, total_chunks)})"
            update_status("running", current=batch_num, total=total_batches, message=batch_msg, start_time=start_time)
            yield f"{batch_msg}\n"

            batch = chunks[i : i + batch_size]
            for attempt in range(2):
                try:
                    store.add_documents(batch)
                    break
                except Exception as e:
                    if attempt == 1:
                        yield f"ERROR batch {batch_num}: {e}\n"
                        print(f"CRITICAL: {traceback.format_exc()}")
                    time.sleep(1)
            gc.collect()

        for fp, fid in processed_meta:
            tracking_data[fp] = fid
        save_tracking_data(tracking_data)

        msg = f"SUCCESS: Ingested {total_chunks} chunks from {len(processed_meta)} documents."
        update_status("completed", message=msg)
        yield f"{msg}\n"

    except Exception as e:
        yield f"FATAL ERROR: {e}\n"
        print(f"FATAL: {traceback.format_exc()}")
