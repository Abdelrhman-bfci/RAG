import os
import json
import time
import glob
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore
import trafilatura
from bs4 import BeautifulSoup
import pymupdf4llm
from langchain_community.document_loaders import (
    Docx2txtLoader, 
    TextLoader, 
    CSVLoader, 
    UnstructuredExcelLoader
)
import sqlite3

METADATA_DB = Config.CRAWLER_DB
DOWNLOAD_FOLDER = Config.DOWNLOAD_FOLDER

def init_tracking_db():
    """Initialize the tracking tables in the database."""
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    
    # Table for tracking ingested files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingested_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url TEXT UNIQUE,
            filename TEXT,
            chunks INTEGER,
            timestamp REAL,
            last_updated REAL
        )
    ''')
    
    # Table for tracking ingestion status
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingestion_status (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            status TEXT,
            current_batch INTEGER,
            total_batches INTEGER,
            message TEXT,
            timestamp REAL,
            eta_seconds INTEGER
        )
    ''')
    
    # Insert default status row if it doesn't exist
    cursor.execute('''
        INSERT OR IGNORE INTO ingestion_status (id, status, current_batch, total_batches, message, timestamp, eta_seconds)
        VALUES (1, 'idle', 0, 0, '', 0, NULL)
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_url ON ingested_files(source_url)')
    conn.commit()
    conn.close()

def get_ingested_files():
    """Get all ingested files from the database."""
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    cursor.execute('SELECT source_url, filename, chunks, timestamp FROM ingested_files ORDER BY last_updated DESC')
    results = cursor.fetchall()
    conn.close()
    return {row[0]: {"filename": row[1], "chunks": row[2], "timestamp": row[3]} for row in results}

def save_ingested_file(source_url, filename, chunks):
    """Save or update an ingested file in the database."""
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    import time
    timestamp = time.time()
    cursor.execute('''
        INSERT OR REPLACE INTO ingested_files (source_url, filename, chunks, timestamp, last_updated)
        VALUES (?, ?, ?, COALESCE((SELECT timestamp FROM ingested_files WHERE source_url = ?), ?), ?)
    ''', (source_url, filename, chunks, source_url, timestamp, timestamp))
    conn.commit()
    conn.close()

def update_status(status, current=0, total=0, message="", start_time=None):
    """Update the ingestion status in the database."""
    import time
    timestamp = time.time()
    eta_seconds = None
    
    if start_time and current > 0:
        elapsed = timestamp - start_time
        avg_time_per_batch = elapsed / current
        remaining_batches = total - current
        eta_seconds = int(remaining_batches * avg_time_per_batch)
    
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE ingestion_status 
        SET status = ?, current_batch = ?, total_batches = ?, message = ?, timestamp = ?, eta_seconds = ?
        WHERE id = 1
    ''', (status, current, total, message, timestamp, eta_seconds))
    conn.commit()
    conn.close()

def get_ingestion_status():
    """Get the current ingestion status from the database."""
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    cursor.execute('SELECT status, current_batch, total_batches, message, timestamp, eta_seconds FROM ingestion_status WHERE id = 1')
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "status": result[0],
            "current_batch": result[1],
            "total_batches": result[2],
            "message": result[3],
            "timestamp": result[4],
            "eta_seconds": result[5]
        }
    return {"status": "idle", "current_batch": 0, "total_batches": 0, "message": "", "timestamp": 0, "eta_seconds": None}

def get_loader(file_path: str):
    """Returns the appropriate LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext in [".xlsx", ".xls"]:
         return UnstructuredExcelLoader(file_path)
    return None

def ingest_offline_downloads(force_fresh: bool = False):
    """
    Ingest files from the DOWNLOAD_FOLDER.
    """
    if not os.path.exists(DOWNLOAD_FOLDER):
        yield f"Error: Download folder {DOWNLOAD_FOLDER} does not exist.\n"
        return

    # Initialize tracking database
    init_tracking_db()
    
    update_status("running", message="Scanning downloaded files...")
    yield "Scanning downloaded files...\n"

    if force_fresh:
        # Clear all ingested files if fresh start requested
        conn = sqlite3.connect(METADATA_DB)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM ingested_files')
        conn.commit()
        conn.close()
        yield "Fresh start: Cleared previous tracking data.\n"

    # Gather all files
    all_files = []
    for root, dirs, files in os.walk(DOWNLOAD_FOLDER):
        for file in files:
            all_files.append(os.path.join(root, file))

    if not all_files:
        msg = "No files found in downloads folder."
        update_status("completed", message=msg)
        yield f"{msg}\n"
        return

    yield f"Found {len(all_files)} files. Starting processing...\n"

    faiss_store = FAISSStore()
    # We load index once at start, or handle inside loop? 
    # For batch processing, loading once and adding all is better for performance.
    # However, if it's huge, we might need batching.
    vectorstore = faiss_store.load_index()

    processed_count = 0
    total_files = len(all_files)
    start_time = time.time()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", "", "---", "##", "#"] 
    )

    import sqlite3
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    for i, file_path in enumerate(all_files):
        try:
            filename = os.path.basename(file_path)
            # Try to get URL from DB to use as source
            cursor.execute("SELECT url FROM pages WHERE filename = ?", (filename,))
            result = cursor.fetchone()
            source_url = result[0] if result else f"file://{file_path}" # Fallback if not in DB

            # Check if this source is already indexed? (Simplified: we overwrite/update)
            # For efficiency, we could check DB or FAISS, but let's assume update request.
            
            yield f"[{i+1}/{total_files}] Processing: {filename}\n"
            
            docs = []
            ext = os.path.splitext(file_path)[1].lower()
            
            # --- HTML Processing ---
            if ext == ".html":
                with open(file_path, "rb") as f:
                    content = f.read()
                
                # HTML Parse Logic (Reusing web_ingest)
                soup = BeautifulSoup(content, 'html.parser')
                title = soup.title.string if soup.title else "Untitled Page"
                text_content = trafilatura.extract(content, output_format='markdown', include_tables=True)
                if not text_content:
                    text_content = soup.get_text(separator='\n\n')
                
                if text_content and len(text_content.strip()) > 50:
                    readme_content = f"# {title}\n\n"
                    readme_content += f"**Source:** {source_url}\n"
                    readme_content += f"**Ingestion Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    readme_content += "---\n\n"
                    readme_content += text_content
                    docs = [Document(page_content=readme_content, metadata={"source": source_url})]

            # --- PDF Processing ---
            elif ext == ".pdf":
                 try:
                    md_content = pymupdf4llm.to_markdown(file_path)
                    docs = [Document(page_content=md_content, metadata={"source": source_url})]
                 except Exception as e:
                     yield f"  -> PDF Error: {e}\n"

            # --- Other Docs ---
            else:
                loader = get_loader(file_path)
                if loader:
                    docs = loader.load()
                    # Update metadata source to be the URL if known, else file path
                    for doc in docs:
                        doc.metadata["source"] = source_url

            # --- Vectorization ---
            if docs:
                chunks = text_splitter.split_documents(docs)
                if chunks:
                    # Clear old chunks for this source
                    faiss_store.delete_source(source_url)
                    
                    if vectorstore:
                        vectorstore.add_documents(chunks)
                    else:
                        vectorstore = FAISSStore().add_documents(chunks)
                    
                    # Track this file in database
                    save_ingested_file(source_url, filename, len(chunks))
                    
                    processed_count += 1
            
            if (i+1) % 10 == 0:
                update_status("running", current=i+1, total=total_files, message=f"Processed {i+1} files", start_time=start_time)
                # Save checkpoint
                if vectorstore:
                    faiss_store.save_index(vectorstore)

        except Exception as e:
            yield f"  -> Error processing {file_path}: {e}\n"
            continue

    conn.close()

    # Final Save
    if vectorstore:
        try:
            faiss_store.save_index(vectorstore)
            yield "Index saved successfully.\n"
        except Exception as e:
            yield f"Error saving index: {e}\n"

    msg = f"SUCCESS: Ingested {processed_count} files from downloads."
    update_status("completed", message=msg)
    yield f"{msg}\n"
