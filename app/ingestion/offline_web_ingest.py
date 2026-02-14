import os
import json
import time
import glob
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.factory import VectorStoreFactory
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

def extract_slider_text(soup):
    """Extract text from common slider/carousel elements."""
    slider_texts = []
    # Common selectors for sliders and carousels
    selectors = [
        '.slider', '.carousel', '.swiper', '.slide', '.gallery',
        '[class*="slider"]', '[class*="carousel"]', '[class*="swiper"]', '[class*="slide"]'
    ]
    for selector in selectors:
        for element in soup.select(selector):
            # Extract text and avoid duplicates if nested
            text = element.get_text(separator=' ', strip=True)
            if text and len(text) > 20: # Filter out very short text
                slider_texts.append(text)
    
    return "\n\n".join(list(dict.fromkeys(slider_texts))) # Order-preserving deduplication

def init_tracking_db():
    """Initialize the tracking tables in the database."""
    conn = sqlite3.connect(METADATA_DB, timeout=30, check_same_thread=False)
    cursor = conn.cursor()
    
    # Enable Write-Ahead Logging for better concurrency
    cursor.execute('PRAGMA journal_mode=WAL')
    
    # Table for tracking ingested files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingested_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_url TEXT UNIQUE,
            filename TEXT,
            chunks INTEGER,
            timestamp REAL,
            last_updated REAL,
            ingest_status INTEGER DEFAULT 1
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
    
    # --- Permission & Migration ---
    try:
        # 1. Migration: Add ingest_status column if missing
        cursor.execute("PRAGMA table_info(ingested_files)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'ingest_status' not in columns:
            cursor.execute('ALTER TABLE ingested_files ADD COLUMN ingest_status INTEGER DEFAULT 1')
            print("Successfully added 'ingest_status' column.")

        # 2. Permission check: Try a dummy write
        cursor.execute("CREATE TABLE IF NOT EXISTS _write_test (id INTEGER PRIMARY KEY)")
        cursor.execute("DROP TABLE _write_test")
        
        conn.commit()
    except sqlite3.OperationalError as e:
        if "readonly" in str(e).lower():
            print(f"\nCRITICAL ERROR: The database '{METADATA_DB}' is READ-ONLY.")
            print(f"Please run: sudo chown -R $USER:$USER '{os.path.dirname(os.path.abspath(METADATA_DB))}'")
            print("Or check if another process has an exclusive lock.\n")
            raise PermissionError(f"Database {METADATA_DB} is read-only.") from e
        raise e
    except Exception as e:
        print(f"Init DB error: {e}")
        
    conn.close()

def get_ingested_files():
    """Get all ingested files from the database that have chunks and are marked as ingested."""
    conn = sqlite3.connect(METADATA_DB, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT source_url, filename, chunks, timestamp 
        FROM ingested_files 
        WHERE chunks > 0 AND ingest_status = 1
        ORDER BY last_updated DESC
    ''')
    results = cursor.fetchall()
    conn.close()
    return {row[0]: {"filename": row[1], "chunks": row[2], "timestamp": row[3]} for row in results}

def save_ingested_file(source_url, filename, chunks, conn=None):
    """Save or update an ingested file in the database."""
    local_conn = False
    if conn is None:
        conn = sqlite3.connect(METADATA_DB, timeout=30, check_same_thread=False)
        local_conn = True
        
    cursor = conn.cursor()
    import time
    timestamp = time.time()
    cursor.execute('''
        INSERT OR REPLACE INTO ingested_files (source_url, filename, chunks, timestamp, last_updated, ingest_status)
        VALUES (?, ?, ?, COALESCE((SELECT timestamp FROM ingested_files WHERE source_url = ?), ?), ?, 1)
    ''', (source_url, filename, chunks, source_url, timestamp, timestamp))
    conn.commit()
    
    if local_conn:
        conn.close()

def update_status(status, current=0, total=0, message="", start_time=None, conn=None):
    """Update the ingestion status in the database."""
    import time
    timestamp = time.time()
    eta_seconds = None
    
    if start_time and current > 0:
        elapsed = timestamp - start_time
        avg_time_per_batch = elapsed / current
        remaining_batches = total - current
        eta_seconds = int(remaining_batches * avg_time_per_batch)
    
    local_conn = False
    if conn is None:
        conn = sqlite3.connect(METADATA_DB, timeout=30, check_same_thread=False)
        local_conn = True
        
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE ingestion_status 
        SET status = ?, current_batch = ?, total_batches = ?, message = ?, timestamp = ?, eta_seconds = ?
        WHERE id = 1
    ''', (status, current, total, message, timestamp, eta_seconds))
    conn.commit()
    
    if local_conn:
        conn.close()

def get_ingestion_status():
    """Get the current ingestion status from the database."""
    conn = sqlite3.connect(METADATA_DB, check_same_thread=False)
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
        conn = sqlite3.connect(METADATA_DB, timeout=30, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
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

    store = VectorStoreFactory.get_instance()
    vectorstore = store.get_vectorstore()

    processed_count = 0
    total_files = len(all_files)
    start_time = time.time()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", "", "---", "##", "#"] 
    )

    import sqlite3
    conn = sqlite3.connect(METADATA_DB, timeout=30, check_same_thread=False)
    # Enable WAL mode for the reader connection as well
    conn.execute('PRAGMA journal_mode=WAL')
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
                    
                    # Add Slider Content if found
                    slider_text = extract_slider_text(soup)
                    if slider_text:
                        readme_content += "## Slider Content\n"
                        readme_content += slider_text + "\n\n---\n\n"
                    
                    readme_content += text_content
                    docs = [Document(page_content=readme_content, metadata={"source": source_url})]

            # --- PDF Processing ---
            elif ext == ".pdf":
                 try:
                    md_content = pymupdf4llm.to_markdown(file_path)
                    docs = [Document(page_content=md_content, metadata={"source": source_url})]
                 except Exception as e:
                     yield f"  -> PDF Error: {e}\n"

            # Skip other file types as per requirement
            else:
                yield f"  -> Skipped: {ext} (only HTML and PDF allowed)\n"
                continue

            # --- Vectorization ---
            if docs:
                chunks = text_splitter.split_documents(docs)
                if chunks:
                    # Clear old chunks for this source
                    store.delete_source(source_url)
                    
                    if vectorstore:
                        vectorstore.add_documents(chunks)
                    else:
                        vectorstore = store.add_documents(chunks)
                    
                    # Track this file in database - reuse connection
                    save_ingested_file(source_url, filename, len(chunks), conn=conn)
                    
                    processed_count += 1
            
            if (i+1) % 10 == 0:
                update_status("running", current=i+1, total=total_files, message=f"Processed {i+1} files", start_time=start_time, conn=conn)
                # Checkpoint: ChromaDB is persistent

        except Exception as e:
            error_msg = str(e)
            if "readonly" in error_msg.lower() or "1032" in error_msg:
                db_dir = os.path.dirname(os.path.abspath(METADATA_DB))
                vs_dir = os.path.abspath(Config.VECTOR_DB_PATH)
                yield f"  -> CRITICAL PERMISSION ERROR: The database is read-only.\n"
                yield f"  -> Fix by running: sudo chown -R $USER:$USER {db_dir} {vs_dir}\n"
            yield f"  -> Error processing {file_path}: {e}\n"
            continue

    conn.close()

    # Final Save handled by Chroma persistence
    if vectorstore:
        yield "Data persisted in ChromaDB.\n"

    msg = f"SUCCESS: Ingested {processed_count} files from downloads."
    update_status("completed", message=msg, conn=conn)
    yield f"{msg}\n"

def reset_crawled_ingestion_status(full_wipe: bool = True):
    """
    Reset or wipe the ingested_files tracking table.
    - If full_wipe is True, it deletes all records (default for reset).
    - If full_wipe is False, it just sets status to 0 (legacy behavior).
    """
    conn = sqlite3.connect(METADATA_DB, timeout=30, check_same_thread=False)
    cursor = conn.cursor()
    
    if full_wipe:
        cursor.execute('DELETE FROM ingested_files')
        resources = [] # No specific resources to return if wiped
    else:
        # 1. Get resources and chunks count > 0
        cursor.execute('SELECT source_url FROM ingested_files WHERE chunks > 0')
        resources = [row[0] for row in cursor.fetchall()]
        
        # 2. Update ingest_status to false (0)
        if resources:
            cursor.execute('UPDATE ingested_files SET ingest_status = 0 WHERE chunks > 0')
    
    conn.commit()
    conn.close()
    return resources
