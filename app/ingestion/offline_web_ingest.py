import os
import time
import sqlite3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.factory import VectorStoreFactory
import trafilatura
from bs4 import BeautifulSoup
import pymupdf4llm
from langchain_community.document_loaders import (
    TextLoader,
)
from langchain_community.document_transformers import Html2TextTransformer

METADATA_DB = Config.CRAWLER_DB
DOWNLOAD_FOLDER = Config.DOWNLOAD_FOLDER

# File extensions supported by offline ingest
OFFLINE_ALLOWED_EXTENSIONS = {".html", ".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls"}
MIN_HTML_CONTENT_LENGTH = 50
MIN_TEXT_CONTENT_LENGTH = 10

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
    
    return "\n\n".join(list(dict.fromkeys(slider_texts)))  # Order-preserving deduplication


def table_exists(cursor, table_name: str) -> bool:
    """Check if a table exists in the current database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def get_source_url_for_file(cursor, file_path: str, filename: str) -> str:
    """Resolve source URL from crawler DB pages table, or use file:// path."""
    if table_exists(cursor, "pages"):
        try:
            cursor.execute("SELECT url FROM pages WHERE filename = ?", (filename,))
            result = cursor.fetchone()
            if result:
                return result[0]
        except sqlite3.OperationalError:
            pass
    return f"file://{file_path}"


def init_tracking_db():
    """Initialize the tracking tables in the database."""
    conn = sqlite3.connect(METADATA_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
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
        WHERE ingest_status = 1
        ORDER BY last_updated DESC
    ''')
    results = cursor.fetchall()
    conn.close()
    return {row[0]: {"filename": row[1], "chunks": row[2], "timestamp": row[3]} for row in results}

def save_ingested_file(source_url, filename, chunks, conn=None):
    """Save or update an ingested file in the database."""
    local_conn = False
    if conn is None:
        conn = sqlite3.connect(METADATA_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
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

def save_ingested_files_bulk(file_data_list, conn=None):
    """Save or update multiple ingested files in the database at once."""
    local_conn = False
    if conn is None:
        conn = sqlite3.connect(METADATA_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        local_conn = True
        
    cursor = conn.cursor()
    import time
    timestamp = time.time()
    
    # We use a transaction for bulk updates
    try:
        cursor.execute("BEGIN TRANSACTION")
        for source_url, filename, chunks in file_data_list:
            cursor.execute('''
                INSERT OR REPLACE INTO ingested_files (source_url, filename, chunks, timestamp, last_updated, ingest_status)
                VALUES (?, ?, ?, COALESCE((SELECT timestamp FROM ingested_files WHERE source_url = ?), ?), ?, 1)
            ''', (source_url, filename, chunks, source_url, timestamp, timestamp))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
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
        conn = sqlite3.connect(METADATA_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
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
    try:
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
    except Exception as e:
        print(f"Error fetching ingestion status: {e}")
        
    return {"status": "idle", "current_batch": 0, "total_batches": 0, "message": "", "timestamp": 0, "eta_seconds": None}

def get_loader(file_path: str):
    """Returns the appropriate LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return None  # Handled separately with pymupdf4llm
    if ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path)
    return None


def load_text_file_with_encoding_fallback(file_path: str, source_url: str):
    """Load .txt/.md with encoding fallback; returns list of Document or empty."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            loader = TextLoader(file_path, encoding=encoding)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = source_url
            return docs
        except (UnicodeDecodeError, OSError):
            continue
    return []


def enrich_chunks_metadata(chunks, default_page=0):
    """Add chunk index and page to each chunk for citations and context expansion."""
    for i, doc in enumerate(chunks):
        doc.metadata["chunk"] = i
        if "page" not in doc.metadata or doc.metadata["page"] is None:
            doc.metadata["page"] = default_page
    return chunks

def ingest_offline_downloads(force_fresh: bool = False, silent: bool = False):
    """
    Ingest files from the DOWNLOAD_FOLDER.
    """
    def _log(msg):
        if not silent:
            return msg
        return None

    if not os.path.exists(DOWNLOAD_FOLDER):
        msg = f"Error: Download folder {DOWNLOAD_FOLDER} does not exist.\n"
        if not silent: yield msg
        return

    # Initialize tracking database
    init_tracking_db()
    
    update_status("running", message="Scanning downloaded files...")
    msg = _log("Scanning downloaded files...\n")
    if msg: yield msg

    if force_fresh:
        # Clear all ingested files if fresh start requested
        conn = sqlite3.connect(METADATA_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('DELETE FROM ingested_files')
        conn.commit()
        conn.close()
        msg = _log("Fresh start: Cleared previous tracking data.\n")
        if msg: yield msg

    # Gather all files (only allowed extensions)
    all_files = []
    for root, dirs, files in os.walk(DOWNLOAD_FOLDER):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in OFFLINE_ALLOWED_EXTENSIONS:
                all_files.append(os.path.join(root, file))

    if not all_files:
        msg = "No supported files found in downloads folder."
        update_status("completed", message=msg)
        msg_log = _log(f"{msg}\n")
        if msg_log: yield msg_log
        return

    msg = _log(f"Found {len(all_files)} supported files. Starting processing...\n")
    if msg: yield msg

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

    conn = sqlite3.connect(METADATA_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
    # Enable WAL mode
    conn.execute('PRAGMA journal_mode=WAL')
    cursor = conn.cursor()

    # Get already ingested files to skip
    ingested_map = {}
    if not force_fresh:
        ingested_map = get_ingested_files()
        msg = _log(f"Found {len(ingested_map)} already ingested files. These will be skipped.\n")
        if msg: yield msg

    bulk_data_buffer = []


    for i, file_path in enumerate(all_files):
        try:
            filename = os.path.basename(file_path)
            source_url = get_source_url_for_file(cursor, file_path, filename)

            # Skip Logic
            if not force_fresh and source_url in ingested_map:
                msg = _log(f"[{i+1}/{total_files}] Skipping (already ingested): {filename}\n")
                if msg: yield msg
                continue
            
            msg = _log(f"[{i+1}/{total_files}] Processing: {filename}\n")
            if msg: yield msg
            
            docs = []
            ext = os.path.splitext(file_path)[1].lower()
            
            # --- HTML Processing (Matched with chatbot/loaders/html_loader.py) ---
            if ext == ".html":
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    raw_docs = loader.load()
                    
                    filtered_docs = []
                    for doc in raw_docs:
                        soup = BeautifulSoup(doc.page_content, 'html.parser')
                        
                        # 1. Extract Global Context
                        browser_title = soup.title.get_text(strip=True) if soup.title else ""
                        
                        page_header = ""
                        header_div = soup.find(class_="page-title-heading")
                        if header_div:
                            page_header = header_div.get_text(strip=True)
                            
                        breadcrumbs = ""
                        bread_div = soup.find(class_="breadcrumb-wrapper")
                        if bread_div:
                            crumbs = [t.get_text(strip=True) for t in bread_div.find_all('a')] + \
                                     [t.get_text(strip=True) for t in bread_div.find_all('span', class_='current')]
                            breadcrumbs = " > ".join(crumbs)

                        context_header = (
                            f"### Document Context\n"
                            f"**Source Page:** {browser_title}\n"
                            f"**Section/Department:** {page_header}\n"
                            f"**Path:** {breadcrumbs}\n"
                            f"---\n"
                        )

                        # 2. Isolate Main Content
                        target_tag = soup.find(attrs={"id": "site-main"}) or soup.find(class_="site-main") or soup.find(class_="main-content")

                        if target_tag:
                            doc.page_content = str(target_tag)
                            doc.metadata["context_header"] = context_header
                            doc.metadata["page_title"] = page_header or browser_title
                            filtered_docs.append(doc)
                        else:
                            # Fallback to trafilatura if specific tags aren't found
                            text_content = trafilatura.extract(doc.page_content, output_format='markdown', include_tables=True)
                            if text_content:
                                doc.page_content = text_content
                                doc.metadata["context_header"] = context_header
                                filtered_docs.append(doc)

                    if filtered_docs:
                        # 3. Convert to Markdown
                        html2text = Html2TextTransformer()
                        transformed_docs = html2text.transform_documents(filtered_docs)
                        
                        # 4. Final Metadata Cleanup & Grounding
                        for d in transformed_docs:
                            # Prepend the context header for better LLM grounding
                            grounded_content = d.metadata.get("context_header", "") + d.page_content
                            d.page_content = grounded_content
                            d.metadata["source"] = source_url
                            d.metadata["page"] = "Web"
                            d.metadata["title"] = d.metadata.get("page_title", filename)
                            docs.append(d)
                except Exception as e:
                    msg = _log(f"  -> HTML Loader Error: {e}. Falling back to basic...\n")
                    if msg: yield msg

            # --- PDF Processing ---
            elif ext == ".pdf":
                 try:
                    # Primary: Convert to Markdown for better RAG
                    md_content = pymupdf4llm.to_markdown(file_path)
                    docs = [Document(page_content=md_content, metadata={"source": source_url})]
                 except Exception as e:
                     msg = _log(f"  -> pymupdf4llm Error: {e}. Falling back to standard loader...\n")
                     if msg: yield msg
                     try:
                         loader = PyMuPDFLoader(file_path)
                         fallback_docs = loader.load()
                         for d in fallback_docs:
                             d.metadata["source"] = source_url
                         docs = fallback_docs
                     except Exception as e_fallback:
                         msg = _log(f"  -> CRITICAL PDF Error: {e_fallback}\n")
                         if msg: yield msg

            # --- DOCX, TXT, MD, CSV, XLSX (offline documents) ---
            else:
                if ext in [".txt", ".md"]:
                    docs = load_text_file_with_encoding_fallback(file_path, source_url)
                    if not docs:
                        msg = _log(f"  -> Skipped: could not decode {filename}\n")
                        if msg: yield msg
                        continue
                else:
                    loader = get_loader(file_path)
                    if loader:
                        try:
                            loaded = loader.load()
                            for d in loaded:
                                d.metadata["source"] = source_url
                            docs = loaded
                        except Exception as e_loader:
                            msg = _log(f"  -> Loader error for {ext}: {e_loader}\n")
                            if msg: yield msg
                            continue
                    else:
                        msg = _log(f"  -> Skipped: {ext} (unsupported)\n")
                        if msg: yield msg
                        continue

            # --- Vectorization ---
            if docs:
                chunks = text_splitter.split_documents(docs)
                chunks = enrich_chunks_metadata(chunks, default_page=0)
                if chunks:
                    # Skip empty or too-short single-chunk content
                    if len(chunks) == 1 and len(chunks[0].page_content.strip()) < MIN_TEXT_CONTENT_LENGTH:
                        msg = _log(f"  -> Skipped: content too short ({filename})\n")
                        if msg: yield msg
                    else:
                        store.delete_source(source_url)
                        if vectorstore:
                            vectorstore.add_documents(chunks)
                        else:
                            vectorstore = store.add_documents(chunks)
                        bulk_data_buffer.append((source_url, filename, len(chunks)))
                        processed_count += 1
                        msg = _log(f"  -> Indexed {len(chunks)} chunks.\n")
                        if msg: yield msg
            
            # Periodically commit status, file tracking, and FAISS checkpoint
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                if bulk_data_buffer:
                    save_ingested_files_bulk(bulk_data_buffer, conn=conn)
                    bulk_data_buffer = []
                update_status(
                    "running",
                    current=i + 1,
                    total=total_files,
                    message=f"Processed {i + 1}/{total_files} files",
                    start_time=start_time,
                    conn=conn,
                )
                if (
                    vectorstore
                    and getattr(Config, "VECTOR_STORE_PROVIDER", "chroma").lower() == "faiss"
                    and hasattr(store, "save_index")
                ):
                    try:
                        store.save_index(vectorstore)
                    except Exception:
                        pass

        except Exception as e:
            error_msg = str(e)
            if "readonly" in error_msg.lower() or "1032" in error_msg:
                db_dir = os.path.dirname(os.path.abspath(METADATA_DB))
                vs_dir = os.path.abspath(Config.VECTOR_DB_PATH)
                msg1 = _log(f"  -> CRITICAL PERMISSION ERROR: The database is read-only.\n")
                msg2 = _log(f"  -> Fix by running: sudo chown -R $USER:$USER {db_dir} {vs_dir}\n")
                if msg1: yield msg1
                if msg2: yield msg2
            msg = _log(f"  -> Error processing {file_path}: {e}\n")
            if msg: yield msg
            continue

    # Final persistence (Chroma is auto-persisted; FAISS needs explicit save)
    if vectorstore and getattr(Config, "VECTOR_STORE_PROVIDER", "chroma").lower() == "faiss" and hasattr(store, "save_index"):
        try:
            store.save_index(vectorstore)
        except Exception:
            pass

    msg = f"SUCCESS: Ingested {processed_count} files from downloads."
    update_status("completed", message=msg, conn=conn)
    conn.close()
    msg_log = _log(f"{msg}\n")
    if msg_log: yield msg_log

def reset_crawled_ingestion_status(full_wipe: bool = True):
    """
    Reset or wipe the ingested_files tracking table.
    - If full_wipe is True, it deletes all records (default for reset).
    - If full_wipe is False, it just sets status to 0 (legacy behavior).
    """
    conn = sqlite3.connect(METADATA_DB, timeout=Config.DB_TIMEOUT, check_same_thread=False)
    cursor = conn.cursor()
    
    if full_wipe:
        cursor.execute('DELETE FROM ingested_files')
        cursor.execute('DELETE FROM ingestion_status')
        # Re-initialize default status
        cursor.execute('''
            INSERT OR IGNORE INTO ingestion_status (id, status, current_batch, total_batches, message, timestamp, eta_seconds)
            VALUES (1, 'idle', 0, 0, '', 0, NULL)
        ''')
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
