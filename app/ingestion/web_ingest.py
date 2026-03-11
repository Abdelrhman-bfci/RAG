import os
import json
import time
import requests
import hashlib
import mimetypes
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import trafilatura
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.factory import VectorStoreFactory
import pymupdf4llm
STATUS_FILE = "web_ingestion_status.json"
DOWNLOAD_FOLDER = Config.DOWNLOAD_FOLDER

import sqlite3

class WebMetadataStore:
    def __init__(self, db_path=Config.CRAWLER_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                filename TEXT,
                checksum TEXT,
                parent_url TEXT,
                chunk_count INTEGER DEFAULT 0,
                timestamp REAL DEFAULT 0
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checksum ON pages(checksum)')
        
        # Migrate existing schema if needed
        try:
            cursor.execute("PRAGMA table_info(pages)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'chunk_count' not in columns:
                cursor.execute('ALTER TABLE pages ADD COLUMN chunk_count INTEGER DEFAULT 0')
            if 'timestamp' not in columns:
                cursor.execute('ALTER TABLE pages ADD COLUMN timestamp REAL DEFAULT 0')
        except:
            pass
            
        conn.commit()
        conn.close()

    def get_page(self, url):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, checksum, parent_url, chunk_count, timestamp FROM pages WHERE url = ?", (url,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "filename": row[0],
                "checksum": row[1],
                "parent_url": row[2],
                "chunk_count": row[3],
                "timestamp": row[4]
            }
        return None

    def update_page(self, url, metadata):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO pages (url, filename, checksum, parent_url, chunk_count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                filename=excluded.filename,
                checksum=excluded.checksum,
                parent_url=excluded.parent_url,
                chunk_count=excluded.chunk_count,
                timestamp=excluded.timestamp
        ''', (
            url, 
            metadata.get("filename"), 
            metadata.get("checksum"), 
            metadata.get("parent_url"), 
            metadata.get("chunk_count", 0), 
            metadata.get("timestamp", 0)
        ))
        conn.commit()
        conn.close()

    def get_by_checksum(self, checksum):
        if not checksum: return None
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM pages WHERE checksum = ?", (checksum,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
        
    def clear_all(self):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM pages")
            conn.commit()
        except:
            pass
        finally:
            conn.close()

def calculate_checksum(content):
    """Calculates SHA256 hash of binary content."""
    return hashlib.sha256(content).hexdigest()

def get_extension(response):
    content_type = response.headers.get('content-type', '').split(';')[0]
    extension = mimetypes.guess_extension(content_type)
    if not extension:
        if "html" in content_type: return ".html"
        return ".dat"
    return extension

def update_status(status, current=0, total=0, message="", start_time=None):
    """Update the ingestion status file with ETA calculation."""
    data = {
        "status": status,
        "current_batch": current,
        "total_batches": total,
        "message": message,
        "timestamp": time.time()
    }
    
    if start_time and current > 0:
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / current
        remaining_batches = total - current
        eta_seconds = remaining_batches * avg_time_per_batch
        data["eta_seconds"] = int(eta_seconds)
    else:
        data["eta_seconds"] = None

    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)

def enrich_chunks_metadata(chunks, default_page=0):
    """Add chunk index and page to each chunk for citations and context expansion."""
    for i, doc in enumerate(chunks):
        doc.metadata["chunk"] = i
        if "page" not in doc.metadata or doc.metadata["page"] is None:
            doc.metadata["page"] = default_page
    return chunks

def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes, but keep pagination params."""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    if not path: path = '/'
    
    # Keep standard pagination/query params
    keep_params = ['page', 'p', 'offset', 'start', 'limit', 'pagination']
    from urllib.parse import parse_qsl, urlencode
    query_params = parse_qsl(parsed.query)
    filtered_params = [(k, v) for k, v in query_params if k in keep_params]
    
    new_query = urlencode(filtered_params)
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if new_query:
        normalized += f"?{new_query}"
    return normalized

def is_pagination_link(url: str, text: str) -> bool:
    """Heuristic to detect if a link is likely for pagination."""
    text = text.lower().strip()
    url = url.lower()
    
    # Text patterns
    text_patterns = ['next', 'previous', 'prev', 'more', 'page', '>', '»']
    if any(p == text for p in text_patterns) or text.isdigit():
        return True
    
    # URL patterns
    url_patterns = ['page=', 'p=', 'offset=', 'start=', '/p/']
    if any(p in url for p in url_patterns):
        return True
        
    return False

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

def ingest_websites(urls: list = None, force_fresh: bool = False, **kwargs):
    """
    Ingest website content using BFS crawling with pagination support.
    """
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    store = WebMetadataStore()
    
    if force_fresh:
        yield "Fresh start requested. Clearing metadata...\n"
        store.clear_all()
    store_vs = VectorStoreFactory.get_instance()
    vectorstore = store_vs.get_vectorstore()

    links = urls if urls else Config.WEBSITE_LINKS
    if not links:
        yield "No website links found.\n"
        update_status("completed", message="No links to ingest.")
        return

    max_depth = kwargs.get('depth', 8)
    max_pages = kwargs.get('max_pages', -1)
    
    yield f"Starting crawl on {len(links)} seed URLs (Max Depth: {max_depth})\n"
    update_status("running", message="Crawling websites...")

    processed_pages = 0
    start_time = time.time()

    for seed_url in links:
        # Tuple: (url, current_depth, parent_url, is_pagination)
        queue = deque([(seed_url, 0, None, False)])
        original_domain = urlparse(seed_url).netloc
        visited = set()

        while queue:
            if max_pages != -1 and processed_pages >= max_pages:
                break

            current_url, current_depth, parent_url, is_pagination = queue.popleft()
            
            # Don't increment depth if it's just pagination of the same results
            adj_depth = current_depth if is_pagination else current_depth
            
            if adj_depth > max_depth:
                continue
            
            if current_url in visited:
                continue
            visited.add(current_url)

            # Check if we already have this URL in metadata
            existing_meta = store.get_page(current_url)
            content = None
            content_type = ""
            
            should_download = True
            if not force_fresh and existing_meta:
                # Check for file on disk
                filename = existing_meta.get("filename")
                filepath = os.path.join(DOWNLOAD_FOLDER, filename) if filename else None
                if filepath and os.path.exists(filepath):
                    # Check timestamp (e.g. 24h)
                    if time.time() - existing_meta.get("timestamp", 0) < 86400:
                        yield f"[SKIP] Already indexed: {current_url}\n"
                        should_download = False
                        # Still need to extract links if it's HTML
                        if filename.endswith(".html"):
                            with open(filepath, "rb") as f:
                                content = f.read()
                            content_type = "text/html"
                
            if should_download:
                try:
                    yield f"[FETCH] Depth {adj_depth}: {current_url}\n"
                    response = requests.get(current_url, timeout=15)
                    if response.status_code != 200:
                        yield f"  -> Failed: Status {response.status_code}\n"
                        continue

                    content = response.content
                    content_type = response.headers.get("Content-Type", "").lower()
                    checksum = calculate_checksum(content)
                    
                    # Deduplication check
                    existing_filename = store.get_by_checksum(checksum)
                    if existing_filename:
                        yield f"  -> Identical content already exists. Reusing {existing_filename}\n"
                        filename = existing_filename
                    else:
                        ext = get_extension(response)
                        
                        # Restriction: Only HTML and allowed binary extensions from .env
                        is_allowed_binary = 'application/pdf' in content_type or ext in Config.ALLOWED_EXTENSIONS
                        is_html = 'text/html' in content_type or ext == '.html'
                        
                        if not (is_allowed_binary or is_html):
                            yield f"  -> Skipped: Type {content_type} / {ext} not allowed\n"
                            continue

                        filename = f"{hashlib.md5(current_url.encode()).hexdigest()}{ext}"
                        filepath = os.path.join(DOWNLOAD_FOLDER, filename)
                        with open(filepath, "wb") as f:
                            f.write(content)

                    # Update metadata (initial)
                    store.update_page(current_url, {
                        "filename": filename,
                        "checksum": checksum,
                        "parent_url": parent_url,
                        "timestamp": time.time(),
                        "chunk_count": 0
                    })

                    # Process content for vector store
                    if 'text/html' in content_type:
                        soup = BeautifulSoup(content, 'html.parser')
                        title = soup.title.string if soup.title else "Untitled Page"
                        
                        # Extract with trafilatura (Markdown support)
                        text_content = trafilatura.extract(content, output_format='markdown', include_tables=True)
                        if not text_content:
                            text_content = soup.get_text(separator='\n\n')
                        
                        if text_content and len(text_content.strip()) > 200:
                            # Enhance Markdown as a README format
                            readme_content = f"# {title}\n\n"
                            readme_content += f"**Source URL:** {current_url}\n"
                            readme_content += f"**Parent URL:** {parent_url if parent_url else 'Root'}\n"
                            readme_content += f"**Ingestion Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            readme_content += "---\n\n"
                            
                            # Add Slider Content if found
                            slider_text = extract_slider_text(soup)
                            if slider_text:
                                readme_content += "## Slider Content\n"
                                readme_content += slider_text + "\n\n---\n\n"
                            
                            readme_content += text_content
                            
                            doc = Document(page_content=readme_content, metadata={"source": current_url})
                            
                            # Split
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=Config.CHUNK_SIZE,
                                chunk_overlap=Config.CHUNK_OVERLAP,
                                separators=["\n\n", "\n", " ", ""]
                            )
                            chunks = text_splitter.split_documents([doc])
                            chunks = enrich_chunks_metadata(chunks, default_page=0)
                            
                            yield f"  -> Indexing {len(chunks)} chunks...\n"
                            
                            # Clean old chunks first
                            store_vs.delete_source(current_url)
                            
                            if vectorstore:
                                vectorstore.add_documents(chunks)
                            else:
                                vectorstore = store_vs.add_documents(chunks)
                            
                            # Update chunk count
                            meta = store.get_page(current_url)
                            meta["chunk_count"] = len(chunks)
                            store.update_page(current_url, meta)
                            
                            # Save periodic checkpoint (FAISS only)
                            if processed_pages % 5 == 0 and Config.VECTOR_STORE_PROVIDER.lower() == "faiss" and hasattr(store_vs, "save_index"):
                                store_vs.save_index(vectorstore)

                    elif 'application/pdf' in content_type or filename.endswith('.pdf'):
                        yield f"  -> Processing PDF with pymupdf4llm...\n"
                        try:
                            filepath = os.path.join(DOWNLOAD_FOLDER, filename)
                            md_content = pymupdf4llm.to_markdown(filepath)
                            
                            if md_content:
                                doc = Document(page_content=md_content, metadata={"source": current_url})
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=Config.CHUNK_SIZE,
                                    chunk_overlap=Config.CHUNK_OVERLAP,
                                    separators=["\n\n", "\n", " ", ""]
                                )
                                chunks = text_splitter.split_documents([doc])
                                chunks = enrich_chunks_metadata(chunks, default_page=0)
                                
                                yield f"  -> Indexing {len(chunks)} PDF chunks...\n"
                                store_vs.delete_source(current_url)
                                
                                if vectorstore:
                                    vectorstore.add_documents(chunks)
                                else:
                                    vectorstore = store_vs.add_documents(chunks)
                                
                                # Update chunk count
                                meta = store.get_page(current_url)
                                meta["chunk_count"] = len(chunks)
                                store.update_page(current_url, meta)
                        except Exception as pdf_err:
                            yield f"  -> PDF Processing Error: {pdf_err}\n"

                    processed_pages += 1
                    update_status("running", current=processed_pages, message=f"Processed {processed_pages} pages")
                    time.sleep(0.5)

                except Exception as e:
                    yield f"  -> Error: {str(e)}\n"
                    continue

            # Link Extraction (for both resumed and downloaded)
            if content and 'text/html' in content_type:
                try:
                    soup = BeautifulSoup(content, "html.parser")
                    for link in soup.find_all("a", href=True):
                        link_text = link.get_text()
                        full_url = urljoin(current_url, link.get("href"))
                        parsed = urlparse(full_url)
                        
                        # Normalize to avoid duplicates
                        clean_url = normalize_url(full_url)
                        
                        if (clean_url not in visited and 
                            parsed.netloc == original_domain and 
                            parsed.scheme in ['http', 'https']):
                            
                            # Detection for pagination
                            is_pag = is_pagination_link(clean_url, link_text)
                            
                            # Pagination links stay at the same depth level
                            next_depth = current_depth if is_pag else current_depth + 1
                            queue.append((clean_url, next_depth, current_url, is_pag))
                except Exception as e:
                    yield f"  -> Link extraction error: {e}\n"

    # Final Save (FAISS: persist index; Chroma: already persistent)
    if vectorstore and Config.VECTOR_STORE_PROVIDER.lower() == "faiss" and hasattr(store_vs, "save_index"):
        store_vs.save_index(vectorstore)
    
    msg = f"SUCCESS: Ingested {processed_pages} pages."
    update_status("completed", message=msg)
    yield f"{msg}\n"
