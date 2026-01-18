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
from app.vectorstore.faiss_store import FAISSStore

TRACKING_FILE = "web_metadata.json"
STATUS_FILE = "web_ingestion_status.json"
DOWNLOAD_FOLDER = "downloads"

class WebMetadataStore:
    def __init__(self, file_path=TRACKING_FILE):
        self.file_path = file_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def get_page(self, url):
        return self.data.get(url)

    def update_page(self, url, metadata):
        # metadata: {filename, checksum, parent_url, chunk_count, timestamp}
        self.data[url] = metadata
        self.save()

    def get_by_checksum(self, checksum):
        for url, meta in self.data.items():
            if meta.get("checksum") == checksum:
                return meta.get("filename")
        return None

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

def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes."""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    if not path: path = '/'
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized

def ingest_websites(urls: list = None, force_fresh: bool = False, **kwargs):
    """
    Ingest website content using BFS crawling.
    """
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    if force_fresh:
        yield "Fresh start requested. Clearing metadata...\n"
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)

    store = WebMetadataStore()
    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index()

    links = urls if urls else Config.WEBSITE_LINKS
    if not links:
        yield "No website links found.\n"
        update_status("completed", message="No links to ingest.")
        return

    max_depth = kwargs.get('depth', 2)
    max_pages = kwargs.get('max_pages', -1)
    
    yield f"Starting crawl on {len(links)} seed URLs (Max Depth: {max_depth})\n"
    update_status("running", message="Crawling websites...")

    processed_pages = 0
    start_time = time.time()

    for seed_url in links:
        queue = deque([(seed_url, 0, None)])
        original_domain = urlparse(seed_url).netloc
        visited = set()

        while queue:
            if max_pages != -1 and processed_pages >= max_pages:
                break

            current_url, current_depth, parent_url = queue.popleft()
            
            if current_depth > max_depth:
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
                    yield f"[FETCH] Depth {current_depth}: {current_url}\n"
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
                        # Extract with trafilatura (Markdown support)
                        text_content = trafilatura.extract(content, output_format='markdown', include_tables=True)
                        if not text_content:
                            # Fallback to BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            text_content = soup.get_text(separator='\n\n')
                        
                        if text_content and len(text_content.strip()) > 200:
                            doc = Document(page_content=text_content, metadata={"source": current_url})
                            
                            # Split
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=Config.CHUNK_SIZE,
                                chunk_overlap=Config.CHUNK_OVERLAP,
                                separators=["\n\n", "\n", " ", ""]
                            )
                            chunks = text_splitter.split_documents([doc])
                            
                            yield f"  -> Indexing {len(chunks)} chunks...\n"
                            
                            # Clean old chunks from FAISS first
                            faiss_store.delete_source(current_url)
                            
                            if vectorstore:
                                vectorstore.add_documents(chunks)
                            else:
                                vectorstore = FAISSStore().add_documents(chunks) # This handles creation
                            
                            # Update chunk count
                            meta = store.get_page(current_url)
                            meta["chunk_count"] = len(chunks)
                            store.update_page(current_url, meta)
                            
                            # Save periodic checkpoint
                            if processed_pages % 5 == 0:
                                faiss_store.save_index(vectorstore)

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
                        full_url = urljoin(current_url, link.get("href"))
                        parsed = urlparse(full_url)
                        # Normalize to avoid duplicates
                        clean_url = normalize_url(full_url)
                        
                        if (clean_url not in visited and 
                            parsed.netloc == original_domain and 
                            parsed.scheme in ['http', 'https']):
                            
                            queue.append((clean_url, current_depth + 1, current_url))
                except Exception as e:
                    yield f"  -> Link extraction error: {e}\n"

    # Final Save
    if vectorstore:
        faiss_store.save_index(vectorstore)
    
    msg = f"SUCCESS: Ingested {processed_pages} pages."
    update_status("completed", message=msg)
    yield f"{msg}\n"
