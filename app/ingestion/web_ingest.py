import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore
import trafilatura
from langchain_core.documents import Document

TRACKING_FILE = "ingested_websites.json"
STATUS_FILE = "web_ingestion_status.json"

def load_tracking_data():
    """Load the tracking registry from disk."""
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_tracking_data(data):
    """Save the tracking registry to disk."""
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=4)

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
    """Strip fragments, query params (except pagination), and trailing slashes."""
    parsed = urlparse(url)
    # Keep standard pagination/query params if they seem relevant (e.g. page, p, offset)
    # Otherwise strip them to avoid duplicates
    path = parsed.path.rstrip('/')
    if not path: path = '/'
    
    # Simple list of query params to preserve for discovery (pagination)
    keep_params = ['page', 'p', 'offset', 'start', 'limit']
    from urllib.parse import parse_qsl, urlencode
    query_params = parse_qsl(parsed.query)
    filtered_params = [(k, v) for k, v in query_params if k in keep_params]
    
    new_query = urlencode(filtered_params)
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if new_query:
        normalized += f"?{new_query}"
    return normalized

def has_repeated_segments(url: str, threshold: int = 2) -> bool:
    """Detect if path segments repeat too many times to prevent recursion loops."""
    path = urlparse(url).path.strip('/')
    if not path: return False
    segments = path.split('/')
    from collections import Counter
    counts = Counter(segments)
    for seg, count in counts.items():
        if count > threshold and len(seg) > 2: # Ignore short things like /e/ or /v/
            return True
    return False

def ingest_websites(urls: list = None, force_fresh: bool = False, **kwargs):
    """
    Ingest website content from the given urls or Config.WEBSITE_LINKS.
    Only processes links that haven't been successfully indexed recently.
    Yields progress updates as strings.
    """
    if force_fresh:
        yield "Fresh start requested for websites. Clearing website tracking...\n"
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)

    yield "Checking website links...\n"
    update_status("running", message="Scanning links...")
    
    links = urls if urls else Config.WEBSITE_LINKS
    if not links:
        msg = "No website links found in configuration."
        update_status("completed", message=msg)
        yield "WARNING: No websites to ingest.\n"
        return

    tracking_data = load_tracking_data()
    new_documents = []
    processed_links = []
    
    # Configure crawling limits
    max_depth = kwargs.get('depth', 8)
    max_pages = kwargs.get('max_pages', -1) # -1 implies unlimited
    
    yield f"Found {len(links)} seed links. Starting crawl (Max Depth: {max_depth}, Max Pages: {'Unlimited' if max_pages == -1 else max_pages})...\n"

    for url in links:
        try:
            # Check if main URL was recently indexed (unless forced)
            last_indexed = tracking_data.get(url, 0)
            if not force_fresh and (time.time() - last_indexed < 86400): 
                yield f"Skipping (indexed recently): {url}\n"
                continue
            
            # BFS Initialization
            # Queue stores tuples: (url, current_depth)
            queue = [(url, 0)]
            visited = set([url])
            target_urls = []
            
            domain = urlparse(url).netloc
            base_url = url.rstrip('/')

            yield f"Starting crawl on: {url}\n"

            while queue:
                # Check for max_pages limit if not unlimited
                if max_pages != -1 and len(target_urls) >= max_pages:
                    break

                current_url, current_depth = queue.pop(0)
                
                if current_depth > max_depth:
                    continue
                
                # Check extensions first to save a request
                IGNORED_EXTENSIONS = (
                    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
                    '.mp4', '.mp3', '.wav', '.avi', '.mov', '.zip', '.tar', '.gz',
                    '.rar', '.7z', '.exe', '.dmg', '.iso', '.bin', '.doc', '.docx',
                    '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.xml', '.json', '.css', '.js'
                )
                
                path = urlparse(current_url).path.lower()
                if path.endswith(IGNORED_EXTENSIONS):
                    yield f"Skipping media/resource file: {current_url}\n"
                    continue

                try:
                    yield f"Scanning [Depth {current_depth}]: {current_url}\n"
                    
                    # Head request or stream=True to check headers first
                    try:
                        head_response = requests.head(current_url, timeout=5, allow_redirects=True)
                        content_type = head_response.headers.get('Content-Type', '').lower()
                        
                        if 'text/html' not in content_type:
                            yield f"Skipping non-HTML content ({content_type}): {current_url}\n"
                            continue
                            
                    except Exception:
                        # If HEAD fails, we might still try GET but typically it's risky if we want to avoid downloads.
                        # We'll proceed to GET with stream=True as a fallback check
                        pass

                    response = requests.get(current_url, timeout=10)
                    
                    # Double check content type on the actual response
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'text/html' not in content_type:
                        yield f"Skipping non-HTML content ({content_type}): {current_url}\n"
                        continue

                    # Validation passed, add to targets
                    target_urls.append(current_url)
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    found_on_page = 0
                    for a in soup.find_all('a', href=True):
                        if max_pages != -1 and len(target_urls) >= max_pages:
                            break
                            
                        href = a['href']
                        text = a.get_text().lower().strip()
                        full_url = urljoin(current_url, href)
                        
                        # Normalize and check for loops
                        full_url = normalize_url(full_url)
                        
                        # Prevent loops with repeated path segments
                        if has_repeated_segments(full_url):
                            continue

                        parsed_full = urlparse(full_url)
                        
                        if parsed_full.netloc == domain and full_url not in visited:
                            # Improved Heuristic filter
                            path = parsed_full.path.lower()
                            # Keywords for general discovery
                            discover_keywords = [
                                "about", "history", "vision", "mission", "leader", "academic", "program", 
                                "department", "news", "blog", "service", "product", "page", "p="
                            ]
                            
                            # Check if it looks like a pagination link or matches keywords
                            is_discovery_path = any(kw in path for kw in discover_keywords) or \
                                              any(kw in text for kw in discover_keywords) or \
                                              any(p in parsed_full.query for p in ['page', 'p', 'offset']) or \
                                              current_depth == 0
                            
                            if is_discovery_path:
                                visited.add(full_url)
                                queue.append((full_url, current_depth + 1))
                                found_on_page += 1
                    
                    # Yield progress occasionally
                    # yield f" - Found {found_on_page} new links on {current_url}\n"

                except Exception as e:
                    yield f"Failed to crawl {current_url}: {e}\n"

            
            yield f"Discovered {len(target_urls)} pages to process for {url}\n"

            # Batch process the discovered URLs
            if target_urls:
                yield "Starting extraction...\n"
                for i, target_url in enumerate(target_urls):
                    try:
                        yield f"Processing ({i+1}/{len(target_urls)}): {target_url}\n"
                        # Fetch with small retry
                        downloaded = None
                        for attempt in range(2):
                            try:
                                downloaded = trafilatura.fetch_url(target_url)
                                if downloaded: break
                                time.sleep(1)
                            except: pass
                            
                        if downloaded:
                            # Use trafilatura extraction with table support
                            result = trafilatura.extract(downloaded, output_format='markdown', include_tables=True, include_images=False, include_links=False)
                            
                            # Fallback to BeautifulSoup logic if Markdown extraction is empty or too short
                            if not result or len(result.strip()) < 200:
                                # Simple fallback to text-only if we need it
                                soup = BeautifulSoup(downloaded, 'html.parser')
                                result = soup.get_text(separator='\n\n')
                                
                            if result and len(result.strip()) > 300:
                                doc = Document(page_content=result, metadata={"source": target_url})
                                new_documents.append(doc)
                                yield f" - Successfully extracted\n"
                            else:
                                yield f" - Skipping (insufficient content)\n"
                        else:
                            yield f" - Failed to fetch URL\n"
                    except Exception as e:
                        yield f" - Extraction failed: {e}\n"
    
            processed_links.append(url)

        except Exception as e:
            yield f"FAILED to process hierarchy for {url}: {e}\n"

    if not new_documents:
        yield "No new website content to index.\n"
        update_status("completed", message="No new websites indexed.")
        return

    yield f"Splitting content into chunks...\n"
    update_status("running", message="Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", "", "---", "##", "#"] # Markdown-friendly separators
    )
    
    splitted_docs = text_splitter.split_documents(new_documents)
    total_chunks = len(splitted_docs)
    yield f"Generated {total_chunks} chunks. Starting vectorization...\n"
    
    # Clean up old versions of these pages to prevent duplicates
    try:
        # Extract unique source URLs from the new documents
        current_sources = list(set([d.metadata.get("source") for d in new_documents if d.metadata.get("source")]))
        if current_sources:
            yield f"Syncing index for {len(current_sources)} pages...\n"
            temp_store = FAISSStore()
            temp_store.delete_sources(current_sources)
    except Exception as e:
        yield f"WARNING: Failed to sync old index: {e}\n"

    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index() # Load once
    
    # PROCESS ONE BY ONE FOR STABILITY (copied pattern from pdf_ingest)
    start_time = time.time()
    
    import gc
    import traceback

    try:
        for i, doc in enumerate(splitted_docs):
            current_num = i + 1
            
            # 1. Add single document with simple retry
            max_retries = 2
            success = False
            for attempt in range(max_retries):
                try:
                    if vectorstore:
                        vectorstore.add_documents([doc])
                    else:
                        from langchain_community.vectorstores import FAISS
                        vectorstore = FAISS.from_documents([doc], faiss_store.embeddings)
                    success = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    yield f"ERROR processing chunk {current_num} after retries: {e}\n"
                    print(f"CRITICAL ERROR embedding chunk {current_num}: {traceback.format_exc()}")
            
            if not success:
                continue

            # 2. Yield progress occasionally
            if current_num % 10 == 0:
                msg = f"Ingesting chunk {current_num}/{total_chunks}"
                yield f"{msg}\n"
                update_status("running", current=current_num, total=total_chunks, message=msg, start_time=start_time)
                gc.collect()

            # 3. Checkpoint occasionally
            if current_num % 100 == 0:
                try:
                    faiss_store.save_index(vectorstore)
                except Exception as e:
                     yield f"WARNING: Failed to save checkpoint at chunk {current_num}: {e}\n"

    except Exception as e:
        yield f"FATAL ERROR during ingestion loop: {e}\n"


    # Save once at the end
    faiss_store.save_index(vectorstore)

    for url in processed_links:
        tracking_data[url] = time.time()
    save_tracking_data(tracking_data)

    success_msg = f"SUCCESS: Ingested {total_chunks} chunks from {len(processed_links)} websites."
    update_status("completed", message=success_msg)
    yield f"{success_msg}\n"
