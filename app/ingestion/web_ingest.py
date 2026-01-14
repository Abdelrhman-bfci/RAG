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
    max_depth = kwargs.get('depth', 10)
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
                
                # Add to targets if it's not the seed URL (or if seed needs re-indexing which is checked above)
                # For the seed URL itself, we adding it to targets to be loaded
                target_urls.append(current_url)
                
                # Stop if we reached max depth, no need to parse for links
                if current_depth >= max_depth:
                    continue

                try:
                    yield f"Scanning [Depth {current_depth}]: {current_url}\n"
                    response = requests.get(current_url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    found_on_page = 0
                    for a in soup.find_all('a', href=True):
                        if max_pages != -1 and len(target_urls) >= max_pages:
                            break
                            
                        href = a['href']
                        text = a.get_text().lower().strip()
                        full_url = urljoin(current_url, href)
                        
                        # Remove fragment identifier to avoid duplicate crawling
                        full_url = full_url.split('#')[0]
                        parsed_full = urlparse(full_url)
                        
                        if parsed_full.netloc == domain and full_url not in visited:
                            # Heuristic filter for relevance (optional, but good to keep to reduce noise)
                            path = parsed_full.path.lower()
                            keywords = ["about", "history", "vision", "mission", "leader", "academic", "program", "department", "news", "blog", "service", "product"]
                            
                            # We can be a bit more lenient with deep crawling or keep strict
                            if any(kw in path for kw in keywords) or any(kw in text for kw in keywords) or current_depth == 0:
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
                # Load in batches to avoid overwhelming
                batch_size = 10
                for i in range(0, len(target_urls), batch_size):
                    batch_urls = target_urls[i:i+batch_size]
                    yield f"Loading batch {i//batch_size + 1} ({len(batch_urls)} urls)...\n"
                    
                    try:
                        loader = WebBaseLoader(batch_urls)
                        # Set requests timeout or other loader parameters if possible in this version of langchain
                        # loader.requests_kwargs = {'timeout': 10} 
                        
                        docs = loader.load()
                        
                        # Filter valid docs
                        valid_batched_docs = []
                        for d in docs:
                            if len(d.page_content.strip()) > 300:
                                valid_batched_docs.append(d)
                                source_url = d.metadata.get("source", "Unknown URL")
                                yield f" - Extracted: {source_url}\n"
                        
                        new_documents.extend(valid_batched_docs)
                        
                    except Exception as e:
                        yield f"Batch extraction failed: {e}\n"
    
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
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    splitted_docs = text_splitter.split_documents(new_documents)
    total_chunks = len(splitted_docs)
    yield f"Generated {total_chunks} chunks. Starting vectorization...\n"

    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index() # Load once
    
    batch_size = 100 # Increased from 10
    total_batches = (total_chunks + batch_size - 1) // batch_size
    start_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        current_batch_num = i // batch_size + 1
        batch_msg = f"Ingesting batch {current_batch_num}/{total_batches} (Chunks {i+1}-{min(i+batch_size, total_chunks)})"
        
        update_status("running", current=current_batch_num, total=total_batches, message=batch_msg, start_time=start_time)
        yield f"{batch_msg}\n"
        
        batch = splitted_docs[i:i + batch_size]
        
        # Add documents to memory
        if vectorstore:
            vectorstore.add_documents(batch)
        else:
            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.from_documents(batch, faiss_store.embeddings)

    # Save once at the end
    faiss_store.save_index(vectorstore)

    for url in processed_links:
        tracking_data[url] = time.time()
    save_tracking_data(tracking_data)

    success_msg = f"SUCCESS: Ingested {total_chunks} chunks from {len(processed_links)} websites."
    update_status("completed", message=success_msg)
    yield f"{success_msg}\n"
