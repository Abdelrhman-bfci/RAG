import os
import json
import time
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

def ingest_websites(force_fresh: bool = False):
    """
    Ingest website content from the configured WEBSITE_LINKS.
    Only processes links that haven't been successfully indexed recently.
    Yields progress updates as strings.
    """
    if force_fresh:
        yield "Fresh start requested for websites. Clearing website tracking...\n"
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)

    yield "Checking website links...\n"
    update_status("running", message="Scanning links...")
    
    links = Config.WEBSITE_LINKS
    if not links:
        msg = "No website links found in configuration."
        update_status("completed", message=msg)
        yield "WARNING: No websites to ingest.\n"
        return

    tracking_data = load_tracking_data()
    new_documents = []
    processed_links = []

    yield f"Found {len(links)} links. Starting extraction...\n"

    for url in links:
        try:
            # Simple tracking based on URL and date (as websites change content)
            # For a more robust solution, we could hash the content, but that requires loading it first.
            last_indexed = tracking_data.get(url, 0)
            if not force_fresh and (time.time() - last_indexed < 86400): # Skip if indexed in last 24h
                yield f"Skipping (indexed recently): {url}\n"
                continue

            yield f"Extracting content from: {url}\n"
            loader = WebBaseLoader(url)
            docs = loader.load()
            new_documents.extend(docs)
            processed_links.append(url)

        except Exception as e:
            yield f"FAILED to load {url}: {e}\n"

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
    batch_size = 10
    total_batches = (total_chunks + batch_size - 1) // batch_size
    start_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        current_batch_num = i // batch_size + 1
        batch_msg = f"Ingesting batch {current_batch_num}/{total_batches} (Chunks {i+1}-{min(i+batch_size, total_chunks)})"
        
        update_status("running", current=current_batch_num, total=total_batches, message=batch_msg, start_time=start_time)
        yield f"{batch_msg}\n"
        
        batch = splitted_docs[i:i + batch_size]
        faiss_store.add_documents(batch)

    for url in processed_links:
        tracking_data[url] = time.time()
    save_tracking_data(tracking_data)

    success_msg = f"SUCCESS: Ingested {total_chunks} chunks from {len(processed_links)} websites."
    update_status("completed", message=success_msg)
    yield f"{success_msg}\n"
