import os
import glob
import json
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore

TRACKING_FILE = "ingested_files.json"
STATUS_FILE = "ingestion_status.json"

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

def ingest_pdfs():
    """
    Ingest PDF files from the configured RESOURCE directory.
    Only processes files that have changed since the last run.
    """
    update_status("running", message="Scanning files...")
    
    resource_dir = Config.RESOURCE_DIR
    if not os.path.exists(resource_dir):
        msg = f"Resource directory {resource_dir} does not exist."
        print(msg)
        update_status("error", message=msg)
        return {"status": "error", "message": "Resource directory not found"}

    pdf_files = glob.glob(os.path.join(resource_dir, "*.pdf"))
    
    if not pdf_files:
        msg = "No PDF files found in resource directory."
        print(msg)
        update_status("completed", message=msg)
        return {"status": "warning", "message": "No PDFs found"}

    tracking_data = load_tracking_data()
    new_documents = []
    processed_files_metadata = []

    print(f"Scanning {len(pdf_files)} files...")

    for pdf_path in pdf_files:
        try:
            # Create a unique ID based on file path, size, and modification time
            stats = os.stat(pdf_path)
            file_id = f"{pdf_path}_{stats.st_size}_{stats.st_mtime}"
            
            # Check if file is already ingested and valid
            if pdf_path in tracking_data and tracking_data[pdf_path] == file_id:
                print(f"Skipping unmodified: {os.path.basename(pdf_path)}")
                continue

            print(f"Processing new/modified: {os.path.basename(pdf_path)}")
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            new_documents.extend(docs)
            processed_files_metadata.append((pdf_path, file_id))

        except Exception as e:
            print(f"Failed to load {pdf_path}: {e}")

    if not new_documents:
        msg = "All files are up to date."
        print(msg)
        update_status("completed", message=msg)
        return {"status": "success", "message": "All files up to date"}

    # Split documents
    update_status("running", message="Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    splitted_docs = text_splitter.split_documents(new_documents)
    total_chunks = len(splitted_docs)
    print(f"Generated {total_chunks} new chunks.")

    # Store in FAISS in batches
    faiss_store = FAISSStore()
    batch_size = 10  # Keeping small batch size for stability with Ollama
    
    total_batches = (total_chunks + batch_size - 1) // batch_size
    start_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        current_batch_num = i // batch_size + 1
        ensure_message = f"Processing batch {current_batch_num}/{total_batches}"
        
        # Update status with ETA
        update_status("running", current=current_batch_num, total=total_batches, message=ensure_message, start_time=start_time)
        
        print(f"{ensure_message} (Chunks {i+1}-{min(i+batch_size, total_chunks)})")
        batch = splitted_docs[i:i + batch_size]
        faiss_store.add_documents(batch)
        print(f"Saved batch {current_batch_num}")

    # Update tracking data
    for pdf_path, file_id in processed_files_metadata:
        tracking_data[pdf_path] = file_id
    save_tracking_data(tracking_data)

    success_msg = f"Ingested {total_chunks} new chunks from {len(processed_files_metadata)} files."
    update_status("completed", message=success_msg)
    
    return {"status": "success", "message": success_msg}
