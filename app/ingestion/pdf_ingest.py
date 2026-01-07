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

def ingest_pdfs(force_fresh: bool = False):
    """
    Ingest PDF files from the configured RESOURCE directory.
    Only processes files that have changed since the last run unless force_fresh=True.
    Yields progress updates as strings.
    """
    if force_fresh:
        yield "Fresh start requested. Clearing existing index...\n"
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)
        faiss_store = FAISSStore()
        if os.path.exists(faiss_store.vector_db_path):
            import shutil
            shutil.rmtree(faiss_store.vector_db_path)
            yield "Old index deleted.\n"

    yield "Scanning files...\n"
    update_status("running", message="Scanning files...")
    
    resource_dir = Config.RESOURCE_DIR
    if not os.path.exists(resource_dir):
        msg = f"Resource directory {resource_dir} does not exist."
        update_status("error", message=msg)
        yield f"ERROR: {msg}\n"
        return

    pdf_files = glob.glob(os.path.join(resource_dir, "*.pdf"))
    
    if not pdf_files:
        msg = "No PDF files found in resource directory."
        update_status("completed", message=msg)
        yield "WARNING: No PDFs found.\n"
        return

    tracking_data = load_tracking_data()
    new_documents = []
    processed_files_metadata = []

    yield f"Found {len(pdf_files)} PDF files. Checking for updates...\n"

    for pdf_path in pdf_files:
        try:
            stats = os.stat(pdf_path)
            file_id = f"{pdf_path}_{stats.st_size}_{stats.st_mtime}"
            
            if not force_fresh and pdf_path in tracking_data and tracking_data[pdf_path] == file_id:
                continue

            yield f"Reading: {os.path.basename(pdf_path)}\n"
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            new_documents.extend(docs)
            processed_files_metadata.append((pdf_path, file_id))

        except Exception as e:
            yield f"FAILED to load {os.path.basename(pdf_path)}: {e}\n"

    if not new_documents:
        yield "All files are already up to date.\n"
        update_status("completed", message="All files are up to date.")
        return

    yield f"Splitting {len(new_documents)} pages into chunks...\n"
    update_status("running", message="Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    splitted_docs = text_splitter.split_documents(new_documents)
    total_chunks = len(splitted_docs)
    yield f"Generated {total_chunks} new chunks. Starting vectorization...\n"

    # Create/Load vectorstore
    # Create/Load vectorstore
    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index() # Load once
    
    # PROCESS ONE CHUNK AT A TIME (Slowest but most stable)
    total_chunks = len(splitted_docs)
    start_time = time.time()
    
    import gc

    try:
        # We manually iterate to keep strict control
        for i, doc in enumerate(splitted_docs):
            current_num = i + 1
            
            # 1. Add single document
            if vectorstore:
                vectorstore.add_documents([doc])
            else:
                from langchain_community.vectorstores import FAISS
                vectorstore = FAISS.from_documents([doc], faiss_store.embeddings)
            
            # 2. Yield progress every single chunk to keep connection alive
            msg = f"Ingesting chunk {current_num}/{total_chunks}"
            yield f"{msg}\n"
            
            # 3. Update status file less frequently to save I/O
            if current_num % 10 == 0:
                update_status("running", current=current_num, total=total_chunks, message=msg, start_time=start_time)
                gc.collect() # Free RAM
            
            # 4. Checkpoint every 50 chunks (approx every minute)
            if current_num % 50 == 0:
                yield "Saving checkpoint...\n"
                faiss_store.save_index(vectorstore)

        # Update metadata for processed files
        for pdf_path, file_id in processed_files_metadata:
            tracking_data[pdf_path] = file_id
        save_tracking_data(tracking_data)

        success_msg = f"SUCCESS: Ingested {total_chunks} chunks from {len(processed_files_metadata)} files."
        update_status("completed", message=success_msg)
        yield f"{success_msg}\n"

    finally:
        if vectorstore:
            yield "Finalizing index save...\n"
            faiss_store.save_index(vectorstore)
