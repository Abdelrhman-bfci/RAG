import os
import glob
import json
import time
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    Docx2txtLoader, 
    TextLoader, 
    CSVLoader, 
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore
import pymupdf4llm
from langchain.docstore.document import Document

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

def get_loader(file_path: str):
    """Returns the appropriate LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyMuPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext in [".xlsx", ".xls"]:
         return UnstructuredExcelLoader(file_path)
    return None

def ingest_documents(force_fresh: bool = False):
    """
    Ingest various document files from the configured RESOURCE directory.
    Only processes files that have changed since the last run unless force_fresh=True.
    Yields progress updates as strings.
    """
    if force_fresh:
        yield "Fresh start requested. Clearing document tracking data...\n"
        if os.path.exists(TRACKING_FILE):
            os.remove(TRACKING_FILE)

    yield "Scanning documents...\n"
    update_status("running", message="Scanning files...")
    
    resource_dir = Config.RESOURCE_DIR
    if not os.path.exists(resource_dir):
        msg = f"Resource directory {resource_dir} does not exist."
        update_status("error", message=msg)
        yield f"ERROR: {msg}\n"
        return

    # Support multiple extensions
    extensions = ["*.pdf", "*.docx", "*.txt", "*.md", "*.csv", "*.xlsx", "*.xls"]
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(resource_dir, ext)))
    
    if not all_files:
        msg = "No documents found in resource directory."
        update_status("completed", message=msg)
        yield "WARNING: No documents found.\n"
        return

    tracking_data = load_tracking_data()
    new_documents = []
    processed_files_metadata = []

    yield f"Found {len(all_files)} documents. Checking for updates...\n"

    for file_path in all_files:
        try:
            stats = os.stat(file_path)
            file_id = f"{file_path}_{stats.st_size}_{stats.st_mtime}"
            
            if not force_fresh and file_path in tracking_data and tracking_data[file_path] == file_id:
                continue

            yield f"Reading: {os.path.basename(file_path)}\n"
            
            # Ensure we remove old versions of this file from the index before adding new ones
            try:
                faiss_store = FAISSStore()
                faiss_store.delete_source(file_path)
            except Exception as e:
                print(f"Warning: Failed to clear old index for {file_path}: {e}")

            if ext == ".pdf":
                # Use pymupdf4llm for high-quality Markdown conversion
                md_content = pymupdf4llm.to_markdown(file_path)
                docs = [Document(page_content=md_content, metadata={"source": file_path})]
                new_documents.extend(docs)
                processed_files_metadata.append((file_path, file_id))
            else:
                loader = get_loader(file_path)
                if loader:
                    docs = loader.load()
                    new_documents.extend(docs)
                    processed_files_metadata.append((file_path, file_id))
                else:
                    yield f"Skipping unsupported file type: {os.path.basename(file_path)}\n"

        except Exception as e:
            yield f"FAILED to load {os.path.basename(file_path)}: {e}\n"

    if not new_documents:
        yield "All documents are already up to date.\n"
        update_status("completed", message="All documents are up to date.")
        return

    yield f"Splitting {len(new_documents)} items into chunks...\n"
    update_status("running", message="Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", "", "---", "##", "#"] 
    )
    
    splitted_docs = text_splitter.split_documents(new_documents)
    total_chunks = len(splitted_docs)
    yield f"Generated {total_chunks} new chunks. Starting vectorization...\n"

    # Create/Load vectorstore
    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index() # Load once
    
    # PROCESS ONE CHUNK AT A TIME
    start_time = time.time()
    
    import gc
    import traceback
    
    try:
        for i, doc in enumerate(splitted_docs):
            current_num = i + 1
            
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
            
            msg = f"Ingesting chunk {current_num}/{total_chunks}"
            if current_num % 5 == 0:
                yield f"{msg}\n"
            
            if current_num % 10 == 0:
                update_status("running", current=current_num, total=total_chunks, message=msg, start_time=start_time)
                gc.collect() 
            
            if current_num % 50 == 0:
                yield "Saving checkpoint...\n"
                try:
                    faiss_store.save_index(vectorstore)
                except Exception as e:
                     yield f"WARNING: Failed to save checkpoint at chunk {current_num}: {e}\n"

        for file_path, file_id in processed_files_metadata:
            tracking_data[file_path] = file_id
        save_tracking_data(tracking_data)

        success_msg = f"SUCCESS: Ingested {total_chunks} chunks from {len(processed_files_metadata)} documents."
        update_status("completed", message=success_msg)
        yield f"{success_msg}\n"

    except Exception as e:
        yield f"FATAL ERROR during ingestion loop: {e}\n"
        print(f"FATAL INGESTION ERROR: {traceback.format_exc()}")
        
    finally:
        if vectorstore:
            yield "Finalizing index save...\n"
            try:
                faiss_store.save_index(vectorstore)
                yield "Index saved successfully.\n"
            except Exception as e:
                yield f"ERROR saving final index: {e}\n"
