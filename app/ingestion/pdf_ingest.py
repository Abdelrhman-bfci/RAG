import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore

def ingest_pdfs():
    """
    Ingest all PDF files from the configured RESOURCE directory.
    """
    resource_dir = Config.RESOURCE_DIR
    if not os.path.exists(resource_dir):
        print(f"Resource directory {resource_dir} does not exist.")
        return {"status": "error", "message": "Resource directory not found"}

    pdf_files = glob.glob(os.path.join(resource_dir, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in resource directory.")
        return {"status": "warning", "message": "No PDFs found"}

    documents = []
    print(f"Found {len(pdf_files)} PDF(s). Processing...")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {pdf_path}")
        except Exception as e:
            print(f"Failed to load {pdf_path}: {e}")

    if not documents:
         return {"status": "warning", "message": "No documents extracted"}

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    splitted_docs = text_splitter.split_documents(documents)
    print(f"Generated {len(splitted_docs)} chunks.")

    # Store in FAISS
    faiss_store = FAISSStore()
    faiss_store.add_documents(splitted_docs)

    return {"status": "success", "message": f"Ingested {len(splitted_docs)} chunks from {len(pdf_files)} files."}
