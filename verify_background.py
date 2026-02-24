import sys
import os
import sqlite3
from unittest.mock import MagicMock, patch

# Ensure app is in path
sys.path.append(os.getcwd())

# Mock the heavy modules BEFORE importing app components
def mock_package(name):
    # If the name has dots, we need to ensure parent exists and is a mock
    parts = name.split('.')
    for i in range(1, len(parts) + 1):
        pname = '.'.join(parts[:i])
        if pname not in sys.modules:
            m = MagicMock()
            m.__path__ = [] # Essential to be treated as a package
            sys.modules[pname] = m
    return sys.modules[name]

mock_package('langchain_community')
mock_package('langchain_community.document_loaders')
mock_package('langchain_community.vectorstores')
mock_package('langchain_community.vectorstores.chroma')
mock_package('chromadb')
mock_package('chromadb.config')
mock_package('langchain_openai')
mock_package('langchain_google_genai')
mock_package('langchain_community.embeddings')
mock_package('langchain_ollama')
mock_package('langchain_chroma')
mock_package('langchain_chroma.vectorstores')
mock_package('langchain_text_splitters')
mock_package('langchain_core')
mock_package('langchain_core.documents')

# Mock app components that import these
mock_package('app.vectorstore.factory')
mock_package('app.vectorstore.chroma_store')
mock_package('app.vectorstore.base')
mock_package('app.qa.rag_chain')

from app.ingestion.offline_web_ingest import init_tracking_db, ingest_offline_downloads, get_ingestion_status
from app.config import Config

def verify():
    print(f"Using DOWNLOAD_FOLDER: {Config.DOWNLOAD_FOLDER}")
    print(f"Using CRAWLER_DB: {Config.CRAWLER_DB}")
    
    # 1. Init DB
    init_tracking_db()
    print("DB Initialized")
    
    # 2. Check initial status
    status = get_ingestion_status()
    print(f"Initial Status: {status}")
    
    # 3. Simulate starting a background task (silent=True)
    print("\nEnvironment check: 'offine_downloads' content:")
    if os.path.exists(Config.DOWNLOAD_FOLDER):
        files = os.listdir(Config.DOWNLOAD_FOLDER)
        print(files)
        if not files:
            # Create a dummy file for testing
            with open(os.path.join(Config.DOWNLOAD_FOLDER, "test.pdf"), "w") as f:
                f.write("test content")
            print("Created dummy test.pdf")
    else:
        os.makedirs(Config.DOWNLOAD_FOLDER)
        with open(os.path.join(Config.DOWNLOAD_FOLDER, "test.pdf"), "w") as f:
            f.write("test content")
        print("Created offine_downloads and test.pdf")

    print("\nStarting simulated background (silent) ingestion...")
    
    # We'll patch the actual processing logic to avoid library dependencies
    with patch('app.ingestion.offline_web_ingest.OfflineWebIngestor') as MockIngestor:
        mock_instance = MockIngestor.return_value
        mock_instance.ingest_file.return_value = True
        
        generator = ingest_offline_downloads(silent=True)
        count = 0
        for item in generator:
            count += 1
            # item should be None when silent=True
            if count % 1 == 0:
                current_status = get_ingestion_status()
                print(f"Progress update {count}: {current_status['message']} ({current_status['current_batch']}/{current_status['total_batches']})")
            
    print("\nFinished simulated background ingestion")
    
    # 4. Check final status
    final_status = get_ingestion_status()
    print(f"Final Status: {final_status}")
    
    # 5. Verify skip logic
    print("\nVerifying skip logic (running again)...")
    with patch('app.ingestion.offline_web_ingest.OfflineWebIngestor') as MockIngestor:
        generator2 = ingest_offline_downloads(silent=True)
        for _ in generator2:
            pass
    
    skip_status = get_ingestion_status()
    print(f"Status after second run: {skip_status}")

if __name__ == "__main__":
    verify()
