import sys
import os

# Ensure the app module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.vectorstore.factory import VectorStoreFactory
from app.ingestion.document_ingest import ingest_documents

def run_migration():
    print("Starting Migration...")
    
    # 1. Clear Vector Store
    print("Clearing Vector Store...")
    vs = VectorStoreFactory.get_instance()
    vs.clear_all()
    print("Vector Store cleared.")
    
    # 2. Clear tracking files
    try:
        if os.path.exists("ingested_files.json"):
            os.remove("ingested_files.json")
        if os.path.exists("ingestion_status.json"):
            os.remove("ingestion_status.json")
    except Exception as e:
        print(f"Error removing tracking files: {e}")
        
    # 3. Trigger Ingestion
    print("Triggering Ingestion...")
    try:
        for update in ingest_documents(force_fresh=True):
            print(update)
    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    run_migration()
