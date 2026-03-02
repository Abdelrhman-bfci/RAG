import os
import json
import sys

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.vectorstore.factory import VectorStoreFactory
from app.config import Config

def verify():
    print(f"--- RAG Diagnostic ---")
    print(f"Vector Store Provider: {Config.VECTOR_STORE_PROVIDER}")
    print(f"Embedding Provider: {Config.EMBEDDING_PROVIDER}")
    print(f"Chroma Path: {Config.VECTOR_DB_PATH}")
    print(f"Chroma Collection: {Config.CHROMA_COLLECTION_NAME}")
    
    try:
        from app.vectorstore.factory import VectorStoreFactory
        store = VectorStoreFactory.get_instance()
        print(f"Store Instance Type: {type(store)}")
        
        stats = store.get_index_stats()
        print("\n--- Statistics ---")
        print(json.dumps(stats, indent=2))
        
        if stats.get("total_chunks", 0) > 0:
            print("\nSUCCESS: Chunks found in the index!")
        else:
            print("\nWARNING: No chunks found in the current collection.")
            if "available_collections" in stats:
                print(f"Available collections: {stats['available_collections']}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
