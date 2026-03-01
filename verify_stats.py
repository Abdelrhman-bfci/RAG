import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.vectorstore.chroma_store import ChromaStore
from app.config import Config

def verify_stats():
    print(f"Vector DB Path: {Config.VECTOR_DB_PATH}")
    print(f"Collection Name: {Config.CHROMA_COLLECTION_NAME}")
    
    store = ChromaStore()
    stats = store.get_index_stats()
    
    print("\n--- Index Statistics ---")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Unique Sources Found: {len(stats['sources'])}")
    
    if stats['total_chunks'] > 0:
        if stats['total_documents'] > 0:
            print("\n✅ SUCCESS: Total Documents is non-zero.")
        else:
            print("\n❌ FAILURE: Total Documents is 0 despite non-zero chunks.")
            
        print("\nSources and Chunk Counts:")
        for source, count in list(stats['sources'].items())[:10]:
            print(f" - {source}: {count} chunks")
        if len(stats['sources']) > 10:
            print(f" ... and {len(stats['sources']) - 10} more sources")
            
        # Check if individual chunk counts are non-zero (which they should be if they are in the dict)
        all_non_zero = all(count > 0 for count in stats['sources'].values())
        if all_non_zero:
            print("\n✅ SUCCESS: All identified sources have non-zero chunk counts.")
        else:
            print("\n❌ FAILURE: Some sources have 0 chunks.")
    else:
        print("\n⚠️ WARNING: Collection is empty. Re-run after ingestion.")

if __name__ == "__main__":
    verify_stats()
