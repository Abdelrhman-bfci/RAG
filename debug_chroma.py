import chromadb
from app.config import Config

client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
collections = client.list_collections()
print(f"Collections: {[c.name for c in collections]}")

for coll_name in [c.name for c in collections]:
    coll = client.get_collection(coll_name)
    count = coll.count()
    print(f"Collection '{coll_name}' has {count} items.")

    # Try to get metadatas with no limit
    try:
        results = coll.get(include=["metadatas"], limit=10)
        print(f"Sample metadata keys: {results['metadatas'][0].keys() if results['metadatas'] else 'None'}")
    except Exception as e:
        print(f"Error getting metadata: {e}")
