import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
from app.config import Config

class NomicPrefixWrapper(Embeddings):
    """Wraps an embedding model to add Nomic-specific prefixes for better retrieval."""
    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed_texts = [f"search_document: {t}" for t in texts]
        return self.base_embeddings.embed_documents(prefixed_texts)

    def embed_query(self, text: str) -> List[float]:
        return self.base_embeddings.embed_query(f"search_query: {text}")

class ChromaStore:
    def __init__(self):
        base_embeddings = None
        if Config.EMBEDDING_PROVIDER == "ollama":
            base_embeddings = OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL
            )
        elif Config.EMBEDDING_PROVIDER == "vllm":
            base_embeddings = OpenAIEmbeddings(
                model=Config.VLLM_EMBEDDING_MODEL,
                openai_api_base=Config.VLLM_BASE_URL,
                openai_api_key="none"
            )
        else:
            base_embeddings = OpenAIEmbeddings(
                model=getattr(Config, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                openai_api_key=Config.OPENAI_API_KEY
            )
            
        # Wrap Nomic models automatically
        is_nomic = False
        if Config.EMBEDDING_PROVIDER == "ollama" and "nomic" in Config.OLLAMA_EMBEDDING_MODEL.lower():
            is_nomic = True
        elif Config.EMBEDDING_PROVIDER == "vllm" and "nomic" in Config.VLLM_EMBEDDING_MODEL.lower():
            is_nomic = True
        
        if is_nomic:
            self.embeddings = NomicPrefixWrapper(base_embeddings)
            print("DEBUG: Using NomicPrefixWrapper for Chroma")
        else:
            self.embeddings = base_embeddings
        
        self.persist_directory = Config.VECTOR_DB_PATH
        self.collection_name = Config.CHROMA_COLLECTION_NAME

    def get_vectorstore(self):
        """Initialize and return the Chroma vector store."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents):
        """Add documents to ChromaDB."""
        vectorstore = self.get_vectorstore()
        vectorstore.add_documents(documents)
        return vectorstore

    def delete_source(self, source_name: str):
        """Delete all documents associated with a specific source."""
        vectorstore = self.get_vectorstore()
        
        # Chroma supports efficient deletion by metadata filters
        try:
            # We check both source and table metadata fields as used in the app
            vectorstore.delete(where={"$or": [
                {"source": source_name},
                {"table": source_name}
            ]})
            
            # Also check basename for files (matching FAISS logic)
            basename = os.path.basename(source_name)
            if basename != source_name:
                 vectorstore.delete(where={"source": basename})
            return True
        except Exception as e:
            print(f"Error deleting from Chroma: {e}")
            return False

    def delete_sources(self, source_names: list):
        """Delete documents for multiple sources."""
        if not source_names:
            return False
        
        success = True
        for name in source_names:
            if not self.delete_source(name):
                success = False
        return success

    def get_index_stats(self):
        """
        Extract statistics from ChromaDB about chunks and sources.
        """
        stats = {
            "total_chunks": 0,
            "total_documents": 0,
            "sources": {},
            "provider": "chroma",
            "collection": self.collection_name,
            "path": os.path.abspath(self.persist_directory),
            "debug_info": []
        }
        
        try:
            print(f"Retrieving stats for Chroma collection: {self.collection_name}")
            stats["debug_info"].append(f"Initializing Chroma with path: {stats['path']}")
            
            vectorstore = self.get_vectorstore()
            if not vectorstore:
                stats["error"] = "Failed to initialize vectorstore"
                return stats
                
            collection = getattr(vectorstore, "_collection", None)
            if not collection:
                stats["error"] = "vectorstore._collection is None"
                # Try listing collections to see what's available
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=self.persist_directory)
                    colls = client.list_collections()
                    stats["available_collections"] = [c.name for c in colls]
                except Exception as e_coll:
                    stats["list_coll_error"] = str(e_coll)
                return stats
            
            # Use count() for efficiency
            total_chunks = collection.count()
            stats["total_chunks"] = total_chunks
            stats["debug_info"].append(f"collection.count() returned {total_chunks}")
            
            if total_chunks == 0:
                # Still check if other collections exist
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=self.persist_directory)
                    colls = client.list_collections()
                    stats["available_collections"] = [c.name for c in colls]
                    stats["debug_info"].append(f"Found {len(colls)} collections: {stats['available_collections']}")
                except:
                    pass
                return stats
            
            # Batch retrieval of metadatas to avoid memory/timeout issues
            batch_size = 5000
            for i in range(0, total_chunks, batch_size):
                results = collection.get(
                    include=["metadatas"], 
                    limit=batch_size, 
                    offset=i
                )
                
                if results and "metadatas" in results:
                    metadatas = results["metadatas"]
                    for meta in metadatas:
                        if not meta: continue
                        
                        # Check both 'source' and 'table' as used in ingestion scripts
                        source = meta.get("source") or meta.get("table") or "Unknown"
                        
                        if source.startswith(("http://", "https://")):
                            source_name = source
                        elif "Table: " in source:
                            source_name = source.replace("Table: ", "")
                        else:
                            # Handle both Windows and Unix style paths for source
                            source_name = os.path.basename(source) if "/" in source or "\\" in source else source
                        
                        if source_name not in stats["sources"]:
                            stats["sources"][source_name] = 0
                        stats["sources"][source_name] += 1
            
            stats["total_documents"] = len(stats["sources"])
            return stats
            
        except Exception as e:
            import traceback
            error_msg = f"Error retrieving index stats: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            stats["error"] = error_msg
            stats["traceback"] = traceback.format_exc()
            return stats

    def get_source_content(self, source_name: str, limit: int = None):
        """Retrieve content chunks for a specific source."""
        vectorstore = self.get_vectorstore()
        collection = vectorstore._collection
        
        # Query by source
        results = collection.get(
            where={"$or": [
                {"source": source_name},
                {"table": source_name}
            ]},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        for i in range(len(results["documents"])):
            doc = results["documents"][i]
            meta = results["metadatas"][i]
            chunks.append({
                "content": doc,
                "page": meta.get("page", "N/A"),
                "id": results["ids"][i]
            })
            
        # Sort by page
        try:
            chunks.sort(key=lambda x: int(x["page"]) if str(x["page"]).isdigit() else 9999)
        except:
            pass
            
        if limit and limit > 0:
            return chunks[:limit]
        return chunks

    def clear_all(self):
        """Wipe the entire vector store."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        return True
