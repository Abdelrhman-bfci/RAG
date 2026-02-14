import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from app.config import Config

class ChromaStore:
    def __init__(self):
        if Config.EMBEDDING_PROVIDER == "ollama":
            self.embeddings = OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL
            )
        elif Config.EMBEDDING_PROVIDER == "vllm":
            self.embeddings = OpenAIEmbeddings(
                model=Config.VLLM_EMBEDDING_MODEL,
                openai_api_base=Config.VLLM_BASE_URL,
                openai_api_key="none"
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=getattr(Config, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                openai_api_key=Config.OPENAI_API_KEY
            )
        
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
        """Get statistics about the Chroma collection."""
        vectorstore = self.get_vectorstore()
        collection = vectorstore._collection
        
        # Get all metadata to aggregate statistics
        results = collection.get(include=["metadatas"])
        metadatas = results["metadatas"]
        
        stats = {
            "total_documents": 0,
            "total_chunks": len(metadatas),
            "sources": {}
        }
        
        for meta in metadatas:
            source = meta.get("source", "Unknown")
            if source.startswith(("http://", "https://")):
                source_name = source
            elif "Table: " in source:
                source_name = source
            else:
                source_name = os.path.basename(source) if "/" in source or "\\" in source else source
            
            if source_name not in stats["sources"]:
                stats["sources"][source_name] = 0
            stats["sources"][source_name] += 1
            
        stats["total_documents"] = len(stats["sources"])
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
