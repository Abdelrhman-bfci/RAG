import os
from langchain_community.vectorstores import FAISS
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

class FAISSStore:
    def __init__(self):
        base_embeddings = None
        if Config.EMBEDDING_PROVIDER == "ollama":
            base_embeddings = OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL
            )
        elif Config.EMBEDDING_PROVIDER == "vllm":
            # vLLM provides an OpenAI-compatible /v1/embeddings endpoint
            base_embeddings = OpenAIEmbeddings(
                model=Config.VLLM_EMBEDDING_MODEL,
                openai_api_base=Config.VLLM_BASE_URL, # Ensure this includes /v1
                openai_api_key="none"
            )
        else:
            base_embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
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
            print("DEBUG: Using NomicPrefixWrapper for FAISS")
        else:
            self.embeddings = base_embeddings
        
        self.vector_db_path = Config.VECTOR_DB_PATH

    def load_index(self):
        """
        Load the FAISS index from disk.
        If it doesn't exist, return None.
        """
        if os.path.exists(self.vector_db_path) and os.path.isdir(self.vector_db_path):
            return FAISS.load_local(
                self.vector_db_path, 
                self.embeddings,
                allow_dangerous_deserialization=True # Required for loading pickle files safely in controlled env
            )
        return None

    def get_vectorstore(self):
        """Alias for load_index to match ChromaStore interface."""
        return self.load_index()

    def save_index(self, vectorstore: FAISS):
        """
        Save the FAISS index to disk.
        """
        vectorstore.save_local(self.vector_db_path)

    def add_documents(self, documents):
        """
        Add documents to the vector store.
        If the store exists, extend it. If not, create a new one.
        """
        vectorstore = self.load_index()
        
        if vectorstore:
            vectorstore.add_documents(documents)
        else:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
        self.save_index(vectorstore)
        return vectorstore

    def delete_source(self, source_name: str):
        """
        Remove all documents associated with a specific source from the FAISS index.
        """
        vectorstore = self.load_index()
        if not vectorstore:
            return False

        # Filter out documents where metadata['source'] matches source_name
        # Note: FAISS doesn't have a direct 'delete by metadata' easily in LangChain's version
        # So we reconstruct the index if it's small, or use docstore IDs if possible.
        # For our local scale, reconstructing or filtering in memory is fine.
        
        docstore = vectorstore.docstore._dict
        ids_to_delete = [
            doc_id for doc_id, doc in docstore.items() 
            if doc.metadata.get("source") == source_name or 
               doc.metadata.get("table") == source_name or 
               os.path.basename(doc.metadata.get("source", "")) == source_name
        ]

        if ids_to_delete:
            vectorstore.delete(ids_to_delete)
            self.save_index(vectorstore)
            return True
        return False

    def delete_sources(self, source_names: list):
        """
        Remove documents associated with multiple sources efficiently.
        """
        if not source_names:
            return False
            
        vectorstore = self.load_index()
        if not vectorstore:
            return False

        docstore = vectorstore.docstore._dict
        ids_to_delete = []
        
        # Optimize by converting source_names to set for O(1) lookup ?? 
        # Actually metadata check is scanning all docs anyway.
        source_set = set(source_names)
        
        for doc_id, doc in docstore.items():
            src = doc.metadata.get("source")
            tbl = doc.metadata.get("table")
            # Check exact matches
            if src in source_set or tbl in source_set:
                ids_to_delete.append(doc_id)
                continue
                
            # Check basename matches (for files)
            if src and os.path.basename(src) in source_set:
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            vectorstore.delete(ids_to_delete)
            self.save_index(vectorstore)
            return True
        return False

    def get_index_stats(self):
        """
        Analyze the index and return statistics about ingested documents.
        """
        stats = {"total_documents": 0, "total_chunks": 0, "sources": {}}
        try:
            vectorstore = self.load_index()
            if not vectorstore:
                return stats
                
            docstore = vectorstore.docstore._dict
            stats["total_chunks"] = len(docstore)
            
            for doc_id, doc in docstore.items():
                source = doc.metadata.get("source", "Unknown")
                # Distinguish between URLs and file paths
                if source.startswith(("http://", "https://")):
                    source_name = source
                elif "Table: " in source:
                    # Strip prefix for consistent matching with resource lists
                    source_name = source.replace("Table: ", "")
                else:
                    source_name = os.path.basename(source) if "/" in source or "\\" in source else source
                
                if source_name not in stats["sources"]:
                    stats["sources"][source_name] = 0
                stats["sources"][source_name] += 1
                
            stats["total_documents"] = len(stats["sources"])
        except Exception as e:
            print(f"Error retrieving FAISS stats: {e}")
            
        return stats

    def get_source_content(self, source_name: str, limit: int = None):
        """
        Retrieve content chunks for a specific source.
        If limit is None or -1, returns all chunks.
        """
        vectorstore = self.load_index()
        if not vectorstore:
            return []
            
        docstore = vectorstore.docstore._dict
        chunks = []
        
        for doc_id, doc in docstore.items():
            src = doc.metadata.get("source", "Unknown")
            basename = os.path.basename(src) if "/" in src or "\\" in src else src
            
            # loose match to handle full paths vs basenames
            if basename == source_name or src == source_name:
                chunks.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "N/A"),
                    "id": doc_id
                })
                
        # Sort by page number if possible
        try:
            chunks.sort(key=lambda x: int(x["page"]) if isinstance(x["page"], int) or (isinstance(x["page"], str) and x["page"].isdigit()) else 9999)
        except:
            pass
            
        if limit and limit > 0:
            return chunks[:limit]
        return chunks

    def clear_all(self):
        """Wipe the entire FAISS index."""
        import shutil
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
        return True
