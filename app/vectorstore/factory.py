class VectorStoreFactory:
    @staticmethod
    def get_instance():
        from app.config import Config
        provider = Config.VECTOR_STORE_PROVIDER.lower()
        
        if provider == "faiss":
            from app.vectorstore.faiss_store import FAISSStore
            return FAISSStore()
        else:
            # Default to Chroma
            try:
                from app.vectorstore.chroma_store import ChromaStore
                return ChromaStore()
            except ImportError as e:
                print(f"ChromaStore import failed: {e}. Falling back to FAISS if possible.")
                from app.vectorstore.faiss_store import FAISSStore
                return FAISSStore()
