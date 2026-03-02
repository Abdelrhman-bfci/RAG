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
            from app.vectorstore.chroma_store import ChromaStore
            return ChromaStore()
