from app.config import Config
from app.vectorstore.faiss_store import FAISSStore
from app.vectorstore.chroma_store import ChromaStore

class VectorStoreFactory:
    @staticmethod
    def get_instance():
        provider = Config.VECTOR_STORE_PROVIDER.lower()
        if provider == "faiss":
            return FAISSStore()
        else:
            # Default to Chroma
            return ChromaStore()
