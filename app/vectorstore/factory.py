class VectorStoreFactory:
    @staticmethod
    def get_instance():
        from app.vectorstore.chroma_store import ChromaStore
        return ChromaStore()
