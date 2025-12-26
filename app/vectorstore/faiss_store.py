import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from app.config import Config

class FAISSStore:
    def __init__(self):
        if Config.LLM_PROVIDER == "ollama":
            self.embeddings = OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY
            )
        
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
