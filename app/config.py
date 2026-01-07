import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Vector Database Path (Local FAISS)
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_index")
    
    # Resource Directory for PDFs
    RESOURCE_DIR = os.getenv("RESOURCE_DIR", "/RESOURCE")
    
    # Database Connection String
    # Example: mysql+pymysql://user:password@host/db_name
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # PDF Ingestion Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1100"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))

    # Website Ingestion Settings
    _website_links_raw = os.getenv("WEBSITE_LINKS", "")
    WEBSITE_LINKS = [link.strip() for link in _website_links_raw.split(",") if link.strip()]

    # Database Ingestion Settings
    _ingest_tables_raw = os.getenv("INGEST_TABLES", "")
    INGEST_TABLES = [table.strip() for table in _ingest_tables_raw.split(",") if table.strip()]

    # Model Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # 'openai' or 'ollama'
    
    # OpenAI Settings
    EMBEDDING_MODEL = "text-embedding-3-large"
    LLM_MODEL = "gpt-4-turbo-preview"

    # Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    OLLAMA_CONTEXT_WINDOW = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "32768"))

    @classmethod
    def get_ollama_models(cls):
        """Fetch available models from Ollama API."""
        import requests
        try:
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except:
            pass
        return [cls.OLLAMA_LLM_MODEL]

    @classmethod
    def update_model(cls, model_name: str):
        """Update the current LLM model."""
        cls.OLLAMA_LLM_MODEL = model_name
        # Optional: persist to .env if needed, but for now just in-memory
        return True

if Config.LLM_PROVIDER == "openai" and not Config.OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")
