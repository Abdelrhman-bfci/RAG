import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Auth
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "ASUAIADMIN")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "ASUAIADMIN")

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Vector Database Settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
    VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma") # 'chroma' or 'faiss'
    VECTOR_SEARCH_WEIGHT = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7")) # 0.0 to 1.0 (Vector vs Keyword)
    
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
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # 'openai', 'ollama', or 'vllm'
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER) # default to same as LLM
    
    # OpenAI Settings
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4-turbo-preview")

    # Gemini Settings
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    # Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    VLLM_EMBEDDING_MODEL = os.getenv("VLLM_EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    OLLAMA_CONTEXT_WINDOW = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "32768"))

    # vLLM Settings (OpenAI Compatible)
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:9090/v1")
    VLLM_MODEL = os.getenv("VLLM_MODEL", "model")

    @classmethod
    def get_ollama_models(cls, base_url: str = None):
        """Fetch available models from Ollama API."""
        import requests
        url = base_url or cls.OLLAMA_BASE_URL
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except:
            pass
        return [cls.OLLAMA_LLM_MODEL]

    @classmethod
    def get_vllm_models(cls, base_url: str = None):
        """Fetch available models from vLLM API."""
        import requests
        url = base_url or cls.VLLM_BASE_URL
        try:
            response = requests.get(f"{url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [m["id"] for m in models]
        except:
            pass
        return [cls.VLLM_MODEL]

    @classmethod
    def get_openai_models(cls, api_key: str = None, base_url: str = None):
        """Fetch available models from OpenAI."""
        import requests
        url = base_url or cls.OPENAI_BASE_URL
        key = api_key or cls.OPENAI_API_KEY
        if not key: return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"]
        try:
            response = requests.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [m["id"] for m in models if "gpt" in m["id"] or "o1" in m["id"]]
        except:
            pass
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"]

    @classmethod
    def get_gemini_models(cls, api_key: str = None):
        """List common Gemini models."""
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]

    # Crawler Settings
    DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER", "downloads")
    CRAWLER_DB = os.getenv("CRAWLER_DB", "crawler_data.db")
    _allowed_ext_raw = os.getenv("ALLOWED_EXTENSIONS", ".pdf")
    ALLOWED_EXTENSIONS = {ext.strip() if ext.strip().startswith('.') else f".{ext.strip()}" 
                          for ext in _allowed_ext_raw.split(',') if ext.strip()}
    
    # Document Summarization Settings
    SUMMARY_CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", "4000"))
    SUMMARY_CHUNK_OVERLAP = int(os.getenv("SUMMARY_CHUNK_OVERLAP", "200"))
    SHOW_SUMMARY_CHUNKS = os.getenv("SHOW_SUMMARY_CHUNKS", "False").lower() == "true"

    @classmethod
    def update_config(cls, updates: dict):
        """Update multiple configuration values and persist to .env."""
        env_path = ".env"
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        
        # Track which keys we've updated in the file
        applied_updates = set()
        new_lines = []
        
        for line in lines:
            updated_line = line
            for key, value in updates.items():
                if line.startswith(f"{key}="):
                    updated_line = f"{key}={value}\n"
                    applied_updates.add(key)
                    # Also update in memory
                    if hasattr(cls, key):
                        if key == "SHOW_SUMMARY_CHUNKS":
                            setattr(Config, key, str(value).lower() == "true")
                        elif key == "VECTOR_SEARCH_WEIGHT":
                            setattr(Config, key, float(value))
                        else:
                            setattr(Config, key, value)
            new_lines.append(updated_line)
            
        # Add any new keys that weren't in the file
        for key, value in updates.items():
            if key not in applied_updates:
                new_lines.append(f"{key}={value}\n")
                if hasattr(cls, key):
                    if key == "SHOW_SUMMARY_CHUNKS":
                        setattr(Config, key, str(value).lower() == "true")
                    elif key == "VECTOR_SEARCH_WEIGHT":
                        setattr(Config, key, float(value))
                    else:
                        setattr(Config, key, value)
                    
        with open(env_path, "w") as f:
            f.writelines(new_lines)
            
        return True

    @classmethod
    def update_model(cls, model_name: str):
        """Legacy method for backward compatibility."""
        return cls.update_config({"OLLAMA_LLM_MODEL": model_name})

if Config.LLM_PROVIDER == "openai" and not Config.OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")
