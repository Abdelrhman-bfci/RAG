import os
import sqlite3
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # DB Settings config
    _SETTINGS_DB = "settings.db"

    # Auth
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "ASUAIADMIN")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "ASUAIADMIN")

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Vector Database Settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_index_v4")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
    VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma") # 'chroma'
    VECTOR_SEARCH_WEIGHT = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7")) # 0.0 to 1.0 (Vector vs Keyword)
    
    # Resource Directory for PDFs
    RESOURCE_DIR = os.getenv("RESOURCE_DIR", "/RESOURCE")
    
    # Database Connection String
    # Example: mysql+pymysql://user:password@host/db_name
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # PDF Ingestion Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Reranker Settings
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", "0.1"))
    LLM_K_FINAL = int(os.getenv("LLM_K_FINAL", "10"))
    LLM_HISTORY_K = int(os.getenv("LLM_HISTORY_K", "3"))
    LLM_HISTORY_SCORE_THRESHOLD = float(os.getenv("LLM_HISTORY_SCORE_THRESHOLD", "0.35"))

    # Mode-specific LLM settings
    CHAT_LLM_NUM_PREDICT = int(os.getenv("CHAT_LLM_NUM_PREDICT", "512"))
    DOC_LLM_NUM_CTX = int(os.getenv("DOC_LLM_NUM_CTX", "32768"))

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
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
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
    DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", "300"))
    CRAWL_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "300"))
    CRAWL_RETRIES = int(os.getenv("CRAWL_RETRIES", "3"))
    CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))
    CRAWL_SKIP_IMAGES = os.getenv("CRAWL_SKIP_IMAGES", "false").lower() in ("true", "1", "yes")
    CRAWL_USER_AGENT = os.getenv("CRAWL_USER_AGENT", "")  # Empty = use default in crawler_service
    _allowed_ext_raw = os.getenv("ALLOWED_EXTENSIONS", ".pdf")
    ALLOWED_EXTENSIONS = {ext.strip() if ext.strip().startswith('.') else f".{ext.strip()}" 
                          for ext in _allowed_ext_raw.split(',') if ext.strip()}
    
    # Document Summarization Settings
    SUMMARY_CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", "4000"))
    SUMMARY_CHUNK_OVERLAP = int(os.getenv("SUMMARY_CHUNK_OVERLAP", "200"))
    SUMMARY_MAX_WORKERS = int(os.getenv("SUMMARY_MAX_WORKERS", "4"))
    SHOW_SUMMARY_CHUNKS = os.getenv("SHOW_SUMMARY_CHUNKS", "False").lower() == "true"
    
    # Prompt Templates
    CHAT_TEMPLATE = os.getenv("CHAT_TEMPLATE", """You are a professional Document Assistant acting as a closed-domain reasoning engine.

CORE DIRECTIVE:
You must answer the user's question using ONLY the information provided in the "Context" below.
You are strictly forbidden from using outside knowledge, external facts, or training data.

INSTRUCTIONS:
1. **Search**: Look for the answer in the Context.
2. **Match**: If the answer is explicitly written there, rewrite it clearly.
3. **Logical Inference**: You are allowed to infer relationships based on document structure.
4. **Synthesis**: You may combine information from multiple parts of the Context to form a complete answer.
5. **Formatting**: Preserve lists, tables, and data structures from the original text when beneficial for clarity.
6. **Inline Citations**: You MUST cite your sources using numbered references.
   - After every fact or claim, append the reference number in square brackets, e.g. [1], [2].
   - If a single claim uses multiple sources, list them: [1][3].
   - At the END of your answer, include a "References" section listing each number with its source:
     ```
     **References:**
     [1] [Source Name (Page X)](URL)
     [2] [Source Name](URL)
     ```
   - Use Markdown link format: `[Display Text](URL)`.
   - If a page number is available, include it: `[Source Name (Page X)](URL)`.

CHAT HISTORY RULES:
- The "Chat History" is provided solely for resolving references (e.g., "it", "he", "that course").
- If the Current Question represents a topic change, **completely ignore** the subject matter of the Chat History.

FALLBACK:
If the answer cannot be reasonably derived from the provided Context using the rules above, you MUST output exactly:
"I cannot answer this based on the provided documents."

PROHIBITED ACTIONS:
- Do NOT write stories, poems, or jokes.
- Do NOT use outside knowledge (e.g. do not explain general concepts like "what is engineering" unless defined in Context).
- Do NOT ignore these rules.

Context:
{context}

Chat History:
{history}

Question: {question}""")

    DOCUMENT_TEMPLATE = os.getenv("DOCUMENT_TEMPLATE", """You are an expert analyst reviewing the provided full documents.

CONTEXT (Full Documents):
{context}

HISTORY:
{history}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide a comprehensive answer proportional to the document size.
2. Structure your response using Markdown: use clear Headings, Subheadings, and Bullet Points.
3. If the documents contain data, format it into Tables where appropriate.
4. Do not omit key details. Prioritize completeness over brevity.
5. Cite sources using numbered inline references [1], [2] and list them at the end:
   ```
   **References:**
   [1] [Source Name (Page X)](URL)
   ```""")

    @classmethod
    def init_db(cls):
        """Initialize settings DB and load persisted overrides."""
        try:
            conn = sqlite3.connect(cls._SETTINGS_DB)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT
                )
            ''')
            
            
            # Check if table is empty
            cursor.execute('SELECT COUNT(*) FROM settings')
            count = cursor.fetchone()[0]
            
            if count == 0:
                # First run: Dump ALL current Config capabilities into the DB as defaults
                defaults = []
                for k in dir(cls):
                    if k.isupper() and not k.startswith('_'):
                        val = getattr(cls, k)
                        if isinstance(val, (str, int, float, bool)):
                            defaults.append((k, str(val)))
                cursor.executemany('INSERT INTO settings (setting_key, setting_value) VALUES (?, ?)', defaults)
            else:
                # Load from DB and override defaults
                cursor.execute('SELECT setting_key, setting_value FROM settings')
                for row in cursor.fetchall():
                    key, str_value = row
                    if hasattr(cls, key):
                        setattr(cls, key, cls._cast_value(key, str_value))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"WARNING: Could not load settings from DB: {e}")

    @classmethod
    def update_config(cls, updates: dict):
        """Update multiple configuration values and persist to settings DB."""
        try:
            conn = sqlite3.connect(cls._SETTINGS_DB)
            cursor = conn.cursor()
            
            for key, value in updates.items():
                str_value = str(value)
                cursor.execute(
                    '''INSERT INTO settings (setting_key, setting_value) 
                       VALUES (?, ?) 
                       ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value''', 
                    (key, str_value)
                )
                
                # Also update in memory
                if hasattr(cls, key):
                    setattr(cls, key, cls._cast_value(key, str_value))
                        
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"ERROR: Failed to update settings in DB: {e}")
            return False

    @classmethod
    def _cast_value(cls, key: str, str_value: str):
        """Intelligently cast string to original attribute type."""
        current_val = getattr(cls, key)
        if isinstance(current_val, bool):
            return str(str_value).lower() in ("true", "1", "yes", "t", "on")
        elif isinstance(current_val, int):
            try: return int(float(str_value))
            except ValueError: return current_val
        elif isinstance(current_val, float):
            try: return float(str_value)
            except ValueError: return current_val
        return str_value

    @classmethod
    def update_model(cls, model_name: str):
        """Legacy method for backward compatibility."""
        return cls.update_config({"OLLAMA_LLM_MODEL": model_name})

if Config.LLM_PROVIDER == "openai" and not Config.OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")

# Initialize the settings DB and load persisted values
Config.init_db()
