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
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-large"
    LLM_MODEL = "gpt-4-turbo-preview" # Or standard gpt-4 or gpt-3.5-turbo

if not Config.OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")
