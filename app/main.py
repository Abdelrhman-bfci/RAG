from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json
import os
import shutil
import asyncio

from app.ingestion.document_ingest import ingest_documents, TRACKING_FILE as DOC_TRACKING
from app.ingestion.db_ingest import ingest_database, DB_TRACKING_FILE as DB_TRACKING
from app.ingestion.web_ingest import ingest_websites, TRACKING_FILE as WEB_TRACKING
from app.ingestion.offline_web_ingest import ingest_offline_downloads
from app.services.crawler_service import CrawlerService
from app.config import Config # Assuming Config is needed for RESOURCE_DIR

# Gold-standard client API router (v2 endpoints with structured citations)

from typing import List, Optional
from fastapi import APIRouter, Form
from app.qa.query_engine import answer_question, stream_answer
from app.services.chat_session import (
    create_session,
    get_session_history,
    add_message,
    session_exists,
    get_all_sessions,
    delete_session as delete_session_svc,
    clear_all_sessions,
)


client_api_router = APIRouter(tags=["Client API – Gold Standard"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./static/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ======================================================================
# Pydantic schemas
# ======================================================================

class AskRequest(BaseModel):
    question: str
    deep_thinking: bool = False
    is_continuation: bool = False
    last_answer: str = ""
    session_id: Optional[str] = None


class SourceRef(BaseModel):
    """A single structured source reference returned to the client."""
    name: str
    url: str = "#"
    page: Optional[str] = None
    score: Optional[float] = None
    content_preview: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceRef]
    performance: str
    session_id: str


# ======================================================================
# 1. SYNCHRONOUS ASK  (returns full answer + structured sources array)
# ======================================================================

@client_api_router.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """
    Non-streaming answer endpoint.
    Returns the answer text plus a structured `sources` array with name,
    URL, page, relevance score, and a content preview for each reference.
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    session_id = request.session_id
    if not session_id or not session_exists(session_id):
        session_id = create_session()

    history = get_session_history(session_id, limit=10)
    add_message(session_id, "user", request.question)

    result = answer_question(
        request.question,
        deep_thinking=request.deep_thinking,
        is_continuation=request.is_continuation,
        last_answer=request.last_answer,
        conversation_history=history,
    )

    if "error" in result:
        raise HTTPException(500, result["error"])

    add_message(session_id, "assistant", result["answer"])

    structured_sources = _build_source_refs(result.get("used_docs", []))

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        sources=structured_sources,
        performance=result.get("performance", ""),
        session_id=session_id,
    )


# ======================================================================
# 2. STREAMING ASK  (SSE: sources → chunks → metadata → done)
# ======================================================================

@client_api_router.get("/ask/stream")
async def ask_stream_endpoint(
    question: str,
    deep_thinking: bool = False,
    is_continuation: bool = False,
    last_answer: str = "",
    session_id: Optional[str] = None,
):
    """
    Server-Sent Events streaming endpoint.
    Protocol (newline-delimited JSON):
      {"type":"sources",  "sources": [...]}
      {"type":"chunk",    "content": "..."}
      {"type":"metadata", "sources":[...], "performance":{...}, ...}
      {"type":"done"}
    """
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty")

    if not session_id or not session_exists(session_id):
        session_id = create_session()

    history = get_session_history(session_id, limit=10)
    add_message(session_id, "user", question)

    async def _generate():
        accumulated = ""
        for chunk_json in stream_answer(
            question,
            deep_thinking=deep_thinking,
            is_continuation=is_continuation,
            last_answer=last_answer,
            conversation_history=history,
        ):
            try:
                data = json.loads(chunk_json)
                if data.get("type") == "chunk":
                    accumulated += data.get("content", "")
            except Exception:
                pass
            yield chunk_json

        if accumulated:
            add_message(session_id, "assistant", accumulated)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id,
        },
    )


# ======================================================================
# 3. CHAT WITH FILE UPLOAD  (mirrors chatbot/api.py /api/chat)
# ======================================================================

@client_api_router.post("/chat")
async def chat_with_files(
    session_id: str = Form(...),
    message: str = Form(...),
    files: List[UploadFile] = File(None),
):
    """
    Multipart chat endpoint that accepts file uploads.
    Files are immediately ingested into the session RAG context before
    the query is answered.  Streams the response back.
    """
    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")

    history = get_session_history(session_id, limit=10)
    add_message(session_id, "user", message)

    file_paths: list[str] = []
    if files:
        from app.ingestion.ingestion import process_documents
        from app.vectorstore.factory import VectorStoreFactory

        store = VectorStoreFactory.get_instance()

        for f in files:
            try:
                safe_name = f"{session_id}_{f.filename.replace(' ', '_')}"
                fpath = os.path.join(UPLOAD_DIR, safe_name)
                with open(fpath, "wb") as buf:
                    shutil.copyfileobj(f.file, buf)

                from app.ingestion.ingestion import get_loader

                loader = get_loader(fpath)
                if loader:
                    raw = loader.load()
                    enriched = process_documents(raw)
                    store.add_documents(enriched)
                    file_paths.append(fpath)
            except Exception as e:
                print(f"File upload error ({f.filename}): {e}")

    async def _generate():
        accumulated = ""
        for chunk_json in stream_answer(
            message,
            conversation_history=history,
        ):
            try:
                data = json.loads(chunk_json)
                if data.get("type") == "chunk":
                    accumulated += data.get("content", "")
            except Exception:
                pass
            yield chunk_json

        if accumulated:
            add_message(session_id, "assistant", accumulated)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id,
        },
    )


# ======================================================================
# 4. SESSION MANAGEMENT  (mirrors chatbot/api.py session endpoints)
# ======================================================================

@client_api_router.post("/session/new")
async def create_new_session_endpoint():
    sid = create_session()
    return {"session_id": sid}


@client_api_router.get("/session/{session_id}/history")
async def get_history_endpoint(session_id: str, limit: int = 50):
    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "history": get_session_history(session_id, limit=limit),
    }


@client_api_router.get("/sessions")
async def list_sessions_endpoint():
    return {"sessions": get_all_sessions()}


@client_api_router.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")
    delete_session_svc(session_id)
    return {"status": "deleted"}


@client_api_router.post("/sessions/clear")
async def clear_all_sessions_endpoint():
    clear_all_sessions()
    return {"status": "all sessions cleared"}


# ======================================================================
# Helpers
# ======================================================================

def _build_source_refs(used_docs: list) -> List[SourceRef]:
    """
    Convert raw used_docs dicts into structured SourceRef objects,
    resolving URLs from the crawler DB.
    """
    from app.qa.query_engine import _get_url_from_source

    seen: set[str] = set()
    refs: list[SourceRef] = []

    for doc in used_docs:
        src = doc.get("source", "Unknown")
        basename = os.path.basename(src)
        if basename in seen:
            continue
        seen.add(basename)

        url = _get_url_from_source(src) if not src.startswith("http") else src
        refs.append(
            SourceRef(
                name=basename,
                url=url if url and url != "#" else f"/files/{basename}",
                page=str(doc.get("page", "")),
                score=doc.get("score"),
                content_preview=doc.get("content", "")[:200],
            )
        )

    return refs

app = FastAPI(title="RAG System API", description="PDF & SQL RAG with Strict Context Control")

# Mount the gold-standard client API at /v2 alongside existing endpoints
app.include_router(client_api_router, prefix="/v2")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files ---
app.mount("/client", StaticFiles(directory="client"), name="client")

# Ensure resource directory exists
if not os.path.exists(Config.RESOURCE_DIR):
    os.makedirs(Config.RESOURCE_DIR)
app.mount("/files", StaticFiles(directory=Config.RESOURCE_DIR), name="files")

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str
    deep_thinking: bool = False
    is_continuation: bool = False
    last_answer: str = ""
    session_id: Optional[str] = None

class ModelUpdateRequest(BaseModel):
    model: str

class LoginRequest(BaseModel):
    username: str
    password: str

# --- Endpoints ---

@app.post("/auth/login")
async def login(request: LoginRequest):
    from app.config import Config
    if request.username == Config.ADMIN_USERNAME and request.password == Config.ADMIN_PASSWORD:
        return {"status": "success", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to the chat interface."""
    with open("client/index.html", "r") as f:
        return f.read()

@app.get("/models/ollama")
async def list_ollama_models(base_url: str = None):
    """List available Ollama models."""
    from app.config import Config
    return {"models": Config.get_ollama_models(base_url=base_url), "current": Config.OLLAMA_LLM_MODEL}

@app.get("/models/vllm")
async def list_vllm_models(base_url: str = None):
    """List available vLLM models."""
    from app.config import Config
    return {"models": Config.get_vllm_models(base_url=base_url), "current": Config.VLLM_MODEL}

@app.get("/models/openai")
async def list_openai_models(api_key: str = None, base_url: str = None):
    """List available OpenAI models."""
    from app.config import Config
    return {"models": Config.get_openai_models(api_key=api_key, base_url=base_url), "current": Config.OPENAI_LLM_MODEL}

@app.get("/models/gemini")
async def list_gemini_models():
    """List available Gemini models."""
    from app.config import Config
    return {"models": Config.get_gemini_models(), "current": Config.GEMINI_MODEL}

@app.get("/config/current")
async def get_current_config():
    """Get current provider and settings."""
    from app.config import Config
    return {
        "provider": Config.LLM_PROVIDER,
        "embedding_provider": Config.EMBEDDING_PROVIDER,
        "ollama": {
            "model": Config.OLLAMA_LLM_MODEL,
            "embedding_model": Config.OLLAMA_EMBEDDING_MODEL,
            "base_url": Config.OLLAMA_BASE_URL
        },
        "vllm": {
            "model": Config.VLLM_MODEL,
            "embedding_model": Config.VLLM_EMBEDDING_MODEL,
            "base_url": Config.VLLM_BASE_URL
        },
        "openai": {
            "model": Config.OPENAI_LLM_MODEL,
            "embedding_model": Config.OPENAI_EMBEDDING_MODEL,
            "api_key": Config.OPENAI_API_KEY,
            "base_url": Config.OPENAI_BASE_URL
        },
        "gemini": {
            "model": Config.GEMINI_MODEL,
            "api_key": Config.GEMINI_API_KEY
        },
        "vector_store_provider": Config.VECTOR_STORE_PROVIDER,
        "vector_search_weight": Config.VECTOR_SEARCH_WEIGHT,
        "show_summary_chunks": Config.SHOW_SUMMARY_CHUNKS
    }

@app.post("/config/update")
async def update_configuration(updates: dict):
    """Update general configuration."""
    from app.config import Config
    Config.update_config(updates)
    return {"status": "success"}

@app.post("/config/update_model")
async def update_llm_model(request: ModelUpdateRequest):
    """Legacy endpoint for backward compatibility."""
    from app.config import Config
    Config.update_model(request.model)
    return {"status": "success", "model": Config.OLLAMA_LLM_MODEL}

# --- Document Summarization Endpoint ---

@app.post("/summarize/stream")
async def summarize_document_stream_endpoint(file: UploadFile, include_chunks: Optional[bool] = None):
    """
    Summarize an uploaded document with streaming progress.
    Supports PDF, DOCX, XLSX, CSV, TXT files.
    """
    import tempfile
    import shutil
    from app.services.document_summarizer import summarize_document_stream
    from app.config import Config
    
    # Use config default if parameter not provided
    if include_chunks is None:
        include_chunks = Config.SHOW_SUMMARY_CHUNKS
    
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {str(e)}")
    
    async def generate():
        """Generator for streaming summarization progress."""
        try:
            for update in summarize_document_stream(
                tmp_path, 
                chunk_size=Config.SUMMARY_CHUNK_SIZE,
                include_chunks=include_chunks
            ):
                yield json.dumps(update) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/ingest/document/stream")
async def stream_document_ingestion(fresh: bool = False):
    """
    Stream document ingestion progress in real-time.
    """
    return StreamingResponse(
        ingest_documents(force_fresh=fresh), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/resources")
async def list_resources():
    """
    List all ingested resources by type.
    """
    # Get actual stats from vector store to ensure consistency
    try:
        from app.vectorstore.factory import VectorStoreFactory
        store = VectorStoreFactory.get_instance()
        stats = store.get_index_stats()
        index_sources = stats.get("sources", {})
    except:
        index_sources = {}

    resources = {
        "documents": [],
        "websites": [],
        "databases": [],
        "crawled": []
    }

    # Documents
    if os.path.exists(DOC_TRACKING):
        try:
            with open(DOC_TRACKING, "r") as f:
                tracking = json.load(f)
                resources["documents"] = [os.path.basename(path) for path in tracking.keys()]
        except: pass
    
    # Fallback: find documents in index but not in tracking
    for src in index_sources:
        if not src.startswith("http") and src not in resources["documents"] and src not in resources["websites"] and src not in resources["databases"]:
             # If it looks like a table name from Config, it might be database
            from app.config import Config
            if src in (Config.INGEST_TABLES or []):
                if src not in resources["databases"]:
                    resources["databases"].append(src)
            elif "." in src: # Likely a file if it has an extension
                resources["documents"].append(src)
    
    # Websites
    if os.path.exists(WEB_TRACKING):
        try:
            with open(WEB_TRACKING, "r") as f:
                tracking = json.load(f)
                resources["websites"] = list(tracking.keys())
        except: pass
    
    # Fallback for websites from index
    for src in index_sources:
        if src.startswith("http") and src not in resources["websites"]:
            resources["websites"].append(src)
        
    # Databases
    if os.path.exists(DB_TRACKING):
        try:
            with open(DB_TRACKING, "r") as f:
                resources["databases"] = json.load(f)
        except: pass
    
    # Crawled files (offline ingestion) - read from database
    try:
        from app.ingestion.offline_web_ingest import get_ingested_files
        ingested = get_ingested_files()
        resources["crawled"] = ingested
    except:
        pass

    # Final deduplication
    for k in resources:
        resources[k] = list(set(resources[k]))

    return resources

@app.delete("/resources/{res_type}/{name:path}")
async def delete_resource(res_type: str, name: str):
    """
    Delete a specific resource.
    """
    from app.vectorstore.factory import VectorStoreFactory
    store = VectorStoreFactory.get_instance()
    
    if res_type == "documents":
        # Find path in tracking
        path_to_remove = None
        tracking = {}
        if os.path.exists(DOC_TRACKING):
            with open(DOC_TRACKING, "r") as f:
                tracking = json.load(f)
            for path in tracking.keys():
                if os.path.basename(path) == name:
                    path_to_remove = path
                    break
        
        if path_to_remove:
            # Delete file
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
            
            # Remove from tracking
            del tracking[path_to_remove]
            with open(DOC_TRACKING, "w") as f:
                json.dump(tracking, f, indent=4)
            
            # Remove from Vector Store
            store.delete_source(path_to_remove)
            return {"status": "success", "message": f"Deleted document {name}"}
            
    elif res_type == "databases":
        # Handle database table deletion
        tracking = []
        if os.path.exists(DB_TRACKING):
            with open(DB_TRACKING, "r") as f:
                try:
                    tracking = json.load(f)
                except: tracking = []
        
        if name in tracking:
            # Delete from tracking
            tracking.remove(name)
            with open(DB_TRACKING, "w") as f:
                json.dump(tracking, f)
            
            # Delete from Vector Store
            # Metadata source format defined in db_ingest.py: f"Table: {table}"
            store.delete_source(f"Table: {name}")
            
            return {"status": "success", "message": f"Deleted Table {name}"}
        else:
             raise HTTPException(status_code=404, detail="Table not found in tracking")
            
    elif res_type == "websites":
        tracking = {}
        if os.path.exists(WEB_TRACKING):
            with open(WEB_TRACKING, "r") as f:
                tracking = json.load(f)
            
            if name in tracking:
                del tracking[name]
                with open(WEB_TRACKING, "w") as f:
                    json.dump(tracking, f, indent=4)
                
                # Remove from Vector Store
                store.delete_source(name)
                return {"status": "success", "message": f"Deleted website {name}"}

    elif res_type == "databases":
        if name in Config.INGEST_TABLES:
            new_tables = [t for t in Config.INGEST_TABLES if t != name]
            Config.update_config({"INGEST_TABLES": ",".join(new_tables)})
            
            # Remove from Vector Store
            store.delete_source(name)
            return {"status": "success", "message": f"Deleted table {name}"}

    raise HTTPException(status_code=404, detail="Resource not found")

@app.post("/resources/reset")
async def reset_all_resources():
    """
    Wipe all ingested data and reset the system.
    """
    # 1. Clear PDFs and uploaded resources recursively
    # dirs_to_clear = [Config.RESOURCE_DIR, Config.DOWNLOAD_FOLDER]
    # for target_dir in dirs_to_clear:
    #     if target_dir and os.path.exists(target_dir):
    #         print(f"Clearing directory: {target_dir}")
    #         for item in os.listdir(target_dir):
    #             item_path = os.path.join(target_dir, item)
    #             try:
    #                 if os.path.isfile(item_path) or os.path.islink(item_path):
    #                     os.unlink(item_path)
    #                 elif os.path.isdir(item_path):
    #                     shutil.rmtree(item_path)
    #             except Exception as e:
    #                 print(f"Error deleting {item_path}: {e}")
    
    # 2. Clear Tracking JSON Files (including /tmp)
    tracking_files = [
        DOC_TRACKING, WEB_TRACKING, DB_TRACKING, 
        "ingestion_status.json", "web_ingestion_status.json", "db_ingestion_status.json",
        "/tmp/ingested_files_v2.json", "/tmp/ingestion_status_v2.json",
        "web_ingested_links.json"
    ]
    for tracking in tracking_files:
        if os.path.exists(tracking):
            try:
                os.remove(tracking)
                print(f"Removed tracking file: {tracking}")
            except: pass
            
    # 3. Reset Config Tables
    Config.update_config({"INGEST_TABLES": ""})
    
    # 4. Clear Vector Store (Thorough cleanup)
    from app.vectorstore.factory import VectorStoreFactory
    try:
        # Wipe current used store instance
        store = VectorStoreFactory.get_instance()
        store.clear_all()
        print("Cleared current vector store via provider.")
    except Exception as e:
        print(f"Error clearing current vector store: {e}")

    # Wipe ALL potential physical storage folders dynamically (chroma_db*)
    try:
        current_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        for folder in current_dirs:
            if folder.startswith(("chroma_db")):
                try:
                    shutil.rmtree(folder)
                    print(f"Purged storage folder: {folder}")
                except PermissionError as pe:
                    print(f"Skipping locked folder '{folder}' (PermissionError — owned by another user): {pe}")
                except Exception as e:
                    print(f"Failed to purge {folder}: {e}")
        
        # Also check the VECTOR_DB_PATH specifically
        if os.path.exists(Config.VECTOR_DB_PATH) and os.path.isdir(Config.VECTOR_DB_PATH):
            try:
                shutil.rmtree(Config.VECTOR_DB_PATH)
                print(f"Purged VECTOR_DB_PATH: {Config.VECTOR_DB_PATH}")
            except PermissionError as pe:
                print(f"Skipping locked VECTOR_DB_PATH '{Config.VECTOR_DB_PATH}' (PermissionError): {pe}")
    except Exception as e:
        print(f"Error during folder purging: {e}")
    
    # 5. Reset Ingested Files SQL Table & Delete Crawler DB
    try:
        from app.ingestion.offline_web_ingest import reset_crawled_ingestion_status
        reset_crawled_ingestion_status(full_wipe=True)
        print("Wiped ingested_files tracking table via SQL.")
        
        # Hard wipe of the database file itself to clear schema/WAL files
        if os.path.exists(Config.CRAWLER_DB):
            try:
                os.remove(Config.CRAWLER_DB)
                print(f"Deleted crawler database file: {Config.CRAWLER_DB}")
            except Exception as db_e:
                print(f"Could not delete database file: {db_e}")
                
        # 6. Clear Chat Sessions
        from app.services.chat_session import clear_all_sessions
        clear_all_sessions()
        
    except Exception as e:
        print(f"Error resetting crawled status or sessions: {e}")
        
    return {
        "status": "success", 
        "message": "All resources reset successfully. You can now re-ingest data fresh."
    }

@app.get("/ingest/db/schemas")
async def list_database_schemas():
    """
    List all available database schemas/databases.
    """
    if not Config.DATABASE_URL:
        raise HTTPException(status_code=400, detail="DATABASE_URL not configured")
    
    try:
        from sqlalchemy import create_engine, inspect
        engine = create_engine(Config.DATABASE_URL)
        inspector = inspect(engine)
        
        # For MySQL/MariaDB, schemas are databases
        schemas = inspector.get_schema_names()
        
        # Filter out system schemas
        filtered_schemas = [s for s in schemas if s not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        
        return {"schemas": filtered_schemas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch schemas: {str(e)}")

@app.get("/ingest/db/tables")
async def list_database_tables(schema: str = ""):
    """
    List all tables in a specific schema.
    """
    if not Config.DATABASE_URL:
        raise HTTPException(status_code=400, detail="DATABASE_URL not configured")
    
    try:
        from sqlalchemy import create_engine, inspect
        engine = create_engine(Config.DATABASE_URL)
        inspector = inspect(engine)
        
        # Get tables for the specified schema
        tables = inspector.get_table_names(schema=schema if schema else None)
        
        return {"tables": tables, "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tables: {str(e)}")

@app.get("/ingest/db/stream")
async def stream_db_ingestion(schema: str = "", tables: str = ""):
    """
    Stream database ingestion progress for selected tables in real-time.
    """
    table_list = [t.strip() for t in tables.split(",") if t.strip()] if tables else None
    
    return StreamingResponse(
        ingest_database(tables=table_list, schema=schema if schema else None),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/crawl/stream")
async def stream_crawler(url: str, depth: int = 2):
    """
    Stream crawler progress in real-time.
    """
    service = CrawlerService()
    return StreamingResponse(
        service.crawl_website(url, depth), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

def _crawl_then_ingest_stream(url: str, depth: int, fresh: bool):
    """Generator: run crawler then offline ingest for one unified stream."""
    from app.services.crawler_service import CrawlerService
    service = CrawlerService()
    for chunk in service.crawl_website(url, depth):
        yield chunk
    yield "\n--- Crawl complete. Starting ingestion ---\n"
    for chunk in ingest_offline_downloads(force_fresh=fresh):
        yield chunk

@app.get("/crawl-and-ingest/stream")
async def stream_crawl_then_ingest(url: str, depth: int = 2, fresh: bool = False):
    """
    Crawl a URL (saves to download folder), then ingest from downloads into the vector store.
    Single stream for both phases. Uses the same vector store as the rest of the app.
    """
    return StreamingResponse(
        _crawl_then_ingest_stream(url, depth, fresh),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/ingest/offline/stream")
async def stream_offline_ingestion(fresh: bool = False):
    """
    Stream offline ingestion progress from downloads folder.
    Wraps the generator in a safe handler so the stream always closes cleanly,
    preventing ERR_INCOMPLETE_CHUNKED_ENCODING on client crashes.
    """
    def safe_stream():
        try:
            for chunk in ingest_offline_downloads(force_fresh=fresh):
                yield chunk
        except Exception as e:
            import traceback
            err_msg = f"\nFATAL ERROR during ingestion: {e}\n{traceback.format_exc()}\n"
            print(err_msg)
            yield err_msg
        finally:
            yield "\n[STREAM COMPLETE]\n"

    return StreamingResponse(
        safe_stream(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/ingest/offline/start")
async def start_offline_ingestion(background_tasks: BackgroundTasks, fresh: bool = False):
    """
    Start offline ingestion in the background.
    """
    def run_ingest():
        # Call with silent=True to avoid yield errors in background
        for _ in ingest_offline_downloads(force_fresh=fresh, silent=True):
            pass
            
    background_tasks.add_task(run_ingest)
    return {"status": "accepted", "message": "Offline ingestion started in background"}

@app.get("/ingest/status")
async def get_ingestion_progress():
    """
    Poll the current ingestion progress from the database.
    """
    from app.ingestion.offline_web_ingest import get_ingestion_status
    return get_ingestion_status()


@app.get("/")
async def get_chat_interface():
    return FileResponse(os.path.join("client", "index.html"))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the RESOURCE_DIR.
    """
    try:
        if not os.path.exists(Config.RESOURCE_DIR):
            os.makedirs(Config.RESOURCE_DIR)
        
        file_path = os.path.join(Config.RESOURCE_DIR, file.filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"message": f"Successfully uploaded {file.filename}", "filepath": file_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to upload file: {str(e)}"})

@app.get("/index/stats")
async def get_index_statistics():
    """
    Get statistics about the Chroma index content.
    """
    try:
        from app.vectorstore.factory import VectorStoreFactory
        store = VectorStoreFactory.get_instance()
        return store.get_index_stats()
    except Exception as e:
        print(f"Error in /index/stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve index statistics", "details": str(e)}
        )

@app.post("/index/sync")
async def sync_resources(background_tasks: BackgroundTasks):
    """
    Trigger ingestion for any resources that are missing from the index.
    """
    return StreamingResponse(
        ingest_documents(force_fresh=False), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/index/preview")
async def preview_index_content(source: str):
    """
    Get content chunks for a specific source from the index.
    """
    from app.vectorstore.factory import VectorStoreFactory
    store = VectorStoreFactory.get_instance()
    return store.get_source_content(source)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=80, reload=True)
