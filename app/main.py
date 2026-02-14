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
from app.qa.rag_chain import answer_question, stream_answer
from app.config import Config # Assuming Config is needed for RESOURCE_DIR

app = FastAPI(title="RAG System API", description="PDF & SQL RAG with Strict Context Control")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Answer a question using the RAG system.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check ingestion status
    status_msg = ""
    if os.path.exists("ingestion_status.json"):
        try:
            with open("ingestion_status.json", "r") as f:
                status = json.load(f)
                if status.get("status") == "running":
                    current = status.get("current_batch", 0)
                    total = status.get("total_batches", 1)
                    eta = status.get("eta_seconds")
                    
                    percent = int((current / total) * 100) if total > 0 else 0
                    eta_str = f"{eta}s" if eta else "calculating..."
                    
                    status_msg = f"\n\n[⚠️ Indexing in progress: {percent}% complete. ETA: {eta_str}. Answer may be incomplete.]"
        except:
            pass

    from app.services.chat_session import (
        create_session, 
        get_session_history, 
        add_message, 
        session_exists
    )
    
    # Create or validate session
    session_id = request.session_id
    if not session_id or not session_exists(session_id):
        session_id = create_session()
    
    # Get conversation history
    history = get_session_history(session_id, limit=10)
    
    # Save user question to session
    add_message(session_id, "user", request.question)

    result = answer_question(
        request.question, 
        deep_thinking=request.deep_thinking,
        is_continuation=request.is_continuation,
        last_answer=request.last_answer,
        conversation_history=history
    )
    
    # Save assistant response
    if "answer" in result:
        add_message(session_id, "assistant", result["answer"])
    
    if "error" in result:
        return {"question": request.question, "answer": result["error"] + status_msg}

    return {
        "question": request.question, 
        "answer": result["answer"] + status_msg,
        "sources": result["sources"],
        "performance": result["performance"],
        "session_id": session_id
    }

@app.get("/ask/stream")
async def ask_question_stream(
    question: str, 
    deep_thinking: bool = False, 
    is_continuation: bool = False, 
    last_answer: str = "",
    session_id: str = None
):
    """
    Answer a question using the RAG system with streaming and conversation history.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    from app.services.chat_session import (
        create_session, 
        get_session_history, 
        add_message, 
        session_exists
    )
    
    # Create or validate session
    if not session_id or not session_exists(session_id):
        session_id = create_session()
    
    # Get conversation history (exclude current question)
    history = get_session_history(session_id, limit=10)
    
    # Save user question to session
    add_message(session_id, "user", question)
    
    async def generate_with_session():
        """Generator that streams response and saves to session."""
        accumulated_answer = ""
        
        # Stream the answer with conversation history
        for chunk_data in stream_answer(
            question, 
            deep_thinking=deep_thinking,
            is_continuation=is_continuation,
            last_answer=last_answer,
            conversation_history=history
        ):
            # Accumulate answer chunks
            try:
                data = json.loads(chunk_data)
                if data.get("type") == "chunk":
                    accumulated_answer += data.get("content", "")
            except:
                pass
            
            yield chunk_data
        
        # Save assistant response to session
        if accumulated_answer:
            add_message(session_id, "assistant", accumulated_answer)
    
    return StreamingResponse(
        generate_with_session(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id  # Return session ID in header
        }
    )

# --- Chat Session Management Endpoints ---

@app.post("/chat/session/new")
async def create_new_session():
    """Create a new chat session."""
    from app.services.chat_session import create_session
    session_id = create_session()
    return {"session_id": session_id}

@app.get("/chat/session/{session_id}/history")
async def get_session_history_endpoint(session_id: str, limit: int = 50):
    """Get conversation history for a session."""
    from app.services.chat_session import get_session_history, session_exists
    
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = get_session_history(session_id, limit=limit)
    return {"session_id": session_id, "history": history}

@app.delete("/chat/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete a chat session and all its messages."""
    from app.services.chat_session import delete_session, session_exists
    
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    delete_session(session_id)
    return {"status": "success", "message": f"Session {session_id} deleted"}

@app.get("/chat/sessions")
async def list_sessions():
    """List all chat sessions."""
    from app.services.chat_session import get_all_sessions
    sessions = get_all_sessions()
    return {"sessions": sessions}

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
    
    # Websites
    if os.path.exists(WEB_TRACKING):
        try:
            with open(WEB_TRACKING, "r") as f:
                tracking = json.load(f)
                resources["websites"] = list(tracking.keys())
        except: pass
        
    # Databases
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
        resources["crawled"] = list(ingested.keys())
    except Exception as e:
        print(f"Error loading crawled files: {e}")
        pass
    
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
    # 1. Clear PDFs
    if os.path.exists(Config.RESOURCE_DIR):
        for f in os.listdir(Config.RESOURCE_DIR):
            path = os.path.join(Config.RESOURCE_DIR, f)
            if os.path.isfile(path):
                os.remove(path)
    
    # 2. Clear Tracking Files
    for tracking in [DOC_TRACKING, WEB_TRACKING, DB_TRACKING, "ingestion_status.json", "web_ingestion_status.json", "db_ingestion_status.json"]:
        if os.path.exists(tracking):
            os.remove(tracking)
            
    # 3. Clear Config Tables
    Config.update_config({"INGEST_TABLES": ""})
    
    # 4. Clear Vector Store
    from app.vectorstore.factory import VectorStoreFactory
    store = VectorStoreFactory.get_instance()
    store.clear_all()
    
    # 5. Reset Crawled Ingestion Status and get count > 0 resources
    from app.ingestion.offline_web_ingest import reset_crawled_ingestion_status
    crawled_reset = reset_crawled_ingestion_status()
        
    return {
        "status": "success", 
        "message": "All resources reset successfully",
        "details": {
            "crawled": crawled_reset
        }
    }

@app.get("/ingest/web/stream")
async def stream_web_ingestion(url: str, depth: int = 10, max_pages: int = -1):
    """
    Stream web ingestion progress for a single URL in real-time.
    Defaults to depth 10 and unlimited pages (-1).
    """
    return StreamingResponse(
        ingest_websites(urls=[url] if url else None, depth=depth, max_pages=max_pages), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/ingest/web")
async def trigger_web_ingestion(background_tasks: BackgroundTasks):
    """
    Trigger Website ingestion in the background.
    """
    def run_ingest():
        for _ in ingest_websites():
            pass
            
    background_tasks.add_task(run_ingest)
    return {"status": "accepted", "message": "Website ingestion started in background"}

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

@app.get("/ingest/offline/stream")
async def stream_offline_ingestion(fresh: bool = False):
    """
    Stream offline ingestion progress from downloads folder.
    """
    return StreamingResponse(
        ingest_offline_downloads(force_fresh=fresh), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


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
    Get statistics about the FAISS index content.
    """
    from app.vectorstore.factory import VectorStoreFactory
    store = VectorStoreFactory.get_instance()
    return store.get_index_stats()

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
