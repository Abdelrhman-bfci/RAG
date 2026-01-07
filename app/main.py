```python
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

from app.ingestion.pdf_ingest import ingest_pdfs
from app.ingestion.db_ingest import ingest_database
from app.ingestion.web_ingest import ingest_websites
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

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str

class ModelUpdateRequest(BaseModel):
    model: str

# --- Endpoints ---

@app.get("/models/ollama")
async def list_ollama_models():
    """List available Ollama models."""
    from app.config import Config
    return {"models": Config.get_ollama_models(), "current": Config.OLLAMA_LLM_MODEL}

@app.post("/config/update_model")
async def update_llm_model(request: ModelUpdateRequest):
    """Update the current LLM model used for generation."""
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

    result = answer_question(request.question)
    
    if "error" in result:
        return {"question": request.question, "answer": result["error"] + status_msg}

    return {
        "question": request.question, 
        "answer": result["answer"] + status_msg,
        "sources": result["sources"],
        "performance": result["performance"]
    }

@app.get("/ask/stream")
async def ask_question_stream(question: str):
    """
    Answer a question using the RAG system with streaming.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    return StreamingResponse(
        stream_answer(question), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/ingest/pdf/stream")
async def stream_pdf_ingestion(fresh: bool = False):
    """
    Stream PDF ingestion progress in real-time.
    - Set fresh=true to clear the index first.
    """
    return StreamingResponse(
        ingest_pdfs(force_fresh=fresh), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/ingest/pdf")
async def trigger_pdf_ingestion(background_tasks: BackgroundTasks):
    """
    Trigger PDF ingestion in the background.
    """
    def run_ingest():
        # Consume the generator in background
        for _ in ingest_pdfs():
            pass
            
    background_tasks.add_task(run_ingest)
    return {"status": "accepted", "message": "PDF ingestion started in background"}

@app.get("/ingest/db/stream")
@app.post("/ingest/db")
async def stream_db_ingestion():
    """
    Stream Database ingestion progress in real-time.
    Ingests all tables configured in Config.INGEST_TABLES.
    """
    return StreamingResponse(
        ingest_database(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/ingest/db/status")
async def get_database_status():
    """
    Get the status of ingested tables and their relations.
    """
    from app.ingestion.db_ingest import get_db_status
    return get_db_status()

@app.get("/ingest/web/stream")
async def stream_web_ingestion(fresh: bool = False):
    """
    Stream Website ingestion progress in real-time.
    - Set fresh=true to clear tracking for configured links.
    """
    return StreamingResponse(
        ingest_websites(force_fresh=fresh), 
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

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=80, reload=True)
```
