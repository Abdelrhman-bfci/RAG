from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json
import os

from app.ingestion.pdf_ingest import ingest_pdfs
from app.ingestion.db_ingest import ingest_database
from app.ingestion.web_ingest import ingest_websites
from app.qa.rag_chain import answer_question

app = FastAPI(title="RAG System API", description="PDF & SQL RAG with Strict Context Control")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str

class DBIngestRequest(BaseModel):
    query: Optional[str] = None
    table_name: Optional[str] = None

# --- Endpoints ---

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

    answer = answer_question(request.question)
    return {"question": request.question, "answer": answer + status_msg}

@app.get("/ingest/pdf/stream")
async def stream_pdf_ingestion(fresh: bool = False):
    """
    Stream PDF ingestion progress in real-time.
    - Set fresh=true to clear the index first.
    """
    return StreamingResponse(ingest_pdfs(force_fresh=fresh), media_type="text/plain")

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

@app.post("/ingest/db")
async def trigger_db_ingestion(request: DBIngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger Database ingestion in the background.
    """
    if not request.query and not request.table_name:
        raise HTTPException(status_code=400, detail="Must provide either 'query' or 'table_name'")
        
    background_tasks.add_task(ingest_database, request.query, request.table_name)
    return {"status": "accepted", "message": "Database ingestion started in background"}

@app.get("/ingest/web/stream")
async def stream_web_ingestion(fresh: bool = False):
    """
    Stream Website ingestion progress in real-time.
    - Set fresh=true to clear tracking for configured links.
    """
    return StreamingResponse(ingest_websites(force_fresh=fresh), media_type="text/plain")

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
def read_root():
    return {"message": "RAG System is running. Use /docs to see API."}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=80, reload=True)
