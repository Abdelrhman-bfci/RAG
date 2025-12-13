from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from app.ingestion.pdf_ingest import ingest_pdfs
from app.ingestion.db_ingest import ingest_database
from app.qa.rag_chain import answer_question
import uvicorn

app = FastAPI(title="RAG System API", description="PDF & SQL RAG with Strict Context Control")

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
    
    answer = answer_question(request.question)
    return {"question": request.question, "answer": answer}

@app.post("/ingest/pdf")
async def trigger_pdf_ingestion(background_tasks: BackgroundTasks):
    """
    Trigger PDF ingestion in the background.
    """
    background_tasks.add_task(ingest_pdfs)
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

@app.get("/")
def read_root():
    return {"message": "RAG System is running. Use /docs to see API."}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
