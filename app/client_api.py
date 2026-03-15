"""
Client API layer – the frontend-facing wrapper that ensures every response
includes the answer + a structured list of references.

This module is designed to be mounted into the existing FastAPI app (main.py)
or used as a standalone router.  It clones the chatbot gold-standard's
streaming protocol and metadata packaging.

Integration:
    # In app/main.py, add:
    from app.client_api import router as client_router
    app.include_router(client_router, prefix="/v2")
"""

import json
import os
import time
import shutil
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.config import Config
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

router = APIRouter(tags=["Client API – Gold Standard"])

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

@router.post("/ask", response_model=AskResponse)
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

@router.get("/ask/stream")
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

@router.post("/chat")
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

@router.post("/session/new")
async def create_new_session_endpoint():
    sid = create_session()
    return {"session_id": sid}


@router.get("/session/{session_id}/history")
async def get_history_endpoint(session_id: str, limit: int = 50):
    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "history": get_session_history(session_id, limit=limit),
    }


@router.get("/sessions")
async def list_sessions_endpoint():
    return {"sessions": get_all_sessions()}


@router.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(404, "Session not found")
    delete_session_svc(session_id)
    return {"status": "deleted"}


@router.post("/sessions/clear")
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
