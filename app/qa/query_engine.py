"""
Query engine – cloned from chatbot/chat.py gold standard.

Handles:
  1. Query rephrasing with conversation history context.
  2. Hybrid retrieval (MMR + optional re-ranking via CrossEncoder).
  3. Numbered reference formatting ([1], [2]) for the LLM.
  4. Post-generation citation extraction linking text → retrieved chunks.
  5. Structured result packaging for the client API.
"""

import os
import re
import json
import time
import sqlite3
import traceback
from typing import List, Dict, Optional, Generator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from sentence_transformers import CrossEncoder

from app.config import Config
from app.vectorstore.factory import VectorStoreFactory
from app.qa.prompt_template import get_chat_prompt, get_rephrase_prompt

# ---------------------------------------------------------------------------
# Singleton re-ranker (shared across requests)
# ---------------------------------------------------------------------------
_shared_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> Optional[CrossEncoder]:
    global _shared_reranker
    if Config.USE_RERANKER and _shared_reranker is None:
        print(f"   [QueryEngine] Loading Re-Ranker '{Config.RERANKER_MODEL}'...")
        _shared_reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
    return _shared_reranker


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(deep_thinking: bool = False):
    """Instantiate the correct LLM based on Config.LLM_PROVIDER and mode."""
    provider = Config.LLM_PROVIDER

    if provider == "ollama":
        num_ctx = Config.DOC_LLM_NUM_CTX if deep_thinking else Config.OLLAMA_CONTEXT_WINDOW
        num_predict = -1 if deep_thinking else Config.CHAT_LLM_NUM_PREDICT
        return ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0,
            num_ctx=num_ctx,
            num_predict=num_predict,
            stop=[],
            repeat_penalty=1.1,
        )

    if provider == "vllm":
        return ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0,
            api_key="none",
            max_tokens=8192,
        )

    if provider == "openai":
        return ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=8192,
        )

    if provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0,
            max_output_tokens=8192,
        )

    # Fallback
    return ChatOllama(
        model=Config.OLLAMA_LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
        num_predict=-1,
        stop=[],
        repeat_penalty=1.1,
    )


# ---------------------------------------------------------------------------
# URL resolution from crawler DB  (matches chatbot/chat.py)
# ---------------------------------------------------------------------------

def _get_url_from_source(source: str) -> str:
    """Look up the original URL for a crawled file via the crawler SQLite DB."""
    if not os.path.exists(Config.CRAWLER_DB):
        return "#"
    try:
        conn = sqlite3.connect(Config.CRAWLER_DB, check_same_thread=False)
        cursor = conn.cursor()
        basename = os.path.basename(source)
        cursor.execute(
            "SELECT url FROM pages WHERE filename = ? OR filename = ? LIMIT 1",
            (source, basename),
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "#"
    except Exception:
        return "#"


# ---------------------------------------------------------------------------
# Document formatter  (numbered references for citation)
# ---------------------------------------------------------------------------

def format_documents_numbered(docs: List[Document]) -> str:
    """
    Format retrieved documents with numbered references so the LLM can cite
    them as [1], [2], etc.

    Output per chunk:
        [1]
        CONTENT: <text>
        METADATA: Source: <basename>, Page: <page>, URL: <url>

    Cloned from chatbot/chat.py → _format_documents, enhanced with numbering.
    """
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        content = doc.page_content.replace("search_document: ", "")
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "?")
        url = _get_url_from_source(source) if not source.startswith("http") else source

        parts.append(
            f"[{idx}]\n"
            f"CONTENT: {content}\n"
            f"METADATA: Source: {os.path.basename(source)}, Page: {page}, URL: {url if url else '#'}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Re-ranking filter  (matches chatbot/chat.py → _filter_documents)
# ---------------------------------------------------------------------------

def filter_documents(docs: List[Document], query: str) -> List[Document]:
    """Re-rank using CrossEncoder and return top-K above threshold."""
    if not docs:
        return []

    reranker = _get_reranker()
    if reranker is None:
        return docs[: Config.LLM_K_FINAL]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    candidates: list[Document] = []
    for i, doc in enumerate(docs):
        score = float(scores[i])
        if score < Config.RERANKER_THRESHOLD:
            continue
        doc.metadata["score"] = score
        doc.metadata["type"] = "reranked"
        candidates.append(doc)

    candidates.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
    return candidates[: Config.LLM_K_FINAL]


# ---------------------------------------------------------------------------
# Hybrid Retriever  (MMR + session memory + re-ranking)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Matches chatbot rag.py's MergerRetriever + chat.py's _filter_documents flow:
      1. Broad MMR search from the vectorstore.
      2. Merge in session-history docs if available.
      3. Re-rank everything and keep top-K.
    """

    def __init__(self, vectorstore, history_retriever=None):
        self.vectorstore = vectorstore
        self.history_retriever = history_retriever

    def invoke(self, query: str) -> List[Document]:
        if not isinstance(query, str):
            query = str(query)

        k_search = 100 if Config.USE_RERANKER else 50
        try:
            initial_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=k_search, fetch_k=k_search * 2, lambda_mult=0.5
            )
        except Exception:
            initial_docs = self.vectorstore.similarity_search(query, k=k_search)

        if self.history_retriever:
            try:
                hist_docs = self.history_retriever.invoke(query)
                for d in hist_docs:
                    d.metadata["source"] = "Chat History"
                initial_docs.extend(hist_docs)
            except Exception:
                pass

        for doc in initial_docs:
            if not isinstance(doc.metadata, dict):
                doc.metadata = {"source": "Unknown", "repair_flag": True}

        if not initial_docs:
            return []

        return filter_documents(initial_docs, query)


# ---------------------------------------------------------------------------
# Citation extraction  (post-LLM: link answer text → source list)
# ---------------------------------------------------------------------------

def extract_cited_sources(answer: str, all_sources: List[str]) -> List[str]:
    """
    Parse [1], [Source.pdf], [Source (Page X)] from the LLM answer and return
    only the sources that were actually cited.
    Falls back to all sources if none matched (defensive).
    """
    citations = re.findall(r"\[(.*?)\]", answer)
    cited_names: set[str] = set()
    for c in citations:
        part = c.split(",")[0].strip().split("(")[0].strip()
        cited_names.add(part.lower())

    matched: list[str] = []
    for src in all_sources:
        basename = os.path.basename(src).lower()
        if basename in cited_names or src.lower() in cited_names:
            matched.append(src)

    return sorted(set(matched)) if matched else sorted(set(all_sources))


# ---------------------------------------------------------------------------
# High-level answer functions  (sync + streaming)
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    deep_thinking: bool = False,
    is_continuation: bool = False,
    last_answer: str = "",
    conversation_history: list = None,
) -> dict:
    """
    Synchronous entry point.  Returns:
      { answer, sources, performance, used_docs, session_id? }
    """
    start = time.time()
    try:
        chain, retriever, llm = _build_chain(
            deep_thinking, is_continuation, last_answer, conversation_history
        )

        docs = retriever.invoke(question)
        sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
        used_docs = [
            {
                "source": d.metadata.get("source", "Unknown"),
                "content": d.page_content,
                "page": d.metadata.get("page", 0),
                "score": d.metadata.get("score"),
            }
            for d in docs
        ]

        llm_start = time.time()
        answer = chain.invoke(question)
        end = time.time()

        relevant = extract_cited_sources(answer, sources)
        perf = f"Total {end - start:.1f}s | LLM {end - llm_start:.1f}s"

        return {
            "answer": answer,
            "sources": relevant,
            "performance": perf,
            "used_docs": used_docs,
        }
    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        return {"error": f"Unexpected error: {e}"}


def stream_answer(
    question: str,
    deep_thinking: bool = False,
    is_continuation: bool = False,
    last_answer: str = "",
    conversation_history: list = None,
) -> Generator[str, None, None]:
    """
    Streaming entry point.
    Yields newline-delimited JSON: sources → chunks → metadata → done.
    """
    start = time.time()
    try:
        chain, retriever, llm = _build_chain(
            deep_thinking, is_continuation, last_answer, conversation_history
        )

        ret_start = time.time()
        docs = retriever.invoke(question)
        ret_end = time.time()

        sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
        used_docs = [
            {
                "source": d.metadata.get("source", "Unknown"),
                "content": d.page_content,
                "page": d.metadata.get("page", 0),
                "score": d.metadata.get("score"),
            }
            for d in docs
        ]

        yield json.dumps({"type": "sources", "sources": sources}) + "\n"

        llm_start = time.time()
        accumulated = ""
        for chunk in chain.stream(question):
            accumulated += chunk
            yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
        end = time.time()

        relevant = extract_cited_sources(accumulated, sources)
        estimated_tokens = len(accumulated) // 4

        current_model = (
            Config.OLLAMA_LLM_MODEL
            if Config.LLM_PROVIDER == "ollama"
            else Config.LLM_PROVIDER.upper()
        )

        yield json.dumps(
            {
                "type": "metadata",
                "sources": relevant,
                "performance": {
                    "total_time": round(end - start, 2),
                    "llm_time": round(end - llm_start, 2),
                    "retrieval_time": round(ret_end - ret_start, 2),
                },
                "tokens": estimated_tokens,
                "model": current_model,
                "chunks": used_docs,
                "search_query": question,
                "history": conversation_history or [],
            }
        ) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"


# ---------------------------------------------------------------------------
# Internal chain builder
# ---------------------------------------------------------------------------

def _build_chain(deep_thinking, is_continuation, last_answer, conversation_history):
    """
    Constructs the LCEL chain, retriever, and LLM in one place.
    """
    store = VectorStoreFactory.get_instance()
    vectorstore = store.get_vectorstore()

    # Session memory
    history_retriever = None
    if conversation_history:
        try:
            from app.services.session_memory import LanceDBSessionMemory

            mem = LanceDBSessionMemory(
                store.embeddings,
                k=Config.LLM_HISTORY_K,
                score_threshold=Config.LLM_HISTORY_SCORE_THRESHOLD,
            )
            items = []
            for msg in conversation_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown").capitalize()
                    content = msg.get("content", "")
                    items.append(f"{role}: {content}")
            mem.add_history(items)
            history_retriever = mem.get_retriever()
        except Exception as e:
            print(f"Session memory init error: {e}")

    retriever = HybridRetriever(vectorstore, history_retriever=history_retriever)
    llm = get_llm(deep_thinking)
    prompt = get_chat_prompt(deep_thinking)
    rephrase_prompt = get_rephrase_prompt()

    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    def _history_text() -> str:
        if not conversation_history:
            return ""
        from app.services.chat_session import format_history_for_prompt
        return format_history_for_prompt(conversation_history)

    def _rephrase(q):
        h = _history_text()
        if not h.strip():
            return q
        try:
            return rephrase_chain.invoke({"history": h, "question": q})
        except Exception:
            return q

    def _process_question(q):
        if is_continuation and last_answer:
            return f"--- CONTINUATION ---\nContinue from:\n{last_answer}\n\nQuestion: {q}"
        return q

    chain = (
        {
            "rephrased_query": RunnableLambda(_rephrase),
            "original_question": RunnablePassthrough(),
        }
        | {
            "context": RunnableLambda(
                lambda x: retriever.invoke(x["rephrased_query"] if isinstance(x, dict) else str(x))
            )
            | format_documents_numbered,
            "history": RunnableLambda(lambda _: _history_text()),
            "question": RunnableLambda(
                lambda x: _process_question(x["rephrased_query"] if isinstance(x, dict) else str(x))
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, llm
