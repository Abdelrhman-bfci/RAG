import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None # Will handle failure in get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from app.config import Config
from app.vectorstore.factory import VectorStoreFactory
from sentence_transformers import CrossEncoder
import time
import traceback
import json

# Global shared reranker singleton
_shared_reranker = None


def get_rag_chain(deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = "", conversation_history: list = None):
    """
    Creates and returns the RAG chain for Question Answering.
    """
    # 0. Defensive Initialization
    llm = None
    prompt = None
    
    print(f"DEBUG: Initializing RAG chain. Provider: {Config.LLM_PROVIDER}")

    # 1. Initialize Vector Store
    store = VectorStoreFactory.get_instance()
    vectorstore = store.get_vectorstore()

    # 1.5 Initialize Session Memory if History Exists
    history_retriever = None
    if conversation_history and len(conversation_history) > 0:
        try:
            from app.services.session_memory import LanceDBSessionMemory
            # store.embeddings should exist if FAISSStore or ChromaStore initializes them
            session_memory = LanceDBSessionMemory(
                store.embeddings, 
                k=Config.LLM_HISTORY_K, 
                score_threshold=Config.LLM_HISTORY_SCORE_THRESHOLD
            )
            
            # Format each pair of user/assistant messages as distinct documents
            history_strings = []
            for msg in conversation_history:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                history_strings.append(f"{role}: {content}")
                
            session_memory.add_history(history_strings)
            history_retriever = session_memory.get_retriever()
            print("DEBUG: Loaded LanceDB Session Memory Retriever")
        except Exception as e:
            print(f"DEBUG Error initializing session memory: {e}")

    # 2. Select Prompt (Matching Octopiai exactly)
    # Master Chat Template
    chat_prompt = ChatPromptTemplate.from_template("""
You are a professional Document Assistant acting as a closed-domain reasoning engine.
        
CORE DIRECTIVE:
You must answer the user's question using ONLY the information provided in the "Context" below. You are a highly intelligent and helpful AI assistant for ASU Engineering Faculty. 
Your goal is to provide accurate, concise, and professional answers based on the provided context.

**Core Rules:**
1. **Always** structure your response into TWO distinct blocks:
   - **CONTENT:** The actual answer to the users question.
   - **METADATA:** Clear citations for all sources used.
2. If the answer is not in the context, state that you don't know rather than hallucinating.
3. Maintain a professional and academic tone.
4. Keep the CONTENT concise and to the point.
5. In METADATA, list the source names and page numbers if available.
6. **Synthesis**: You may combine information from multiple parts of the Context to form a complete answer.
7. **Formatting**: Preserve lists, tables, and data structures from the original text when beneficial for clarity.
8. **Citations**: For every claim you make, you must include a clickable links to all sources and directly after every use.
8.1. Use the Markdown format: `[Source Name](URL)`.
8.2. If a page number is available, include it: `[Source Name (Page X)](URL)`.

**Response Structure Example:**
CONTENT: 
[Your detailed answer here...]

METADATA:
- Source: [Filename/URL], Page: [Number]
- Source: [Filename/URL]

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

Question: {question}
    """)

    # Document Analysis Template (Deep Thinking Mode)
    doc_prompt = ChatPromptTemplate.from_template("""
You are an expert analyst reviewing the provided full documents.
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
    """)

    # Query Rephrase Template
    rephrase_prompt = ChatPromptTemplate.from_template("""
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.
The standalone question must be fully self-contained and understood without the chat history. Do not answer the question, just rephrase it.

Chat History:
{history}

Follow Up Input:
{question}

Standalone Question:
    """)

    prompt = doc_prompt if deep_thinking else chat_prompt

    # 3. Initialize LLM
    if Config.LLM_PROVIDER == "ollama":
        num_ctx = Config.DOC_LLM_NUM_CTX if deep_thinking else Config.OLLAMA_CONTEXT_WINDOW
        num_predict = -1 if deep_thinking else Config.CHAT_LLM_NUM_PREDICT
        
        llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0,
            num_ctx=num_ctx,
            num_predict=num_predict,
            stop=[],
            repeat_penalty=1.1
        )
    elif Config.LLM_PROVIDER == "vllm":
        llm = ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0,
            api_key="none",
            max_tokens=8192
        )
    elif Config.LLM_PROVIDER == "openai":
        llm = ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=8192
        )
    elif Config.LLM_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed. Please run 'pip install langchain-google-genai'")
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0,
            max_output_tokens=8192
        )
    else:
        # Default fallback to Ollama
        print(f"DEBUG: Falling back to Ollama default")
        llm = ChatOllama(
            model=Config.OLLAMA_LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            num_predict=-1,
            stop=[],
            repeat_penalty=1.1
        )
    
    if llm is None:
        print("DEBUG: CRITICAL ERROR - LLM is still None after initialization block")
    else:
        print(f"DEBUG: LLM successfully initialized as {type(llm).__name__}")

    # 3. Initialize Shared Re-Ranker if enabled
    global _shared_reranker
    if Config.USE_RERANKER and _shared_reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"DEBUG: Loading Shared Re-Ranker '{Config.RERANKER_MODEL}'...")
            _shared_reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
        except Exception as e:
            print(f"DEBUG: Failed to load reranker: {e}")
            _shared_reranker = None

    # 3. Initialize Retriever
    # Advanced Hybrid Retrieval matching Octopiai exactly
    class AdvancedHybridRetriever:
        def __init__(self, vectorstore, llm=None, history_retriever=None):
            self.vectorstore = vectorstore
            self.llm = llm # Kept for signature compatibility
            self.history_retriever = history_retriever
            
            global _shared_reranker
            if Config.USE_RERANKER and _shared_reranker is None:
                print(f"DEBUG: Loading Shared Re-Ranker '{Config.RERANKER_MODEL}'...")
                # Lazy load to avoid slowing down startup if disabled
                _shared_reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
            self.reranker = _shared_reranker
            
        def invoke(self, query):
            try:
                # Ensure query is a string
                if not isinstance(query, str):
                    query = str(query)
                
                # 1. Broad MMR search (Initial pool) - matching chatbot's diversity strategy
                k_search = 100 if Config.USE_RERANKER else 50
                try:
                    initial_docs = self.vectorstore.max_marginal_relevance_search(
                        query, k=k_search, fetch_k=k_search*2, lambda_mult=0.5
                    )
                except Exception as e:
                    print(f"DEBUG: Vector store does not support MMR, falling back to similarity search: {e}")
                    initial_docs = self.vectorstore.similarity_search(query, k=k_search)
                
                # Merge in session history if available
                if self.history_retriever:
                    # History retrieval uses simple similarity usually
                    hist_docs = self.history_retriever.invoke(query)
                    if hist_docs:
                        for d in hist_docs:
                            d.metadata["source"] = "Chat History"
                        initial_docs.extend(hist_docs)
                        
            except Exception as e:
                print(f"Retrieval error: {e}")
                return []
            
            # Defensive check: ensure all docs have dictionary metadata
            for doc in initial_docs:
                if not isinstance(doc.metadata, dict):
                    print(f"DEBUG: Repairing malformed metadata for document: {getattr(doc, 'metadata', 'N/A')}")
                    doc.metadata = {"source": "Unknown", "repair_flag": True}
            
            if not initial_docs:
                return []
                
            if Config.USE_RERANKER:
                # 2a. Re-rank using BAAI Cross-Encoder
                pairs = [[query, doc.page_content] for doc in initial_docs]
                scores = self.reranker.predict(pairs)
                
                ranked_candidates = []
                for i, doc in enumerate(initial_docs):
                    score = float(scores[i])
                    if score < Config.RERANKER_THRESHOLD:
                        continue
                    
                    if isinstance(doc.metadata, dict):
                        doc.metadata["score"] = score
                    ranked_candidates.append(doc)
                
                # Sort by highest score - add defensive check for metadata existence
                ranked_candidates.sort(key=lambda x: x.metadata.get("score", 0) if isinstance(x.metadata, dict) else 0, reverse=True)
                
                # Filter by threshold after retrieval as well to ensure quality
                final_docs = [d for d in ranked_candidates if d.metadata.get("score", 0) >= Config.RERANKER_THRESHOLD]
                return final_docs[:Config.LLM_K_FINAL]
            else:
                # 2b. Fallback: Extract priority terms exactly like old logic
                keywords = [w.lower() for w in query.split() if len(w) > 3]
                
                website_docs = []
                priority_docs = []
                other_docs = []
                seen_chunk_ids = set()
                
                for d in initial_docs:
                    chunk_id = f"{d.metadata.get('source')}_{d.metadata.get('page')}_{d.metadata.get('chunk', d.page_content[:50])}"
                    if chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk_id)
                    
                    if not isinstance(d.metadata, dict):
                        d.metadata = {"source": "Unknown"}
                        
                    content_lower = d.page_content.lower()
                    source = d.metadata.get("source", "").lower()
                    
                    # Boost website sources as they are usually more specific to recent queries
                    is_website = source.startswith("http")
                    
                    if is_website:
                        website_docs.append(d)
                    elif any(kw in content_lower for kw in keywords):
                        priority_docs.append(d)
                    else:
                        other_docs.append(d)
                
                # Order: Websites first, then keyword matches, then others
                combined = website_docs + priority_docs + other_docs
                return combined[:Config.LLM_K_FINAL]

    retriever = AdvancedHybridRetriever(vectorstore, llm=llm, history_retriever=history_retriever)


    # 4. Construct the Chain (Matching Octopiai entirely)
    def format_docs(docs):
        # Master formatter matching chatbot/_format_documents
        import sqlite3
        conn = None
        if os.path.exists(Config.CRAWLER_DB):
            try:
                conn = sqlite3.connect(Config.CRAWLER_DB)
            except:
                pass

        def get_url(source):
            if not conn: return "#"
            try:
                cursor = conn.cursor()
                # Try absolute path matching or basename matching
                basename = os.path.basename(source)
                cursor.execute("SELECT url FROM pages WHERE filename = ? OR filename = ? LIMIT 1", (source, basename))
                row = cursor.fetchone()
                return row[0] if row else "#"
            except:
                return "#"

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            url = get_url(source) if source != "Chat History" else "#"
            
            # Clean "search_document:" prefix if using NomicWrapper
            content = doc.page_content.replace("search_document: ", "")
            
            context_parts.append(
                f"CONTENT: {content}\n"
                f"METADATA: Source: {os.path.basename(source)}, Page: {page}, URL: {url}"
            )
        
        if conn: conn.close()
        return "\n\n---\n\n".join(context_parts)

    def process_history():
        # Octopiai passes history as a distinct variable block rather than prepending 
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            from app.services.chat_session import format_history_for_prompt
            history_text = format_history_for_prompt(conversation_history)
        return history_text

    # Rephrase Chain
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    def get_search_query(q):
        history = process_history()
        if not history.strip():
            return q
        try:
            return rephrase_chain.invoke({"history": history, "question": q})
        except:
            return q

    def process_question(q):
        if is_continuation and last_answer:
            return f"--- CONTINUATION ---\nContinue from:\n{last_answer}\n\nQuestion: {q}"
        return q

    rag_chain = (
        {
            "rephrased_query": RunnableLambda(get_search_query),
            "original_question": RunnablePassthrough()
        }
        | {
            "context": RunnableLambda(lambda x: retriever.invoke(x.get("rephrased_query", "")) if isinstance(x, dict) else retriever.invoke(str(x))) | format_docs, 
            "history": RunnableLambda(lambda _: process_history()),
            "question": RunnableLambda(lambda x: process_question(x.get("rephrased_query", "")) if isinstance(x, dict) else process_question(str(x)))
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def filter_cited_sources(answer: str, all_sources: list) -> list:
    """
    Filters the list of all retrieved sources to return only those that are explicitly 
    cited in the answer text (e.g., [Document.pdf]).
    """
    import re
    # Extract text in brackets [source, ...]
    # We look for patterns like [Doc.pdf] or [Doc.pdf, Page X]
    citations = re.findall(r'\[(.*?)\]', answer)
    cited_filenames = set()
    for c in citations:
        # Take first part "File.pdf" from "File.pdf, Page 2"
        part = c.split(',')[0].strip()
        cited_filenames.add(part.lower())
    
    final_sources = []
    for src in all_sources:
        # Check if src matches any cited filename
        # We compare basenames to handle paths/URLs
        basename = os.path.basename(src).lower()
        if basename in cited_filenames or src.lower() in cited_filenames:
            final_sources.append(src)
            
    # If no sources are cited but we have an answer, we might strictly want to return nothing,
    # or fallback. The user request implies "just resource that i get information from".
    # If LLM didn't cite, or we want to be generous, we fallback to all retrieved sources.
    if not final_sources:
        return sorted(list(set(all_sources)))
    
    return sorted(list(set(final_sources)))

def answer_question(question: str, deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = "", conversation_history: list = None):
    """
    Entry point to answer a question with performance metrics and source citations.
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(
            deep_thinking=deep_thinking, 
            is_continuation=is_continuation, 
            last_answer=last_answer,
            conversation_history=conversation_history
        )
        
        # 1. Retrieve documents manually to get metadata for citations
        docs = retriever.invoke(question)
        sources = sorted(list(set([doc.metadata.get("source", "Unknown") for doc in docs])))
        
        # Capture used docs with content
        used_docs = []
        for doc in docs:
            used_docs.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0)
            })

        # 2. Invoke the chain
        start_llm = time.time()
        answer = chain.invoke(question)
        end_time = time.time()
        
        # 3. Filter sources based on citation
        relevant_sources = filter_cited_sources(answer, sources)
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        
        performance = f"Total {total_time:.1f}s | LLM {llm_time:.1f}s"
        
        return {
            "answer": answer,
            "sources": relevant_sources,
            "performance": performance,
            "used_docs": used_docs 
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def stream_answer(question: str, deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = "", conversation_history: list = None):
    """
    Entry point to stream chunks of the LLM response.
    Yields newline-delimited JSON strings for client processing.
    conversation_history: List of dicts with 'role' and 'content' keys
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(
            deep_thinking=deep_thinking, 
            is_continuation=is_continuation, 
            last_answer=last_answer,
            conversation_history=conversation_history
        )
        
        # 1. Pre-retrieve docs just to get sources early (optional but better for UX)
        start_retrieval = time.time()
        docs = retriever.invoke(question)
        end_retrieval = time.time()
        
        sources = sorted(list(set([doc.metadata.get("source", "Unknown") for doc in docs])))
        
        # Capture used docs with content
        used_docs = []
        for doc in docs:
            used_docs.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0)
            })
        
        # Send ALL sources first so frontend can build the link map
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        
        # 2. Stream the chain
        start_llm = time.time()
        accumulated_text = ""
        for chunk in chain.stream(question):
            accumulated_text += chunk
            yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
        
        end_time = time.time()
        
        # 3. Filter sources for distinct display
        relevant_sources = filter_cited_sources(accumulated_text, sources)
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        retrieval_time = end_retrieval - start_retrieval
        
        # Simple token estimation (approx 4 chars per token)
        estimated_tokens = len(accumulated_text) // 4
        
        performance = {
            "total_time": round(total_time, 2),
            "llm_time": round(llm_time, 2),
            "retrieval_time": round(retrieval_time, 2)
        }
        
        current_model = Config.OLLAMA_LLM_MODEL if Config.LLM_PROVIDER == "ollama" else Config.LLM_PROVIDER.upper() # Fallback to provider name
        
        # Prepare metadata for the frontend Trace and Context modals
        yield json.dumps({
            "type": "metadata", 
            "sources": relevant_sources, 
            "performance": performance,
            "tokens": estimated_tokens,
            "model": current_model,
            "chunks": used_docs, 
            "search_query": question, 
            "history": conversation_history or [] 
        }) + "\n"
        
        yield json.dumps({"type": "done"}) + "\n"

    except ValueError as e:
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        yield json.dumps({"type": "error", "content": f"An unexpected error occurred: {str(e)}"}) + "\n"
