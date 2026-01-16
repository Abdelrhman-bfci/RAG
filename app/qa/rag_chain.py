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
from app.vectorstore.faiss_store import FAISSStore
import time
import traceback
import json

# --- Prompt Templates ---
# --- Prompt Templates ---
STRICT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a highly precise legal/academic assistant. Your goal is to answer questions strictly based on the provided context.
    
    CRITICAL INSTRUCTION: {language_instruction}
    
    CRITICAL LANGUAGE RULES (ABSOLUTE PRIORITY):
    1. **Strict Matching**: 
       - IF Question is in **English** -> Answer MUST be in **English**.
       - IF Question is in **Arabic** -> Answer MUST be in **Arabic**.
       - IF Question is in **Mixed** -> Answer in the DOMINANT language of the question.
    2. **Context Independence**: The language of the *Context* documents implies NOTHING about the answer language. Ignore context language when deciding output language.
    3. **No Translation Explanations**: Do NOT say "I am translating this". Just answer directly in the correct language.

    STRICT COMPLIANCE RULES:
    1. Answer ONLY using the information from the Context.
    2. If the answer is not in the Context, say "I cannot answer this based on the provided documents" (translate to Arabic if question is Arabic).
    3. **Inline Citations**: You MUST cite sources INLINE (directly after the relevant information) using [Document.pdf, Page X] or [Table: TableName]. DO NOT just list them at the end.
    4. CRITICAL: When listing items (courses, programs, requirements, etc.), you MUST list ALL items found in the context. DO NOT truncate, summarize, or say "and more". Provide the COMPLETE list.
    5. If the answer requires a long response, provide the FULL answer without cutting it short.

    DATABASE & RELATION RULES:
    1. **Resolved Foreign Keys**: If you see fields like `department_id` and `department_name`, they are linked. Use the `_name` field for human-readable answers.
    2. **Implicit Relations**: If a user asks about "Courses in Computer Science", look for `department_name: Computer Science` in the `courses` table context.
    3. **Schema Awareness**: `institutes` table refers to "Faculties" or "Colleges". `departments` refers to academic departments."""),
    ("human", """Context:
    {context}
    
    Question: {question}""")
])

DEEP_THINKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """FIRST AND MOST IMPORTANT: {language_instruction}
    
    - **Enforce Output Language**:
        - Question in **English** -> Response in **English**.
        - Question in **Arabic** -> Response in **Arabic**.
    - **Context Irrelevance**: The documents might be in a different language. You must TRANSLATE the information from the context into the Question's language.
    - **No Metadata/Preachiness**: Do not start with "Here is the answer in English". Just give the answer.
    
    You are an expert academic analyst and research assistant. Your goal is to provide a comprehensive, analytical summary and deep insights based on the provided documents.
    
    INSTRUCTIONS:
    1. Synthesize information from the Context.
    2. Use professional, academic language in the SAME language as the question.
    3. Structure your response with clear sections:
       - For Arabic questions: Start with "ملخص تحليلي" (Analytical Summary) followed by "تفاصيل رئيسية" (Key Details)
       - For English questions: Start with "Analytical Summary" followed by "Key Details"
    4. **Inline Citations**: Cite major points INLINE (immediately after the fact or claim) using [Source Name, Page X] or [Table: TableName].
    5. If information is missing, clearly state what is unknown in the user's language.
    6. CRITICAL: When listing items (courses, programs, requirements, etc.), you MUST list ALL items found in the context. DO NOT truncate, summarize, or say "and more". Provide the COMPLETE list.
    7. If the answer requires a long response, provide the FULL answer without cutting it short. You have sufficient token capacity.
    
    RELATIONAL DATA LOGIC:
    - **Connect the Dots**: Data might be spread across tables (e.g., Course -> Department -> Faculty). Use the enriched `_name` fields (like `department_name`) to bridge these connections.
    - **Terminology**: Treat `institutes` as Faculties/Colleges.
    - **Aggregation**: If analyzing data from a database, summarize trends or groupings where appropriate."""),
    ("human", """Context:
    {context}
    
    Question: {question}""")
])

def get_rag_chain(deep_thinking: bool = False):
    """
    Creates and returns the RAG chain for Question Answering.
    """
    # 1. Initialize Vector Store and Retriever
    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index()
    
    if not vectorstore:
        raise ValueError("Vector store not found. Please run ingestion first.")

    # Custom Hybrid Retrieval: Vector Search + Rapid Keyword check
    class HybridRetriever:
        def __init__(self, vectorstore):
            self.vectorstore = vectorstore
            
        def invoke(self, query):
            # 1. Broad vector search (increase k to ensure we find sparse website content)
            initial_docs = self.vectorstore.similarity_search(query, k=300)
            
            # 2. Extract priority terms
            # Improve keyword extraction: keep words > 3 chars for EN, and all words for AR (as Arabic words can be short)
            import re
            is_arabic = bool(re.search('[\u0600-\u06FF]', query))
            
            if is_arabic:
                keywords = [w.lower() for w in query.split() if len(w) >= 2]
            else:
                keywords = [w.lower() for w in query.split() if len(w) > 3]
            
            # Simple re-ranking: Boost documents that contain exact keywords
            # We want to maintain diversity, so we won't strictly segregate.
            
            priority_docs = []
            other_docs = []
            
            for d in initial_docs:
                content_lower = d.page_content.lower()
                
                # Check for keyword matches
                has_keywords = any(kw in content_lower for kw in keywords)
                
                if has_keywords:
                    priority_docs.append(d)
                else:
                    other_docs.append(d)
            
            # Return priority matches first, then others, but don't filter out non-web sources
            combined = priority_docs + other_docs
            return combined[:30] # Top 30 documents

    retriever = HybridRetriever(vectorstore)

    # 2. Select Prompt
    prompt = DEEP_THINKING_PROMPT if deep_thinking else STRICT_RAG_PROMPT

    # 3. Initialize LLM
    if Config.LLM_PROVIDER == "ollama":
        llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0.2 if deep_thinking else 0.1,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            num_predict=-1,  # Unlimited tokens - generate until naturally complete
            stop=[],  # Remove default stop sequences to allow complete responses
            repeat_penalty=1.1  # Slight penalty to avoid repetition while maintaining completeness
        )
    elif Config.LLM_PROVIDER == "vllm":
        llm = ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0.2 if deep_thinking else 0.1,
            api_key="none",
            max_tokens=8192  # Maximum tokens to generate in response
        )
    elif Config.LLM_PROVIDER == "openai":
        llm = ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0.2 if deep_thinking else 0.1,
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=8192  # Maximum tokens to generate in response
        )
    elif Config.LLM_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed. Please run 'pip install langchain-google-genai'")
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.2 if deep_thinking else 0.1,
            max_output_tokens=8192  # Maximum tokens to generate in response
        )
    else:
        # Default fallback to Ollama
        llm = ChatOllama(
            model=Config.OLLAMA_LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.2 if deep_thinking else 0.1,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            num_predict=-1,  # Unlimited tokens - generate until naturally complete
            stop=[],  # Remove default stop sequences to allow complete responses
            repeat_penalty=1.1  # Slight penalty to avoid repetition while maintaining completeness
        )

    # 4. Construct the Chain
    def format_docs(docs):
        context_parts = []
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            # PyMuPDF uses 0-indexed pages, making it 1-indexed for the LLM/user
            if isinstance(page, int):
                page = page + 1
            
            header = f"--- Document: {source} | Page: {page} ---"
            context_parts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)

    def detect_language_instruction(question):
        import re
        # Stronger check: if any arabic char exists, treat as Arabic
        is_arabic = bool(re.search('[\u0600-\u06FF]', question))
        if is_arabic:
            return "THE USER ASKED IN ARABIC. YOU MUST ANSWER IN ARABIC. TRANSLATE CONTEXT IF NEEDED."
        else:
            return "THE USER ASKED IN ENGLISH. YOU MUST ANSWER IN ENGLISH. TRANSLATE ARABIC CONTEXT TO ENGLISH."

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: retriever.invoke(x)) | format_docs, 
            "question": RunnablePassthrough(),
            "language_instruction": RunnableLambda(detect_language_instruction)
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

def answer_question(question: str, deep_thinking: bool = False):
    """
    Entry point to answer a question with performance metrics and source citations.
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(deep_thinking=deep_thinking)
        
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

def stream_answer(question: str, deep_thinking: bool = False):
    """
    Entry point to stream chunks of the LLM response.
    Yields JSON strings containing either chunks or final metadata.
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(deep_thinking=deep_thinking)
        
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
            "total": f"{total_time:.1f}s",
            "llm": f"{llm_time:.1f}s",
            "retrieval": f"{retrieval_time:.1f}s"
        }
        
        current_model = Config.OLLAMA_LLM_MODEL if Config.LLM_PROVIDER == "ollama" else Config.LLM_MODEL
        
        yield json.dumps({
            "type": "metadata", 
            "sources": relevant_sources, 
            "performance": performance,
            "tokens": estimated_tokens,
            "model": current_model,
            "used_docs": used_docs
        }) + "\n"
        
        yield json.dumps({"type": "done"}) + "\n"

    except ValueError as e:
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        yield json.dumps({"type": "error", "content": f"An unexpected error occurred: {str(e)}"}) + "\n"
