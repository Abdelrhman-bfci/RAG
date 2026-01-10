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
STRICT_RAG_PROMPT = ChatPromptTemplate.from_template("""
    You are a highly precise legal/academic assistant.
    Your goal is to answer questions strictly based on the provided context.
    
    LANGUAGE RULE: 
    - You must detect the language of the user's Question and answer in that SAME language (e.g., if asked in Arabic, answer in Arabic. If asked in English, answer in English).
    
    STRICT RULES:
    1. Answer ONLY using the information from the Context below.
    2. If the answer is not explicitly found in the Context, you MUST say "I cannot answer this based on the provided documents." in the same language as the question.
    3. Do NOT make assumptions or use outside knowledge.
    4. CITE the document name AND page number for EVERY specific detail you provide (e.g., [Document.pdf, Page 5]).
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
""")

DEEP_THINKING_PROMPT = ChatPromptTemplate.from_template("""
    You are an expert academic analyst and research assistant.
    Your goal is to provide a comprehensive, analytical summary and deep insights based on the provided documents.
    
    LANGUAGE RULE: 
    - You must detect the language of the user's Question and answer in that SAME language (e.g., if asked in Arabic, answer in Arabic. If asked in English, answer in English).
    - Adapt your headers (like "Analytical Summary") to the chosen language.
    
    INSTRUCTIONS:
    1. Synthesize information from multiple parts of the Context to provide a detailed, well-structured answer.
    2. Use professional, academic language.
    3. If the Context contains conflicting information, highlight it.
    4. Provide an "Analytical Summary" section followed by "Key Details" (translate these headers if answering in Arabic).
    5. CITE sources and page numbers for major points using [Source Name, Page X].
    6. If the information is missing, clearly state what is unknown while summarizing what IS available.
    
    Context:
    {context}
    
    Question: {question}
    
    Analytical Response:
""")

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
                # For Arabic, filter out very common stop words if possible, or just keep all moderately long words
                keywords = [w.lower() for w in query.split() if len(w) >= 2]
            else:
                keywords = [w.lower() for w in query.split() if len(w) > 3]
            
            website_docs = []
            priority_docs = []
            other_docs = []
            
            for d in initial_docs:
                content_lower = d.page_content.lower()
                source = d.metadata.get("source", "").lower()
                
                # Boost website sources as they are usually more specific to recent queries
                is_website = source.startswith("http")
                
                # Check for keyword matches (handling both languages)
                has_keywords = any(kw in content_lower for kw in keywords)
                
                if is_website:
                    website_docs.append(d)
                elif has_keywords:
                    priority_docs.append(d)
                else:
                    other_docs.append(d)
            
            # Order: Websites first, then keyword matches, then others
            combined = website_docs + priority_docs + other_docs
            return combined[:30] # Increased Top-K to 30 (approx 12k tokens) for very large context synthesisf

    retriever = HybridRetriever(vectorstore)

    # 2. Select Prompt
    prompt = DEEP_THINKING_PROMPT if deep_thinking else STRICT_RAG_PROMPT

    # 3. Initialize LLM
    if Config.LLM_PROVIDER == "ollama":
        llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0.2 if deep_thinking else 0.1,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW
        )
    elif Config.LLM_PROVIDER == "vllm":
        llm = ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0.2 if deep_thinking else 0.1,
            api_key="none"
        )
    elif Config.LLM_PROVIDER == "openai":
        llm = ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0.2 if deep_thinking else 0.1,
            openai_api_key=Config.OPENAI_API_KEY
        )
    elif Config.LLM_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed. Please run 'pip install langchain-google-genai'")
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.2 if deep_thinking else 0.1
        )
    else:
        # Default fallback to Ollama
        llm = ChatOllama(
            model=Config.OLLAMA_LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.2 if deep_thinking else 0.1,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW
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

    rag_chain = (
        {"context": RunnableLambda(lambda x: retriever.invoke(x)) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

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
        
        # 2. Invoke the chain
        start_llm = time.time()
        answer = chain.invoke(question)
        end_time = time.time()
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        
        performance = f"Total {total_time:.1f}s | LLM {llm_time:.1f}s"
        
        return {
            "answer": answer,
            "sources": sources,
            "performance": performance
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
        
        # Send sources first
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        
        # 2. Stream the chain
        start_llm = time.time()
        accumulated_text = ""
        for chunk in chain.stream(question):
            accumulated_text += chunk
            yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
        
        end_time = time.time()
        
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
            "sources": sources, 
            "performance": performance,
            "tokens": estimated_tokens,
            "model": current_model
        }) + "\n"
        
        yield json.dumps({"type": "done"}) + "\n"

    except ValueError as e:
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        yield json.dumps({"type": "error", "content": f"An unexpected error occurred: {str(e)}"}) + "\n"
