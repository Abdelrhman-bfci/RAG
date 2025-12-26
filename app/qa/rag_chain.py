from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore
import time
import traceback

def get_rag_chain():
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
            # 1. Broad vector search (fast)
            initial_docs = self.vectorstore.similarity_search(query, k=50)
            
            # 2. Extract priority terms
            keywords = [w.lower() for w in query.split() if len(w) > 3]
            
            priority_docs = []
            other_docs = []
            
            for d in initial_docs:
                content_lower = d.page_content.lower()
                if any(kw in content_lower for kw in keywords):
                    priority_docs.append(d)
                else:
                    other_docs.append(d)
            
            # Priority first, then top others
            combined = priority_docs + other_docs
            return combined[:30] # Increase to 30 for better coverage

    retriever = HybridRetriever(vectorstore)

    # 2. Define the Prompt (More general and encouraging)
    prompt = ChatPromptTemplate.from_template("""### INSTRUCTION ###
You are a helpful and factual assistant. Answer the user's question using the provided context.
- Use at least some of the provided context to answer. 
- If the answer is absolutely not in the context, say "I don't know based on the provided data."
- DO NOT use outside knowledge.
- Be clear and accurate.

### CONTEXT ###
{context}

### QUESTION ###
{question}
""")

    # 3. Initialize LLM
    if Config.LLM_PROVIDER == "ollama":
        llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0
        )
    else:
        llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0, # strict factual answers
            openai_api_key=Config.OPENAI_API_KEY
        )

    # 4. Construct the Chain
    def format_docs(docs):
        # Simply return docs as ranked by retriever
        sorted_docs = docs
        
        context_parts = []
        for i, doc in enumerate(sorted_docs):
            context_parts.append(f"--- Document Chunk {i+1} ---\n{doc.page_content}")
        
        return "\n\n".join(context_parts)

    rag_chain = (
        {"context": RunnableLambda(lambda x: retriever.invoke(x)) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def answer_question(question: str):
    """
    Entry point to answer a question with performance metrics.
    """
    start_total = time.time()
    try:
        chain = get_rag_chain()
        
        start_llm = time.time()
        response = chain.invoke(question)
        end_time = time.time()
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        
        perf_info = f"\n\n[⏱️ Performance: Total {total_time:.1f}s | LLM {llm_time:.1f}s]"
        return response + perf_info
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        return f"An unexpected error occurred: {str(e)}"
