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
            # 1. Broad vector search (increase k to ensure we find sparse website content)
            initial_docs = self.vectorstore.similarity_search(query, k=100)
            
            # 2. Extract priority terms
            keywords = [w.lower() for w in query.split() if len(w) > 3]
            
            website_docs = []
            priority_docs = []
            other_docs = []
            
            for d in initial_docs:
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
            return combined[:30] 

    retriever = HybridRetriever(vectorstore)

    # 2. Define the Prompt (More general and encouraging)
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful and precise assistant for a university/organization.
        Use the following pieces of retrieved context to answer the user's question.
        
        Instructions:
        1. If multiple sources (e.g., a PDF and a Website) talk about different things, prioritize the one that matches the specific entity in the user's question.
        2. If the answer is not in the context, say "I don't know based on the provided resources". Do not make up information.
        3. Be descriptive but concise.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
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

    return rag_chain, retriever

def answer_question(question: str):
    """
    Entry point to answer a question with performance metrics and source citations.
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain()
        
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
