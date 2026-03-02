import sys
import os
import json

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.qa.rag_chain import get_rag_chain

def test_retrieval(query):
    print(f"Testing retrieval for: {query}")
    print("-" * 50)
    
    chain, retriever = get_rag_chain()
    
    # Test retrieval
    docs = retriever.invoke(query)
    
    print(f"Found {len(docs)} chunks from retrieval.")
    
    sources = {}
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "Unknown")
        if src not in sources:
            sources[src] = 0
        sources[src] += 1
        
        print(f"\n[CHUNK {i+1}] Source: {src}, Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content snippet: {doc.page_content[:300]}...")
    
    print("\n--- Source Summary ---")
    for src, count in sources.items():
        print(f" - {src}: {count} chunks")

if __name__ == "__main__":
    query = "List only professors in Computer and Systems Engineering department?"
    test_retrieval(query)
