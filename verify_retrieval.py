import sys
import os

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.qa.rag_chain import answer_question
import json

def test_retrieval(query):
    print(f"\nTesting Query: {query}")
    print("-" * 50)
    
    result = answer_question(query, deep_thinking=False)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Performance: {result['performance']}")
    print(f"Sources Found: {len(result['sources'])}")
    for src in result['sources']:
        print(f" - {src}")
    
    print("\nUsed Chunks Sample:")
    for i, doc in enumerate(result.get('used_docs', [])[:3]):
        print(f"\n[Chunk {i+1}] Source: {doc['source']}, Page: {doc['page']}")
        print(f"Content Preview: {doc['content'][:200]}...")

if __name__ == "__main__":
    # Test queries
    test_retrieval("ما هي الكورسات المتاحة في هندسة البرمجيات؟")
    test_retrieval("Tell me about the admission requirements")
