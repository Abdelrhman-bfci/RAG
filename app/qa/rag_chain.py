from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore

def get_rag_chain():
    """
    Creates and returns the RAG chain for Question Answering.
    """
    # 1. Initialize Vector Store and Retriever
    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index()
    
    if not vectorstore:
        raise ValueError("Vector store not found. Please run ingestion first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Define the Strict System Prompt
    system_prompt = """You are a factual assistant.
Answer ONLY using the provided context.
If the answer is not explicitly found in the context, say:
'I don’t know based on the provided data.'

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}")
    ])

    # 3. Initialize LLM
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0, # strict factual answers
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 4. Construct the Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def answer_question(question: str):
    """
    Entry point to answer a question.
    """
    try:
        chain = get_rag_chain()
        response = chain.invoke(question)
        return response
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
