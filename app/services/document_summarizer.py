import os
import time
from typing import List, Dict, Generator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from app.config import Config

# Try to import pymupdf4llm for better PDF handling
try:
    import pymupdf4llm
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

def get_document_loader(file_path: str):
    """Get appropriate document loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        if HAS_PYMUPDF:
            return None  # Will use pymupdf4llm directly
        return PyPDFLoader(file_path)
    elif ext == '.docx':
        return Docx2txtLoader(file_path)
    elif ext == '.txt':
        return TextLoader(file_path)
    elif ext == '.csv':
        return CSVLoader(file_path)
    elif ext in ['.xlsx', '.xls']:
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def extract_text_from_document(file_path: str) -> str:
    """Extract text content from various document formats."""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Special handling for PDF with pymupdf4llm
    if ext == '.pdf' and HAS_PYMUPDF:
        try:
            md_text = pymupdf4llm.to_markdown(file_path)
            return md_text
        except Exception as e:
            print(f"pymupdf4llm failed, falling back to PyPDFLoader: {e}")
    
    # Use LangChain loaders for other formats
    loader = get_document_loader(file_path)
    documents = loader.load()
    
    # Combine all document pages/chunks into one text
    full_text = "\n\n".join([doc.page_content for doc in documents])
    return full_text

def split_text_into_chunks(text: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def get_llm_for_summary():
    """Get LLM instance for summarization."""
    from langchain_openai import ChatOpenAI
    from langchain_ollama import ChatOllama
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        ChatGoogleGenerativeAI = None
    
    if Config.LLM_PROVIDER == "ollama":
        return ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0.3,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW
        )
    elif Config.LLM_PROVIDER == "vllm":
        return ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0.3,
            api_key="none"
        )
    elif Config.LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0.3,
            openai_api_key=Config.OPENAI_API_KEY
        )
    elif Config.LLM_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai not installed")
        return ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.3
        )
    else:
        # Default to Ollama
        return ChatOllama(
            model=Config.OLLAMA_LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.3
        )

def summarize_chunk(chunk_text: str, chunk_number: int, total_chunks: int, llm) -> str:
    """Summarize a single chunk of text."""
    prompt = f"""You are summarizing section {chunk_number} of {total_chunks} from a document.

Document Section {chunk_number}/{total_chunks}:
{chunk_text}

Provide a comprehensive summary of this section including:
- Main topics and key points
- Important data, statistics, or findings
- Any conclusions or recommendations
- Critical information that should be retained

Be thorough but concise. Focus on the most important information.

Summary:"""
    
    response = llm.invoke(prompt)
    return response.content

def combine_chunk_summaries(chunk_summaries: List[str], llm) -> str:
    """Combine multiple chunk summaries into one comprehensive summary."""
    combined_text = "\n\n".join([
        f"Section {i+1} Summary:\n{summary}" 
        for i, summary in enumerate(chunk_summaries)
    ])
    
    prompt = f"""You are creating a comprehensive summary by combining multiple section summaries from a document.

All Section Summaries:
{combined_text}

Create a cohesive, comprehensive final summary that:
1. Starts with an executive summary (2-3 sentences)
2. Synthesizes all key points from all sections
3. Maintains logical flow and structure
4. Highlights the most important information
5. Includes all critical data and findings
6. Provides clear conclusions

Final Comprehensive Summary:"""
    
    response = llm.invoke(prompt)
    return response.content

def summarize_document_stream(file_path: str, chunk_size: int = 4000) -> Generator[Dict, None, None]:
    """
    Summarize a document with streaming progress updates.
    Yields progress messages, chunk summaries, and final summary.
    """
    try:
        # Step 1: Extract text
        yield {"type": "progress", "message": "Extracting text from document..."}
        text = extract_text_from_document(file_path)
        
        if not text or len(text.strip()) < 50:
            yield {"type": "error", "message": "Document appears to be empty or contains no extractable text"}
            return
        
        yield {"type": "progress", "message": f"Extracted {len(text)} characters"}
        
        # Step 2: Split into chunks
        yield {"type": "progress", "message": "Splitting document into chunks..."}
        chunks = split_text_into_chunks(text, chunk_size=chunk_size)
        total_chunks = len(chunks)
        
        yield {"type": "progress", "message": f"Document split into {total_chunks} chunks"}
        
        # Step 3: Get LLM
        yield {"type": "progress", "message": "Initializing AI model..."}
        llm = get_llm_for_summary()
        
        # Step 4: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            yield {"type": "progress", "message": f"Summarizing chunk {i+1}/{total_chunks}..."}
            
            summary = summarize_chunk(chunk, i+1, total_chunks, llm)
            chunk_summaries.append(summary)
            
            yield {
                "type": "chunk_summary",
                "chunk": i+1,
                "total": total_chunks,
                "summary": summary
            }
        
        # Step 5: Combine summaries if multiple chunks
        if total_chunks > 1:
            yield {"type": "progress", "message": "Combining all summaries into final comprehensive summary..."}
            final_summary = combine_chunk_summaries(chunk_summaries, llm)
        else:
            final_summary = chunk_summaries[0]
        
        yield {
            "type": "final_summary",
            "content": final_summary,
            "total_chunks": total_chunks
        }
        
        yield {"type": "done"}
        
    except Exception as e:
        yield {"type": "error", "message": f"Error during summarization: {str(e)}"}
