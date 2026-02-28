import os
import time
import concurrent.futures
from typing import List, Dict, Generator, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from app.config import Config

try:
    import pymupdf4llm
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Max parallel workers for chunk summarization (avoid overloading LLM)
SUMMARY_MAX_WORKERS = int(getattr(Config, "SUMMARY_MAX_WORKERS", "4"))


def get_document_loader(file_path: str):
    """Get appropriate document loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return None  # Handled separately
    if ext == ".docx":
        return Docx2txtLoader(file_path)
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8")
    if ext == ".csv":
        return CSVLoader(file_path)
    if ext in (".xlsx", ".xls"):
        return UnstructuredExcelLoader(file_path)
    raise ValueError(f"Unsupported file format: {ext}")


def load_text_with_encoding_fallback(file_path: str) -> str:
    """Load .txt/.md with encoding fallback. Returns concatenated text."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            loader = TextLoader(file_path, encoding=encoding)
            docs = loader.load()
            return "\n\n".join(doc.page_content for doc in docs)
        except (UnicodeDecodeError, OSError):
            continue
    raise ValueError(f"Could not decode {file_path} with any supported encoding")


def extract_text_from_document(file_path: str) -> str:
    """Extract text content from various document formats."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        if HAS_PYMUPDF:
            try:
                return pymupdf4llm.to_markdown(file_path)
            except Exception as e:
                pass  # Fall through to PyMuPDFLoader
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            return "\n\n".join(doc.page_content for doc in documents)
        except Exception:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n\n".join(doc.page_content for doc in documents)

    if ext in (".txt", ".md"):
        return load_text_with_encoding_fallback(file_path)

    try:
        loader = get_document_loader(file_path)
        documents = loader.load()
        return "\n\n".join(doc.page_content for doc in documents)
    except Exception as e:
        raise ValueError(f"Failed to extract text from {ext} file: {str(e)}") from e

def split_text_into_chunks(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[str]:
    """Split text into overlapping chunks for processing. Uses Config if sizes not provided."""
    chunk_size = chunk_size or getattr(Config, "SUMMARY_CHUNK_SIZE", 4000)
    chunk_overlap = chunk_overlap or getattr(Config, "SUMMARY_CHUNK_OVERLAP", 200)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_text(text)

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
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
        )
    elif Config.LLM_PROVIDER == "vllm":
        return ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0.3,
            api_key="none",
        )
    elif Config.LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0.3,
            openai_api_key=Config.OPENAI_API_KEY,
        )
    elif Config.LLM_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai not installed")
        return ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.3,
        )
    return ChatOllama(
        model=Config.OLLAMA_LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        temperature=0.3,
        num_ctx=getattr(Config, "OLLAMA_CONTEXT_WINDOW", 8192),
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

def summarize_document_stream(
    file_path: str,
    chunk_size: Optional[int] = None,
    include_chunks: bool = False,
) -> Generator[Dict, None, None]:
    """
    Summarize a document with streaming progress updates.
    Uses Config.SUMMARY_CHUNK_SIZE and SUMMARY_CHUNK_OVERLAP when chunk_size is not provided.
    """
    chunk_size = chunk_size or getattr(Config, "SUMMARY_CHUNK_SIZE", 4000)
    chunk_overlap = getattr(Config, "SUMMARY_CHUNK_OVERLAP", 200)
    max_workers = min(SUMMARY_MAX_WORKERS, 8)

    try:
        yield {"type": "progress", "message": "Extracting text from document..."}
        text = extract_text_from_document(file_path)

        if not text or len(text.strip()) < 50:
            yield {"type": "error", "message": "Document appears to be empty or contains no extractable text"}
            return

        yield {"type": "progress", "message": f"Extracted {len(text):,} characters"}

        yield {"type": "progress", "message": "Splitting document into chunks..."}
        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks = len(chunks)

        yield {"type": "progress", "message": f"Document split into {total_chunks} chunk(s)"}

        yield {"type": "progress", "message": "Initializing AI model..."}
        llm = get_llm_for_summary()

        chunk_summaries = [None] * total_chunks

        if total_chunks > 1:
            yield {"type": "progress", "message": f"Summarizing {total_chunks} chunks (max {max_workers} parallel)..."}

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(summarize_chunk, chunk, i + 1, total_chunks, llm): i
                    for i, chunk in enumerate(chunks)
                }
                completed = 0
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        summary = future.result()
                        chunk_summaries[index] = summary
                        completed += 1
                        yield {"type": "progress", "message": f"Completed chunk {completed}/{total_chunks}..."}
                        if include_chunks:
                            yield {
                                "type": "chunk_summary",
                                "chunk": index + 1,
                                "total": total_chunks,
                                "summary": summary,
                            }
                    except Exception as e:
                        yield {"type": "error", "message": f"Error summarizing chunk {index + 1}: {str(e)}"}
        else:
            summary = summarize_chunk(chunks[0], 1, 1, llm)
            chunk_summaries[0] = summary
            if include_chunks:
                yield {"type": "chunk_summary", "chunk": 1, "total": 1, "summary": summary}

        if total_chunks > 1:
            yield {"type": "progress", "message": "Combining all summaries into final comprehensive summary..."}
            final_summary = combine_chunk_summaries(chunk_summaries, llm)
        else:
            final_summary = chunk_summaries[0]

        yield {
            "type": "final_summary",
            "content": final_summary,
            "total_chunks": total_chunks,
        }
        yield {"type": "done"}

    except FileNotFoundError as e:
        yield {"type": "error", "message": str(e)}
    except Exception as e:
        yield {"type": "error", "message": f"Error during summarization: {str(e)}"}
