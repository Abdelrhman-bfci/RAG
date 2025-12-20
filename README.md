# Strict RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and OpenAI/Ollama. This system ingests PDF documents and SQL database records into a local FAISS vector store to answer user questions with strict context control.

## 🚀 Features

- **Multi-Source Ingestion**:
  - **PDFs**: Automatically scans and indexes PDFs from a `/RESOURCE` directory.
  - **SQL Database**: Indexes text from SQL tables or arbitrary queries.
- **Flexible LLM Support**:
  - **OpenAI**: Use GPT-4 for top-tier performance.
  - **Ollama (Local)**: Run open-source models (Llama 3, Mistral) locally or on your own server for privacy and cost savings.
- **Strict QA**:
  - Answers **ONLY** from the retrieved context.
  - Responds with "I don't know based on the provided data" if the answer is missing.
- **Vector Search**: Uses local FAISS index for fast and private similarity search.
- **API First**: Fully documented FastAPI endpoints.

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **LLM Providers**: 
  - **OpenAI** (gpt-4-turbo-preview)
  - **Ollama** (Llama 3, Mistral, etc.)
- **Vector Store**: FAISS (CPU)
- **Database**: SQLAlchemy (Generic SQL support)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Install Ollama (Optional)**:
   If you want to run local models:
   - Download from [ollama.com](https://ollama.com)
   - Pull a model: `ollama pull llama3`

4. **Environment Setup**:
   Create a `.env` file in the root directory:
   
   **Option A: Using OpenAI**
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ```

   **Option B: Using Ollama (Local/Server)**
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   ```

   **Common Settings**
   ```env
   DATABASE_URL=mysql+pymysql://user:password@localhost/dbname
   RESOURCE_DIR=RESOURCE
   VECTOR_DB_PATH=faiss_index
   ```

## 🏃 Usage

1. **Start the Server**:
   ```bash
   # Using the full path to the virtual environment's python is the most reliable way with sudo
   sudo ./venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 80 --reload
   ```

2. **Ingest Data**:
   - **PDFs**: Place files in `RESOURCE/` and POST to `/ingest/pdf`.
   - **Database**: POST query/table to `/ingest/db`.

3. **Ask Questions**:
   POST to `/ask`:
   ```json
   {
     "question": "What is the summary of the report?"
   }
   ```

4. **API Documentation**:
   Visit `http://localhost:80/docs` for interactive Swagger UI.

## 📂 Project Structure

```
app/
 ├── main.py            # API Entrypoint
 ├── config.py          # Configuration & Env Vars (OpenAI/Ollama switch)
 ├── ingestion/         # Data Loaders (PDF, SQL)
 ├── qa/                # RAG Logic & Prompts
 ├── vectorstore/       # FAISS Wrapper (Dynamic Embeddings)
requirements.txt        # Dependencies
RESOURCE/               # PDF Directory
README.md               # Documentation
```
