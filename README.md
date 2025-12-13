# Strict RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and OpenAI. This system ingests PDF documents and SQL database records into a local FAISS vector store to answer user questions with strict context control.

## 🚀 Features

- **Multi-Source Ingestion**:
  - **PDFs**: Automatically scans and indexes PDFs from a `/RESOURCE` directory.
  - **SQL Database**: Indexes text from SQL tables or arbitrary queries.
- **Strict QA**:
  - Answers **ONLY** from the retrieved context.
  - Responds with "I don't know based on the provided data" if the answer is missing.
- **Vector Search**: Uses local FAISS index for fast and private similarity search.
- **API First**: Fully documented FastAPI endpoints.

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **LLM**: OpenAI GPT (gpt-4-turbo-preview)
- **Embeddings**: OpenAI text-embedding-3-large
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
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=sk-...
   DATABASE_URL=mysql+pymysql://user:password@localhost/dbname
   RESOURCE_DIR=RESOURCE
   VECTOR_DB_PATH=faiss_index
   ```

## 🏃 Usage

1. **Start the Server**:
   ```bash
   uvicorn app.main:app --reload
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
   Visit `http://localhost:8000/docs` for interactive Swagger UI.

## 📂 Project Structure

```
app/
 ├── main.py            # API Entrypoint
 ├── config.py          # Configuration & Env Vars
 ├── ingestion/         # Data Loaders (PDF, SQL)
 ├── qa/                # RAG Logic & Prompts
 ├── vectorstore/       # FAISS Wrapper
requirements.txt        # Dependencies
RESOURCE/               # PDF Directory
README.md               # Documentation
```
