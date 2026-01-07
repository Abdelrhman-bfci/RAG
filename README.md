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
- **Streaming Chat UI**: Premium real-time chat interface with glassmorphism design.
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

   **Option A: Local Development (Recommended)**
   ```bash
   python3 -m uvicorn app.main:app --reload
   ```

   **Option B: Production (using venv and sudo)**
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

4. **Chat Interface**:
   Access the premium real-time chat UI at `http://localhost:80/client/index.html`.

5. **API Documentation**:
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

## Deployment (Always-on)

To keep the server running even after you close the terminal, use one of the following methods:

### Method 1: Systemd (Recommended for Linux)

1.  **Copy the service file** to the system directory:
    ```bash
    sudo cp rag_server.service /etc/systemd/system/rag_server.service
    ```
2.  **Edit the file** to match your server's actual paths if they differ:
    ```bash
    sudo nano /etc/systemd/system/rag_server.service
    ```
3.  **Reload, start, and enable** it to run on boot:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl start rag_server
    sudo systemctl enable rag_server
    ```
4.  **Check status**:
    ```bash
    sudo systemctl status rag_server
    ```

### Method 2: PM2 (Easiest)

If you have Node.js installed, you can use PM2:

1.  **Install PM2**:
    ```bash
    npm install -g pm2
    ```
2.  **Start the app**:
    ```bash
    sudo pm2 start "./venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 80" --name rag-api
    ```
3.  **Save the list** to restart on reboot:
    ```bash
    sudo pm2 save
    sudo pm2 startup
    ```

### Method 3: Screen / Nohup (Quick)

```bash
# Using nohup
sudo nohup ./venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 80 > server.log 2>&1 &
```
