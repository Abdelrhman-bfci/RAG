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

    **Option C: Using vLLM (Local GPU Server)**
    ```env
    LLM_PROVIDER=vllm
    VLLM_BASE_URL=http://localhost:9090/v1
    VLLM_MODEL=your-model-id
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

## 🐧 vLLM Linux Setup & Deployment

For servers with high-end GPUs (e.g., 24GB VRAM), vLLM provides superior throughput compared to Ollama.

### 1. Installation
On your Linux GPU server, install vLLM within your virtual environment:
```bash
# Activate your venv first
source venv/bin/activate
pip install vllm
```

### 2. Manual Test Run
Before setting up the service, you can test the engine manually:
```bash
# Optimize for 24GB VRAM (e.g. Qwen2.5-7B)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.95 \
    --port 9090 \
    --host 0.0.0.0
```

### 3. Connect ASU RAG
Once the server is running (manually or via service), open the RAG UI:
1. Go to **Settings**.
2. Select **LLM Provider**: `vLLM (Local)`.
3. The available models will be automatically fetched from port 9090.

## Deployment (Always-on)

To keep the server running even after you close the terminal:

### Systemd (Recommended)

You can manage both the RAG API and the vLLM engine using systemd.

#### 1. RAG API Service
1.  **Configure**: Edit `rag_server.service` to update `WorkingDirectory` and `ExecStart` paths.
2.  **Install**:
    ```bash
    sudo cp rag_server.service /etc/systemd/system/rag_server.service
    sudo systemctl daemon-reload
    sudo systemctl start rag_server
    sudo systemctl enable rag_server
    ```

#### 2. vLLM Engine Service (Local GPU)
1.  **Configure**: Edit `vllm.env` to set your model and quantization.
2.  **Install**:
    ```bash
    sudo cp vllm_server.service /etc/systemd/system/vllm_server.service
    sudo systemctl daemon-reload
    sudo systemctl start vllm_server
    sudo systemctl enable vllm_server
    ```

#### 3. Monitoring
```bash
# Check status
sudo systemctl status rag_server
sudo systemctl status vllm_server

# Real-time logs
journalctl -u rag_server -f
journalctl -u vllm_server -f
```

## 🔒 Security & Firewall (UFW)

If you are running on a remote linux server, ensure the necessary ports are open:

```bash
# Allow RAG API (HTTP)
sudo ufw allow 80/tcp

# Allow vLLM API (Internal or External)
sudo ufw allow 9090/tcp

# Enable firewall
sudo ufw enable
```
