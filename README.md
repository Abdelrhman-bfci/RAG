# Strict RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and OpenAI/Ollama. This system ingests PDF documents and SQL database records into a local FAISS vector store to answer user questions with strict context control.

## 🚀 Features

- **Multi-Source Ingestion with Streaming Progress**:
  - **PDFs**: Automatically scans and indexes PDFs from a `/RESOURCE` directory with real-time logs.
  - **Web Crawling**: Enter any URL in the UI to crawl, extract sub-links, and ingest content dynamically.
  - **SQL Database**: Selective ingestion by schema and table. Pick exactly what you need to index.
- **Flexible LLM & Embedding Providers**:
  - **Ollama (Local)**: High-performance local inference.
  - **vLLM (Local GPU)**: Enterprise-grade throughput for local servers.
  - **OpenAI**: Support for ChatGPT models (GPT-4o, etc.) with custom base URL support.
  - **Google Gemini**: Support for Gemini 1.5 Pro and Flash.
- **Advanced QA**:
  - **Deep Thinking Mode**: Analytical prompt engineering for complex reasoning.
  - **Strict Adherence**: Answers ONLY from provided context or admits ignorance.
  - **Precise Citations**: Cites source file and exact page number for transparency.
- **Premium Chat UI**: Responsive, glassmorphism design with real-time markdown rendering and streaming ingestion console.

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM Orchestration**: LangChain
- **Providers**: OpenAI, Google Gemini, Ollama, vLLM
- **Vector Store**: FAISS
- **Database**: SQLAlchemy 

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   # For Gemini support:
   pip3 install langchain-google-genai
   ```

3. **Environment Setup**:
   Create a `.env` file in the root directory (see `.env.example` or the template below):
   
   **Cloud Providers (OpenAI/Gemini)**
   ```env
   LLM_PROVIDER=openai # or gemini
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=AIza...
   ```

   **Local Providers (Ollama/vLLM)**
   ```env
   LLM_PROVIDER=ollama # or vllm
   OLLAMA_BASE_URL=http://localhost:11434
   VLLM_BASE_URL=http://localhost:9090/v1
   ```

## 🏃 Usage

1. **Start the Server**:
   ```bash
   # Local
   python3 -m uvicorn app.main:app --reload
   ```

2. **Manage Knowledge Base (UI)**:
   - Open `http://localhost:80/client/index.html`.
   - Click the **Database** icon in the sidebar.
   - **PDFs**: Drop files or browse to upload.
   - **Web**: Paste a URL to crawl and ingest.
   - **SQL**: Select a schema, pick your tables, and click Ingest.
   - *Monitor real-time progress in the built-in streaming console.*

3. **Configure Providers (UI)**:
   - Click the **Settings** icon.
   - Switch between providers (OpenAI, Gemini, Ollama, vLLM).
   - Enter API keys or Base URLs on the fly.
   - Select your preferred LLM and Embedding models.

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
