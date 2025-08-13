# 🧬 BioMed RAG Chatbot (Gemma + FAISS)

## 📌 Overview
The **BioMed RAG Chatbot** is a **Retrieval-Augmented Generation (RAG)** system designed specifically for **biomedical research data**.  
It retrieves relevant biomedical passages from a **FAISS vector store** and uses a **Gemma 2B model** (downloaded automatically from Hugging Face if not found locally) for generating context-aware, evidence-based responses.

The system also supports **query rewriting** to improve retrieval accuracy and avoid ambiguity in biomedical question answering.

---

## 📂 Project Structure
```plaintext
app/
├── embedder_model_folder/       # Local sentence-transformer model for embeddings
├── index/
│   └── faiss_index_folder/      # Prebuilt FAISS vector index for biomedical corpus
├── models/
│   └── gemma/                   # Local Gemma model cache (auto-downloaded if missing)
├── __init__.py
├── config.py                    # Loads and parses config.json
├── gemma.py                     # Gemma model loader (auto-downloads from Hugging Face)
├── intent.py                    # Intent-specific logic
├── main.py                      # Flask API entry point
├── retriever.py                 # Embedding & FAISS retrieval
├── rewriter.py                  # Query rewriting logic
docker-entrypoint.sh             # Docker container startup script
Dockerfile                       # Docker build instructions
requirements.txt                 # Python dependencies
README.md                        # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MukulGarg0097/Biomed-Rag-chatbot.git
cd Biomed-Rag-chatbot
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📥 Model Download (First Run)
The Gemma 2B model is automatically downloaded from Hugging Face if not already present in `app/models/gemma`.  

Since Gemma requires accepting usage terms, **you must log in to Hugging Face** and set your token:

```bash
huggingface-cli login
# OR
export HF_TOKEN=hf_xxxxxxxx
```

**Optional** — Change to instruction-tuned model:
```bash
export HF_MODEL_REPO=google/gemma-2-2b-it
```

---

## 🚀 Running the Chatbot

### Local Run
```bash
python -m app.main
```
Server starts at:
```
http://localhost:8080
```

### Docker Run
```bash
docker build -t rag_chatbot:latest .
docker run -p 8080:8080 -e HF_TOKEN=hf_xxxxx rag_chatbot:latest
```

**Run with GPU** (if available):
```bash
docker run --gpus all -p 8080:8080 -e HF_TOKEN=hf_xxxxx rag_chatbot:latest
```

---

## 🛠 API Endpoints

### 1. **Health Check**
```bash
curl http://localhost:8080/health
```

### 2. **Ask a Biomedical Question**
```bash
curl -X POST http://localhost:8080/ask   -H "Content-Type: application/json"   -d '{"question": "What are biomarkers for lung cancer?", "k": 3}'
```

### 3. **Reload FAISS Index**
```bash
curl -X POST http://localhost:8080/reload
```

---

## 🔍 Query Rewriting
If `"USE_REWRITER": true` in `config.json`, the chatbot will:
1. Take your biomedical query.
2. Use Gemma to clarify terms, expand abbreviations, and remove ambiguity.
3. Send the rewritten query to FAISS for improved retrieval accuracy.

Example:
```
Original: "COVID treatment"
Rewritten: "What are the current WHO-recommended treatments for COVID-19?"
```

---

## ⚠️ Notes
- **First run requires internet** to download Gemma; later runs can be offline:
```bash
export TRANSFORMERS_OFFLINE=true
```
- Ensure the following are available for offline use:
  - `embedder_model_folder/`
  - `index/faiss_index_folder/`

---

## 📜 License
MIT License — free to use and modify for research & educational purposes.
