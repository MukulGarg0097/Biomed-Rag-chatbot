```markdown
# 🧬 BioMed RAG Chatbot (Gemma + FAISS)

## 📌 Overview
The **BioMed RAG Chatbot** is a **Retrieval-Augmented Generation (RAG)** system designed specifically for **biomedical research data**.  
It retrieves relevant biomedical passages from a **FAISS vector store** and uses a **local Gemma model** for generating context-aware, evidence-based responses.

This system also supports **query rewriting** to improve retrieval accuracy and avoid ambiguity in biomedical question answering.  

---

## 📂 Project Structure
```

app/
├── embedder\_model\_folder/       # Local sentence-transformer model for embeddings
├── index/faiss\_index\_folder/    # Prebuilt FAISS vector index for biomedical corpus
├── models/gemma/                 # Local Gemma model files
├── __init__.py                   # Marks app as a Python package
├── config.py                     # Loads and parses config.json
├── gemma.py                      # Gemma model loading & text generation
├── intent.py                     # (Optional) intent-specific logic
├── main.py                       # Flask API entry point
├── retriever.py                  # Embedding & FAISS retrieval
├── rewriter.py                   # Query rewriting logic
docker-entrypoint.sh              # Docker container startup script
Dockerfile                        # Docker build instructions
requirements.txt                  # Python dependencies
readme.md                         # Project documentation

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-folder>
````

### 2️⃣ Install Dependencies (Local Development)

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Chatbot

### **Local Run**

```bash
python -m app.main
```

Server starts at:

```
http://localhost:8080
```

---

### **Docker Run**

#### Build the Image

```bash
docker build -t rag_chatbot:latest .
```

#### Run the Container

```bash
docker run -p 8080:8080 --rm rag_chatbot:latest
```

#### Run with Custom Config

```bash
docker run -p 8080:8080 -e CONFIG_PATH=./config.json rag_chatbot:latest
```

#### Run with GPU (if available)

```bash
docker run --gpus all -p 8080:8080 rag_chatbot:latest
```

> Requires NVIDIA drivers & `nvidia-docker2`.

---

## 🛠 API Endpoints

### 1. **Health Check**

```bash
curl http://localhost:8080/health
```

**Sample Response:**

```json
{
  "status": "ok",
  "model_dir": "/app/app/models/gemma",
  "faiss_dir": "/app/app/index/faiss_index_folder",
  "use_rewriter": true,
  "top_k_default": 5,
  "index_size": 27975
}
```

---

### 2. **Ask a Biomedical Question**

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are biomarkers for lung cancer?", "k": 3}'
```

**Response Fields:**

* `question`: Original user question.
* `rewritten`: Query after rewrite (for better retrieval).
* `answer`: Generated biomedical answer.
* `sources`: Retrieved biomedical passages from FAISS.

---

### 3. **Reload FAISS Index**

```bash
curl -X POST http://localhost:8080/reload
```

---

## 🔍 How Query Rewriting Works

If enabled (`"USE_REWRITER": true` in config.json), the chatbot will:

1. Take your biomedical query.
2. Use the Gemma-based **rewriter** to clarify terms, expand abbreviations, and remove ambiguity.
3. Send the rewritten query to FAISS for improved retrieval accuracy.

Example:

```
Original: "COVID treatment"
Rewritten: "What are the current WHO-recommended treatments for COVID-19?"
```

---

## ⚠️ Notes

* This chatbot is **restricted to biomedical datasets** only.
* All models and indexes must be pre-downloaded for offline mode:

  * `embedder_model_folder/`
  * `models/gemma/`
  * `index/faiss_index_folder/`
* For offline use:

```bash
export TRANSFORMERS_OFFLINE=true
```

---

## 📜 License

MIT License — free to use and modify for research & educational purposes.

```