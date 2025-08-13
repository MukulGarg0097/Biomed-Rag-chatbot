```markdown
🧬 BioMed RAG Chatbot (Gemma + FAISS)

📌 Overview
The **BioMed RAG Chatbot** is a **Retrieval-Augmented Generation (RAG)** system designed specifically for **biomedical research data**.  
It retrieves relevant biomedical passages from a **FAISS vector store** and uses a **Gemma 2B model** (downloaded automatically from Hugging Face if not found locally) for generating context-aware, evidence-based responses.

The system also supports **query rewriting** to improve retrieval accuracy and avoid ambiguity in biomedical question answering.  

---

📂 Project Structure
```

app/
├── embedder\_model\_folder/       # Local sentence-transformer model for embeddings
├── index/faiss\_index\_folder/    # Prebuilt FAISS vector index for biomedical corpus
├── models/gemma/                # Local Gemma model cache (auto-downloaded if missing)
├── **init**.py
├── config.py                    # Loads and parses config.json
├── gemma.py                     # Gemma model loader (auto-downloads from Hugging Face)
├── intent.py
├── main.py                      # Flask API entry point
├── retriever.py                 # Embedding & FAISS retrieval
├── rewriter.py                  # Query rewriting logic
docker-entrypoint.sh
Dockerfile
requirements.txt
README.md

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-folder>
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Model Download (First Run)

Gemma 2B is automatically downloaded if not already in `app/models/gemma`.
Since it requires accepting usage terms, **you must log in to Hugging Face** and set your token:

```bash
huggingface-cli login
# OR
export HF_TOKEN=hf_xxxxxxxx
```

Optional:

* Change variant (e.g., instruction-tuned) via:

```bash
export HF_MODEL_REPO=google/gemma-2-2b-it
```

---

## 🚀 Running the Chatbot

### Local Run

```bash
python -m app.main
```

Server: `http://localhost:8080`

### Docker Run

```bash
docker build -t rag_chatbot:latest .
docker run -p 8080:8080 -e HF_TOKEN=hf_xxxxx rag_chatbot:latest
```

> GPU:

```bash
docker run --gpus all -p 8080:8080 -e HF_TOKEN=hf_xxxxx rag_chatbot:latest
```

---

## 🛠 API Endpoints

### 1. Health Check

```bash
curl http://localhost:8080/health
```

### 2. Ask a Question

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are biomarkers for lung cancer?", "k": 3}'
```

### 3. Reload FAISS Index

```bash
curl -X POST http://localhost:8080/reload
```

---

## 🔍 Query Rewriting

When `"USE_REWRITER": true` in `config.json`, the chatbot:

1. Takes your question.
2. Uses Gemma to rewrite it for clarity & accuracy.
3. Sends the rewritten query to FAISS.

Example:

```
Original: "COVID treatment"
Rewritten: "What are the current WHO-recommended treatments for COVID-19?"
```

---

## ⚠️ Notes

* **First run requires internet** to download Gemma; later runs can be offline:

```bash
export TRANSFORMERS_OFFLINE=true
```

* Pre-download:

  * `embedder_model_folder/`
  * `index/faiss_index_folder/`

---

## 📜 License

MIT License — free to use & modify for research & education.

```

---

If you want, I can also **add a section showing exactly how your `gemma.py` now handles automatic downloads** so users know they don’t have to manage model files manually. That would make the README even clearer for new contributors.
```
