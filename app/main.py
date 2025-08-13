from typing import Optional
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import load_config
from .retriever import load_embedder, load_faiss, retrieve_top_k
from .gemma import load_gemma, build_rewriter, rewrite_query, answer_with_gemma

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_dir = "/app/app/models/gemma"

def get_gemma_model():
    print(f"🔄 Loading Gemma model from {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    print("✅ Gemma model loaded")
    return tok, model


app = Flask(__name__)
CORS(app)

# Load config at startup
CFG = load_config(os.getenv("CONFIG_PATH", "./config.json"))

# optional: keep transformers offline
if CFG.get("TRANSFORMERS_OFFLINE", False):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Globals
tokenizer = None
model = None
gen_pipeline = None
vector_db = None


def startup():
    """Load all required models and indexes into memory."""
    global tokenizer, model, gen_pipeline, vector_db
    print("🔄 Loading embedder and FAISS index...")
    embedder = load_embedder(CFG)
    vector_db = load_faiss(CFG, embedder)

    print("🔄 Loading Gemma model...")
    tokenizer, model = load_gemma(CFG["MODEL_DIR"])

    if CFG.get("USE_REWRITER", True):
        print("🔄 Building rewriter pipeline...")
        gen_pipeline = build_rewriter(tokenizer, model, max_new_tokens=128)
    else:
        gen_pipeline = None

    print("✅ Startup complete.")


@app.route("/health", methods=["GET"])
def health():
    size = None
    try:
        size = int(getattr(vector_db.index, "ntotal", 0))
    except Exception:
        pass
    return jsonify({
        "status": "ok",
        "model_dir": CFG.get("MODEL_DIR"),
        "faiss_dir": CFG.get("FAISS_DIR"),
        "use_rewriter": bool(CFG.get("USE_REWRITER", True)),
        "top_k_default": int(CFG.get("TOP_K_DEFAULT", 5)),
        "index_size": size
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    k = int(data.get("k", CFG.get("TOP_K_DEFAULT", 5)))
    preferred_option = data.get("preferred_option", "Option 2")

    rewritten = rewrite_query(gen_pipeline, question, preferred_option=preferred_option) \
        if CFG.get("USE_REWRITER", True) else question

    hits = retrieve_top_k(vector_db, rewritten, k=k)
    ctx = "\n".join(h["passage"] for h in hits)

    answer = answer_with_gemma(
        question=question,
        context=ctx,
        tokenizer=tokenizer,
        model=model,
        gen_cfg=CFG
    )

    return jsonify({
        "question": question,
        "rewritten": rewritten if CFG.get("USE_REWRITER", True) else None,
        "answer": answer,
        "sources": hits
    })


@app.route("/reload", methods=["POST"])
def reload_index():
    global vector_db
    try:
        embedder = load_embedder(CFG)
        vector_db = load_faiss(CFG, embedder)
        size = int(getattr(vector_db.index, "ntotal", 0))
        return jsonify({"status": "reloaded", "index_size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    startup()  # Load resources before starting server
    app.run(host="0.0.0.0", port=int(CFG.get("PORT", 8080)), debug=False)
