from typing import Optional
import os
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .config import load_config
from .retriever import load_embedder, load_faiss, retrieve_top_k
from .gemma import load_gemma, build_rewriter, rewrite_query, answer_with_gemma

# -----------------------------
# Flask app (serves UI + API)
# -----------------------------
app = Flask(__name__, static_folder="static")
CORS(app)

@app.route("/", methods=["GET"])
def index():
    # serves app/static/index.html
    return send_from_directory(app.static_folder, "index.html")

# -----------------------------
# Config & Globals
# -----------------------------
CFG = load_config(os.getenv("CONFIG_PATH", "./config.json"))

# Optional: keep transformers offline after first download
if CFG.get("TRANSFORMERS_OFFLINE", False):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

tokenizer = None
model = None
gen_pipeline = None
vector_db = None

# -----------------------------
# Device helpers
# -----------------------------
def device_kind() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon Metal (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"

def move_model_to_device(m, dev: str):
    """
    Safely move model to target device. Use half/bfloat16 on CUDA when possible
    for memory/perf; keep float32 on CPU. MPS generally works best with float32/16
    depending on the model—here we keep default dtype for safety.
    """
    try:
        if dev == "cuda":
            # Prefer bfloat16 if supported (Ampere+), else float16
            # Fall back to default if conversion fails.
            try:
                m = m.to(dtype=torch.bfloat16)
            except Exception:
                try:
                    m = m.to(dtype=torch.float16)
                except Exception:
                    pass
            m = m.to(device="cuda", non_blocking=True)
        elif dev == "mps":
            # MPS supports float32/16; keep model dtype as-is for stability.
            m = m.to(device="mps")
        else:
            # CPU
            m = m.to(device="cpu")
    except Exception as e:
        print(f"⚠️ Could not move model to {dev}: {e}. Using CPU.")
        m = m.to(device="cpu")
    return m

# -----------------------------
# Startup
# -----------------------------
def startup():
    """Load all required models and indexes into memory."""
    global tokenizer, model, gen_pipeline, vector_db

    print("🔄 Loading embedder and FAISS index...")
    embedder = load_embedder(CFG)
    vector_db = load_faiss(CFG, embedder)

    print("🔄 Loading Gemma model...")
    tokenizer, model = load_gemma(CFG["MODEL_DIR"])

    # Move to best device
    dev = device_kind()
    model = move_model_to_device(model, dev)
    print(f"✅ Model device: {model.device}")

    if CFG.get("USE_REWRITER", True):
        print("🔄 Building rewriter pipeline...")
        # build_rewriter internally sets device index (cuda:0 => 0, else -1)
        gen_pipeline = build_rewriter(tokenizer, model, max_new_tokens=128)
    else:
        gen_pipeline = None

    print("✅ Startup complete.")

# -----------------------------
# Routes
# -----------------------------
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
        "index_size": size,
        "device": str(model.device) if model is not None else "uninitialized"
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
    # Expect hits as list of dicts containing 'passage'—adjust if your retriever returns docs
    ctx = "\n".join(h.get("passage", str(h)) for h in hits)

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

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    startup()  # Load resources before starting server
    # NOTE: If you want CUDA to be used inside Docker, run container with: --gpus all
    app.run(host="0.0.0.0", port=int(CFG.get("PORT", 8080)), debug=False)
