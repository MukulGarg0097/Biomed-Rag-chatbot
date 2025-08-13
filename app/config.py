import json, os
from typing import Any, Dict

_DEFAULTS: Dict[str, Any] = {
    "MODEL_DIR": "./app/models/gemma",
    "FAISS_DIR": "./app/index/faiss_index_folder",
    "EMBEDDER_PKL": "./app/embedder_model_folder/embedding_model.pkl",
    # if you don’t want to rely on the pickle, ensure this matches what built the index:
    "EMBEDDING_NAME_OR_DIR": "sentence-transformers/all-MiniLM-L6-v2",
    "EMBEDDER_DIR": "./app/embedder_model_folder",
    
    "USE_REWRITER": True,
    "TOP_K_DEFAULT": 5,

    "MAX_NEW_TOKENS": 320,
    "NUM_BEAMS": 4,
    "NO_REPEAT_NGRAM_SIZE": 3,
    "REPETITION_PENALTY": 1.05,
    "LENGTH_PENALTY": 0.9,

    "TRANSFORMERS_OFFLINE": True,
    "PORT": 8080
}

# optional env overrides (use same keys as config.json if you want)
_ENV_OVERRIDES = {
    "MODEL_DIR", "FAISS_DIR", "EMBEDDER_PKL", "EMBEDDING_NAME_OR_DIR",
    "USE_REWRITER", "TOP_K_DEFAULT", "MAX_NEW_TOKENS", "NUM_BEAMS",
    "NO_REPEAT_NGRAM_SIZE", "REPETITION_PENALTY", "LENGTH_PENALTY",
    "TRANSFORMERS_OFFLINE", "PORT"
}

def load_config(path: str = "./config.json") -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    # env overrides take precedence when present
    for k in _ENV_OVERRIDES:
        if k in os.environ:
            v = os.getenv(k)
            if v is None:
                continue
            # cast numeric/bool
            if k in {"TOP_K_DEFAULT", "MAX_NEW_TOKENS", "NUM_BEAMS", "NO_REPEAT_NGRAM_SIZE", "PORT"}:
                try: v = int(v)
                except: pass
            if k in {"REPETITION_PENALTY", "LENGTH_PENALTY"}:
                try: v = float(v)
                except: pass
            if k in {"USE_REWRITER", "TRANSFORMERS_OFFLINE"}:
                v = v.lower() in {"1","true","yes","on"}
            cfg[k] = v

    # Ensure paths are absolute
    base_dir = os.path.dirname(os.path.abspath(__file__))  # /.../Project/app
    for key in ["MODEL_DIR", "FAISS_DIR", "EMBEDDER_PKL", "EMBEDDER_DIR"]:
        if key in cfg:
            cfg[key] = os.path.abspath(os.path.join(base_dir, "..", cfg[key].replace("./", "")))

    return cfg
