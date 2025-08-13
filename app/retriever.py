import os, pickle
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS

import os
import joblib
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ updated import

def load_embedder(cfg):
    """
    Load a HuggingFaceEmbeddings object for use with FAISS.
    1. Try to load from pickle.
    2. If missing or invalid, rebuild from model name/directory.
    3. Save rebuilt embedder to pickle for next time.
    """
    embedder_pkl = os.path.abspath(cfg["EMBEDDER_PKL"])
    embedding_name_or_dir = cfg["EMBEDDING_NAME_OR_DIR"]

    # 1. Load from pickle if available
    if os.path.exists(embedder_pkl):
        try:
            embedder = joblib.load(embedder_pkl)
            if isinstance(embedder, HuggingFaceEmbeddings):
                print(f"✅ Embedder loaded from pickle: {embedder_pkl}")
                return embedder
            else:
                print(f"⚠ Pickle found but not a HuggingFaceEmbeddings object. Rebuilding from model name.")
        except Exception as e:
            print(f"⚠ Failed to load embedder pickle: {e}. Rebuilding from model name.")

    # 2. Build a fresh embedder
    print(f"🔄 Building new embedder from: {embedding_name_or_dir}")
    embedder = HuggingFaceEmbeddings(model_name=embedding_name_or_dir)

    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(embedder_pkl), exist_ok=True)

    # 3. Save for future use
    joblib.dump(embedder, embedder_pkl)
    print(f"✅ Saved embedder to: {embedder_pkl}")

    return embedder



def load_faiss(cfg, embedder):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # /.../Project/app
    faiss_dir = os.path.join(base_dir, "index", "faiss_index_folder")

    print(f"FAISS_DIR: {faiss_dir}")
    if not os.path.exists(faiss_dir):
        raise RuntimeError(f"FAISS_DIR not found: {faiss_dir}")
    
    return FAISS.load_local(faiss_dir, embedder, allow_dangerous_deserialization=True)

def retrieve_top_k(vs: FAISS, query: str, k: int) -> List[Dict[str, Any]]:
    docs = vs.similarity_search(query, k=k)
    return [{"passage": d.page_content, "doc_id": int(d.metadata.get("doc_id"))} for d in docs]
