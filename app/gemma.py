from typing import Dict, Any
import os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download

from .intent import (
    detect_question_intent,
    filter_context_for_intent,
    INTENT_TEMPLATES,
    FALLBACK_LINE,
)

# Default to base Gemma 2 (2B). Use the -it variant if you want chat-tuned behavior:
#   "google/gemma-2-2b-it"
DEFAULT_HF_REPO = os.getenv("HF_MODEL_REPO", "google/gemma-2-2b")
DEFAULT_REV = os.getenv("HF_MODEL_REVISION")  # optional: pin a tag/commit for reproducibility

def device_kind() -> str:
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _ensure_local_model(model_dir: str, repo_id: str, revision: str | None) -> str:
    """
    If model_dir exists & has files, return it.
    Otherwise download from Hugging Face Hub into model_dir.
    Uses HF_TOKEN implicitly if present in env.
    """
    model_dir = os.path.abspath(model_dir)
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        print(f"Loading Gemma model from local path: {model_dir}")
        return model_dir

    print(f"Downloading Gemma model from Hub: {repo_id} -> {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    # This writes the snapshot into model_dir so later runs can be offline
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        # Respect TRANSFORMERS_OFFLINE if user sets it, but default to download
        allow_patterns=None,
        ignore_patterns=None,
    )
    return model_dir

def load_gemma(model_dir: str | None = None, repo_id: str | None = None, revision: str | None = DEFAULT_REV):
    """
    Preferred call:
        load_gemma(CFG["MODEL_DIR"])               # uses local dir if present, else downloads DEFAULT_HF_REPO
    Or:
        load_gemma(CFG["MODEL_DIR"], "google/gemma-2-2b-it")  # force specific repo
    """
    repo_id = repo_id or DEFAULT_HF_REPO
    if model_dir is None:
        # Fallback local cache path if caller didn’t provide one
        model_dir = "./app/models/gemma"

    local_path = _ensure_local_model(model_dir, repo_id, revision)

    # If you run fully offline later, TRANSFORMERS_OFFLINE=1 will make these calls hit the local_path only
    tok = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    # dtype ‘auto’ is safe; on CPU it will pick float32
    model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype="auto")

    return tok, model

def build_rewriter(tokenizer, model, max_new_tokens: int = 128):
    dev = device_kind()
    dev_idx = 0 if dev == "cuda" else -1
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        device=dev_idx,
    )

def make_rewrite_prompt(question: str) -> str:
    return (
        "Provide several specific rewritten versions of the biomedical question, "
        "ranging from broad to precise.\nOutput as 'Option 1', 'Option 2', etc.\n\n"
        f"Question: {question}\nRewritten:"
    )

def rewrite_query(gen_pipeline, question: str, preferred_option: str = "Option 2") -> str:
    if gen_pipeline is None:
        return question
    out = gen_pipeline(make_rewrite_prompt(question))[0]["generated_text"]
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    for line in lines:
        if line.startswith(preferred_option):
            return line.split(":", 1)[-1].strip() or question
    for line in lines:
        if line.startswith("Option 1"):
            return line.split(":", 1)[-1].strip() or question
    return question

@torch.no_grad()
def answer_with_gemma(
    question: str,
    context: str,
    tokenizer,
    model,
    gen_cfg: Dict[str, Any],
) -> str:
    intent = detect_question_intent(question)
    focused_context = filter_context_for_intent(context, intent)
    if not focused_context.strip():
        return FALLBACK_LINE

    prompt = INTENT_TEMPLATES[intent].format(context=focused_context, question=question)
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=int(gen_cfg["MAX_NEW_TOKENS"]),
        num_beams=int(gen_cfg["NUM_BEAMS"]),
        do_sample=False,
        no_repeat_ngram_size=int(gen_cfg["NO_REPEAT_NGRAM_SIZE"]),
        repetition_penalty=float(gen_cfg["REPETITION_PENALTY"]),
        length_penalty=float(gen_cfg["LENGTH_PENALTY"]),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = raw.replace(prompt, "").strip()

    if not answer or answer.lower().startswith("the context does not") or "cannot answer" in answer.lower():
        return FALLBACK_LINE

    if intent == "causes" and not re.search(r"\b(caused by|due to|results? from|because)\b", answer, re.I):
        if len(answer) < 30:
            return FALLBACK_LINE
    return answer
