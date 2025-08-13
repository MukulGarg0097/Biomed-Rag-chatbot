from typing import Optional, Dict, Any
import re, torch,os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .intent import (
    detect_question_intent,
    filter_context_for_intent,
    INTENT_TEMPLATES,
    FALLBACK_LINE,
)

def device_kind() -> str:
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_gemma(model_dir):
    # Ensure absolute path
    model_dir = os.path.abspath(model_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading Gemma model from: {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

    return tok, model

def build_rewriter(tokenizer, model, max_new_tokens: int = 128):
    dev = device_kind()
    dev_idx = 0 if dev == "cuda" else -1  # HF pipeline expects int index for CUDA else -1 (CPU)
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
