import re
from typing import List

FALLBACK_LINE = "I'm sorry, I cannot answer that question based on the provided information."

INTENT_PATTERNS = {
    "symptoms":  re.compile(r"\b(symptom|sign|clinical presentation|manifestation)\b", re.I),
    "causes":    re.compile(r"\b(cause|etiolog|due to|result[s]? from|lead[s]? to|because)\b", re.I),
    "treatments":re.compile(r"\b(treat|therapy|management|intervention|drug|medication)\b", re.I),
    "risks":     re.compile(r"\b(risk factor|risk|predispos|associated with|correlate)\b", re.I),
    "mechanisms":re.compile(r"\b(pathophysiolog|mechanism|how.*work|underlying process)\b", re.I),
    "definition":re.compile(r"\b(what is|define|definition)\b", re.I),
}

def detect_question_intent(question: str) -> str:
    q = question.lower()
    for intent, pat in INTENT_PATTERNS.items():
        if pat.search(q): return intent
    if any(w in q for w in ["cause","etiology","why"]): return "causes"
    if any(w in q for w in ["symptom","sign","presentation"]): return "symptoms"
    if any(w in q for w in ["treat","therapy","manage","medicat"]): return "treatments"
    if any(w in q for w in ["risk","predispos"]): return "risks"
    if any(w in q for w in ["mechanism","how does it work","pathophys"]): return "mechanisms"
    if q.startswith("what is") or q.startswith("define"): return "definition"
    return "general"

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")

def split_sentences(text: str) -> List[str]:
    import re as _re
    text = _re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

INTENT_CUE_SETS = {
    "causes": [r"\b(cause|caused by|etiolog\w*|due to|because|results? from|triggered by|lead[s]? to)\b"],
    "symptoms":[r"\b(symptom|signs?|presents? with|manifestation)\b"],
    "treatments":[r"\b(treat|therapy|therapies|management|intervention|drug|medication|dose|dosing)\b"],
    "risks":   [r"\b(risk factor|risk|predispos|associated with|correlat)\b"],
    "mechanisms":[r"\b(pathophysiolog\w*|mechanism|biologic\w* process|immune|inflammation|autoimmun\w*)\b"],
    "definition":[r"\b(is defined as|refers to|is a|means)\b"],
    "general": []
}

def filter_context_for_intent(context: str, intent: str, max_sents: int = 15) -> str:
    sents = split_sentences(context)
    if not sents: return ""
    cues = INTENT_CUE_SETS.get(intent, [])
    if not cues:
        ranked = sorted(sents, key=len, reverse=True)[:max_sents]
        return " ".join(ranked)
    compiled = [re.compile(pat, re.I) for pat in cues]
    scored = []
    for s in sents:
        score = sum(1 for pat in compiled if pat.search(s))
        if score > 0: scored.append((score, len(s), s))
    if not scored:
        ranked = sorted(sents, key=len, reverse=True)[:min(max_sents, 8)]
        return " ".join(ranked)
    ranked = sorted(scored, key=lambda x: (-x[0], -x[1]))[:max_sents]
    return " ".join(s for _, __, s in ranked)

INTENT_TEMPLATES = {
    "causes": """You are a biomedical expert.
Using ONLY the context, answer the question by listing the CAUSES/ETIOLOGY explicitly mentioned.
Rules:
- Extract only statements that indicate causation (e.g., "caused by", "due to", "results from").
- If causes are not directly stated, say exactly: "I'm sorry, I cannot answer that question based on the provided information."
- Be concise and structured (bullets).

Context:
{context}

Question:
{question}

Answer:
""",
    "symptoms": """You are a biomedical expert.
Using ONLY the context, list ALL symptoms/signs mentioned, grouped logically (respiratory, neurological, cardiovascular, GI, mental health, etc.). Do not add anything not stated.

Context:
{context}

Question:
{question}

Answer:
""",
    "treatments": """You are a biomedical expert.
Using ONLY the context, summarize evidence-based treatments/management mentioned (drugs, interventions, dose notes if present). If none, use the exact fallback line.

Context:
{context}

Question:
{question}

Answer:
""",
    "risks": """You are a biomedical expert.
Using ONLY the context, list risk factors and associations mentioned. If none are present, use the exact fallback line.

Context:
{context}

Question:
{question}

Answer:
""",
    "mechanisms": """You are a biomedical expert.
Using ONLY the context, explain the pathophysiology/mechanisms mentioned. If mechanisms are not described, use the exact fallback line.

Context:
{context}

Question:
{question}

Answer:
""",
    "definition": """You are a biomedical expert.
Using ONLY the context, provide a crisp definition/description. If no definitional text is present, use the exact fallback line.

Context:
{context}

Question:
{question}

Answer:
""",
    "general": """You are a biomedical expert.
Using ONLY the context, answer as clearly and completely as possible.
- Combine relevant points concisely.
- Do not invent information not in the context.
- If nothing in the context supports an answer, use the exact fallback line.

Context:
{context}

Question:
{question}

Answer:
"""
}
