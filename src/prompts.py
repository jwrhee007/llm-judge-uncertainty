"""
Phase B Prompt Definitions
- P-Thakur:       2-way (CORRECT / INCORRECT), zero-shot
- P-CLEV:         3-way (True / False / under review), zero-shot role-playing
- P-Lee-Standard: 3-way (CORRECT / INCORRECT / NOT_ATTEMPTED), few-shot

각 프롬프트는 dict 형태로 정의:
  {
    "id": str,
    "system": str,            # system message (없으면 "")
    "user_template": str,     # {context}, {question}, {candidate_answer} 포함
    "verdict_k": int,         # verdict 가짓수 (SE 정규화 분모)
    "parse_fn": callable,     # raw response → unified verdict label
  }
"""
from __future__ import annotations

import re

# =============================================================
# Unified Verdict Labels (SE 계산용)
# =============================================================
POSITIVE = "POSITIVE"      # 정답 지지
NEGATIVE = "NEGATIVE"      # 오답 / 비지지
UNCERTAIN = "UNCERTAIN"    # 판단 보류
PARSE_ERROR = "PARSE_ERROR"


# =============================================================
# P-Thakur: Thakur et al. (2024) — 2-way, zero-shot
# =============================================================
_P_THAKUR_USER = """\
Your task is to look at the following question, and based on the context \
passage provided, determine if the candidate answer is correct or incorrect.

Evaluate ONLY based on the information explicitly stated in the context \
passage. Do NOT use your own knowledge or assumptions beyond what is \
provided in the context.

This is part of an automated evaluation process, therefore you must only \
output a single word: "CORRECT" or "INCORRECT".

Context Passage: {context}
Question: {question}
Candidate Answer: {candidate_answer}"""


def _parse_thakur(raw: str) -> str:
    """P-Thakur 응답 파싱: CORRECT / INCORRECT → POSITIVE / NEGATIVE."""
    upper = raw.strip().upper()
    # "INCORRECT"를 먼저 체크 (CORRECT가 부분 매칭되는 것 방지)
    if "INCORRECT" in upper:
        return NEGATIVE
    if "CORRECT" in upper:
        return POSITIVE
    return PARSE_ERROR


P_THAKUR = {
    "id": "P-Thakur",
    "system": "",
    "user_template": _P_THAKUR_USER,
    "verdict_k": 2,
    "parse_fn": _parse_thakur,
}


# =============================================================
# P-CLEV: Choudhary et al. (2024, 2025) — 3-way, zero-shot
# =============================================================
_P_CLEV_USER = """\
As an evaluator, your task is to assess whether a candidate answer is supported \
by a given context passage. Each evaluation task consists of three parts: a \
context passage, which contains the source information; a question, which was \
asked about the context; and a candidate answer, which is the response to \
evaluate.

Here's how to score each response:

- Assign a score of '1' (True) if the candidate answer is directly supported by \
information explicitly stated in the context passage. The answer must be \
verifiable from the context alone, and should directly address the question in \
a manner consistent with the context.

- Assign a score of '0' (False) if the candidate answer is not supported by \
the context passage, if the context lacks sufficient information to verify the \
answer, or if the answer contradicts the context.

Your role requires impartiality and objectivity. It is crucial to evaluate each \
response based ONLY on the provided context passage, without relying on external \
knowledge or prior beliefs about the topic. Treat all responses uniformly, \
ensuring a fair and consistent assessment across all tasks. If you encounter \
ambiguities or are unsure about how to judge a response, mark it as "under \
review".

Context Passage: {context}
Question: {question}
Candidate Answer: {candidate_answer}"""


def _parse_clev(raw: str) -> str:
    """P-CLEV 응답 파싱: 1/True → POSITIVE, 0/False → NEGATIVE, under review → UNCERTAIN."""
    text = raw.strip().lower()
    # "under review" 먼저 체크
    if "under review" in text:
        return UNCERTAIN
    # score 패턴: '1' 또는 'true'
    if re.search(r"\b1\b", text) or "true" in text:
        return POSITIVE
    # score 패턴: '0' 또는 'false'
    if re.search(r"\b0\b", text) or "false" in text:
        return NEGATIVE
    return PARSE_ERROR


P_CLEV = {
    "id": "P-CLEV",
    "system": "",
    "user_template": _P_CLEV_USER,
    "verdict_k": 3,
    "parse_fn": _parse_clev,
}


# =============================================================
# P-Lee-Standard: Lee et al. (2026) — 3-way, few-shot
# =============================================================
_P_LEE_STANDARD_USER = """\
Your job is to look at a question, a context passage, and a candidate answer, \
and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
The context passage is the ONLY source of information you should use. First, I \
will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT candidate answers.
"' Question: What river runs through the capital city?
Context passage: The Danube river flows through Budapest, the capital of Hungary, \
dividing it into Buda on the west bank and Pest on the east bank.
Candidate answer 1: The Danube.
Candidate answer 2: the danube river runs through the capital
Candidate answer 3: Budapest is divided by the Danube river, which flows through \
the city separating Buda and Pest. "'
These candidate answers are all CORRECT because:
- The answer is directly supported by information explicitly stated in the \
context passage.
- The answer does not contain information that contradicts the context.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order \
don't matter.
- Hedging is permissible, provided the answer is supported by the context and \
contains no information contradicting it.

The following are examples of INCORRECT candidate answers.
"' Question: What river runs through the capital city?
Context passage: The Danube river flows through Budapest, the capital of Hungary, \
dividing it into Buda on the west bank and Pest on the east bank.
Candidate answer 1: The Rhine river.
Candidate answer 2: The Danube and the Rhine both flow through Budapest.
Candidate answer 3: No river runs through the capital.
Candidate answer 4: I think it's either the Danube or the Vistula. "'
These candidate answers are all INCORRECT because:
- A factual statement in the answer contradicts the context passage, or the \
answer includes information not supported by the context. Incorrect statements \
with hedging (e.g., "I think", "it is possible that") are also considered \
incorrect.

The following are examples of NOT_ATTEMPTED candidate answers.
"' Question: What river runs through the capital city?
Context passage: The Danube river flows through Budapest, the capital of Hungary, \
dividing it into Buda on the west bank and Pest on the east bank.
Candidate answer 1: I don't know.
Candidate answer 2: I need more context to answer this.
Candidate answer 3: Budapest is the capital of Hungary, but I'm not sure which \
river flows through it. "'
These candidate answers are all NOT_ATTEMPTED because:
- The important information from the context is not included in the answer.
- No statements in the answer contradict the context passage.

Also note the following things:
- You must evaluate the candidate answer ONLY based on the provided context \
passage. Do NOT use your own knowledge or beliefs.
- Even if you know the candidate answer is factually correct from your own \
knowledge, you must mark it INCORRECT if the context passage does not support \
it or if it contradicts the context.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order \
don't matter.
- If the context contains more information than needed to answer the question, \
the candidate answer only needs to address the question.
- Do not punish for typos if the intended meaning is clearly the same.

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT_ATTEMPTED.
Don't apologize or correct yourself if there was a mistake; we are just trying \
to grade the answer.
"' Question: {question}
Context passage: {context}
Candidate answer: {candidate_answer} "'
Grade the candidate answer of this new question as one of:
A: CORRECT B: INCORRECT C: NOT_ATTEMPTED
Just return the letters "A", "B", or "C", with no text around it."""


def _parse_lee_standard(raw: str) -> str:
    """P-Lee-Standard 응답 파싱: A → POSITIVE, B → NEGATIVE, C → UNCERTAIN."""
    text = raw.strip().upper()
    # 단일 문자 우선
    if text in ("A", "B", "C"):
        return {"A": POSITIVE, "B": NEGATIVE, "C": UNCERTAIN}[text]
    # 키워드 fallback — NOT_ATTEMPTED를 INCORRECT보다 먼저 체크
    if "NOT_ATTEMPTED" in text or "NOT ATTEMPTED" in text:
        return UNCERTAIN
    if "INCORRECT" in text:
        return NEGATIVE
    if "CORRECT" in text:
        return POSITIVE
    # 'A:', 'B:', 'C:' 패턴
    m = re.search(r"\b([ABC])\b", text)
    if m:
        return {"A": POSITIVE, "B": NEGATIVE, "C": UNCERTAIN}[m.group(1)]
    return PARSE_ERROR


P_LEE_STANDARD = {
    "id": "P-Lee-Standard",
    "system": "",
    "user_template": _P_LEE_STANDARD_USER,
    "verdict_k": 3,
    "parse_fn": _parse_lee_standard,
}


# =============================================================
# Registry
# =============================================================
PROMPT_REGISTRY: dict[str, dict] = {
    "P-Thakur": P_THAKUR,
    "P-CLEV": P_CLEV,
    "P-Lee-Standard": P_LEE_STANDARD,
}


def get_prompt(prompt_id: str) -> dict:
    """프롬프트 ID로 프롬프트 정의를 반환."""
    if prompt_id not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt: {prompt_id}. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[prompt_id]


def get_active_prompts(config: dict) -> list[dict]:
    """Config의 active 목록에서 프롬프트 정의들을 반환."""
    active_ids = config["prompts"]["active"]
    return [get_prompt(pid) for pid in active_ids]
