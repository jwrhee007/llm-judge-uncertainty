from __future__ import annotations
"""
Step 1: 데이터 준비
- TriviaQA 서브셋 추출 (200문항)
- spaCy NER로 Answer Type 분류 (PERSON / DATE / LOCATION / NUMBER / ORG)
- Rule-based 오답 생성
    - 명백한 오답: Cross-Type Entity Swap
    - 헷갈리는 오답: Same-Type Entity Swap 또는 Numeric Perturbation

Usage:
    python -m src.prepare_data
    python -m src.prepare_data --smoke   # 5문항만 (테스트용)
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import spacy
from datasets import load_dataset

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DATA_PROCESSED,
    DATA_RAW,
    append_jsonl,
    load_config,
    setup_logger,
)

logger = setup_logger("prepare_data")


# =============================================================
# 1. TriviaQA 로드 및 서브셋 추출
# =============================================================
def load_triviaqa(config: dict) -> list[dict]:
    """
    HuggingFace에서 TriviaQA(rc) 로드 → 필터링 → 서브셋 샘플링.
    각 문항을 {question_id, question, context, answer, answer_aliases} 형태로 반환.
    """
    ds_cfg = config["dataset"]
    logger.info(f"Loading TriviaQA (subset={ds_cfg['subset']}, split={ds_cfg['split']})...")

    ds = load_dataset("trivia_qa", ds_cfg["subset"], split=ds_cfg["split"])
    logger.info(f"  Raw dataset size: {len(ds)}")

    # 지문 추출 + 필터링
    candidates = []
    for item in ds:
        # wiki_context를 우선 사용, 없으면 search_context
        context = _extract_context(item)
        if not context:
            continue

        # 지문 길이 필터 (단어 수 기준, 토큰 근사)
        word_count = len(context.split())
        if word_count < ds_cfg["min_context_length"]:
            continue
        if word_count > ds_cfg["max_context_length"]:
            # 너무 길면 앞부분만 잘라서 사용
            context = " ".join(context.split()[:ds_cfg["max_context_length"]])

        answer_value = item["answer"]["value"]
        if not answer_value or len(answer_value.strip()) == 0:
            continue

        candidates.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "context": context,
            "answer": answer_value.strip(),
            "answer_aliases": item["answer"].get("aliases", []),
        })

    logger.info(f"  After filtering: {len(candidates)} questions")

    # 랜덤 샘플링
    rng = random.Random(ds_cfg["seed"])
    n = ds_cfg["n_questions"]
    if len(candidates) < n:
        logger.warning(f"  Available ({len(candidates)}) < requested ({n}), using all")
        n = len(candidates)

    sampled = rng.sample(candidates, n)
    logger.info(f"  Sampled {len(sampled)} questions")
    return sampled


def _extract_context(item: dict) -> str | None:
    """TriviaQA 문항에서 가장 적합한 지문(context)을 추출."""
    # 1순위: entity_pages (Wikipedia)
    wiki_contexts = item.get("entity_pages", {}).get("wiki_context", [])
    if wiki_contexts:
        # 가장 긴 wiki context 사용
        return max(wiki_contexts, key=len)

    # 2순위: search_results
    search_contexts = item.get("search_results", {}).get("search_context", [])
    if search_contexts:
        return max(search_contexts, key=len)

    return None


# =============================================================
# 2. Answer Type 분류 (spaCy NER)
# =============================================================

# spaCy label → 우리 분류 매핑
SPACY_TO_TYPE = {
    # PERSON
    "PERSON": "PERSON",
    # DATE
    "DATE": "DATE",
    "TIME": "DATE",
    # LOCATION
    "GPE": "LOCATION",      # Geopolitical entity (국가, 도시 등)
    "LOC": "LOCATION",      # Non-GPE locations
    "FAC": "LOCATION",      # Facilities (공항, 다리 등)
    # NUMBER
    "CARDINAL": "NUMBER",
    "QUANTITY": "NUMBER",
    "MONEY": "NUMBER",
    "PERCENT": "NUMBER",
    "ORDINAL": "NUMBER",
    # ORG
    "ORG": "ORG",
    "NORP": "ORG",           # Nationalities, religious/political groups
}

# 숫자 패턴으로 DATE/NUMBER 추가 감지
YEAR_PATTERN = re.compile(r"^\d{4}$")
NUMBER_PATTERN = re.compile(r"^[\d,]+\.?\d*$")


def classify_answer_types(questions: list[dict], nlp) -> list[dict]:
    """
    각 문항의 정답에 대해 Answer Type을 분류.
    1차: spaCy NER (질문 컨텍스트 포함)
    2차: spaCy NER (정답 단독)
    3차: 패턴 매칭
    4차: 기본값 OTHER
    """
    logger.info("Classifying answer types with spaCy NER...")

    type_counts = {}
    for q in questions:
        ans_type = _classify_single(q["answer"], q["question"], nlp)
        q["answer_type"] = ans_type
        type_counts[ans_type] = type_counts.get(ans_type, 0) + 1

    logger.info(f"  Answer type distribution: {json.dumps(type_counts, indent=2)}")
    return questions


def _classify_single(answer: str, question: str, nlp) -> str:
    """단일 정답의 Answer Type 분류."""

    # 1차: 질문 + 정답 컨텍스트로 NER (짧은 정답도 맥락에서 인식 가능)
    context_text = f"{question} The answer is {answer}."
    doc = nlp(context_text)
    for ent in doc.ents:
        # 정답 텍스트와 겹치는 엔티티만 사용
        if answer.lower() in ent.text.lower() or ent.text.lower() in answer.lower():
            mapped = SPACY_TO_TYPE.get(ent.label_)
            if mapped:
                return mapped

    # 2차: 정답 단독 NER
    doc_answer = nlp(answer)
    for ent in doc_answer.ents:
        mapped = SPACY_TO_TYPE.get(ent.label_)
        if mapped:
            return mapped

    # 3차: 패턴 기반 fallback
    answer_stripped = answer.strip()
    if YEAR_PATTERN.match(answer_stripped):
        return "DATE"
    if NUMBER_PATTERN.match(answer_stripped.replace(",", "")):
        return "NUMBER"

    # 4차: 기본값
    return "OTHER"


# =============================================================
# 3. Rule-Based 오답 생성
# =============================================================
def generate_wrong_answers(questions: list[dict], config: dict) -> list[dict]:
    """
    각 문항에 대해 2종류의 오답을 생성:
    - obvious_wrong: Cross-Type Entity Swap (명백한 오답)
    - confusing_wrong: Same-Type Entity Swap 또는 Numeric Perturbation (헷갈리는 오답)
    """
    logger.info("Generating wrong answers (rule-based)...")
    wa_cfg = config["wrong_answers"]
    rng = random.Random(config["dataset"]["seed"] + 1)  # 오답 생성용 별도 시드

    # Answer Type별 정답 풀 구축
    type_pool: dict[str, list[str]] = {}
    for q in questions:
        t = q["answer_type"]
        if t not in type_pool:
            type_pool[t] = []
        type_pool[t].append(q["answer"])

    # 모든 타입 목록
    all_types = list(type_pool.keys())

    for q in questions:
        answer = q["answer"]
        ans_type = q["answer_type"]

        # --- 명백한 오답: Cross-Type Entity Swap ---
        q["obvious_wrong"] = _cross_type_swap(answer, ans_type, type_pool, all_types, rng)

        # --- 헷갈리는 오답: Same-Type Swap 또는 Numeric Perturbation ---
        q["confusing_wrong"] = _confusing_wrong(answer, ans_type, type_pool, wa_cfg, rng)

    # 검증
    n_fail_obvious = sum(
        1 for q in questions if q["obvious_wrong"].startswith("[NO_")
    )
    n_fail_confusing = sum(
        1 for q in questions if q["confusing_wrong"].startswith("[NO_")
    )
    if n_fail_obvious > 0:
        logger.warning(f"  ⚠ {n_fail_obvious} obvious_wrong generation failed (fallback)")
    if n_fail_confusing > 0:
        logger.warning(f"  ⚠ {n_fail_confusing} confusing_wrong generation failed (fallback)")

    logger.info("  Wrong answer generation complete")
    return questions


def _cross_type_swap(
    answer: str, ans_type: str, type_pool: dict, all_types: list, rng: random.Random
) -> str:
    """규칙 1: Cross-Type Entity Swap → 명백한 오답."""
    other_types = [t for t in all_types if t != ans_type and len(type_pool[t]) > 0]

    if not other_types:
        return "[NO_CROSS_TYPE_AVAILABLE]"

    chosen_type = rng.choice(other_types)
    candidates = [a for a in type_pool[chosen_type] if a != answer]

    if not candidates:
        return rng.choice(type_pool[chosen_type])

    return rng.choice(candidates)


def _confusing_wrong(
    answer: str, ans_type: str, type_pool: dict, wa_cfg: dict, rng: random.Random
) -> str:
    """규칙 2/3: Same-Type Swap 또는 Numeric Perturbation → 헷갈리는 오답."""
    np_cfg = wa_cfg["confusing"]["numeric_perturbation"]

    # DATE/NUMBER 타입은 Numeric Perturbation 우선 시도
    if ans_type in ("DATE", "NUMBER"):
        perturbed = _numeric_perturbation(answer, np_cfg, rng)
        if perturbed is not None:
            return perturbed

    # Same-Type Entity Swap
    same_type_pool = [a for a in type_pool.get(ans_type, []) if a != answer]
    if same_type_pool:
        return rng.choice(same_type_pool)

    # fallback 1: Numeric Perturbation 재시도 (타입 무관)
    perturbed = _numeric_perturbation(answer, np_cfg, rng)
    if perturbed is not None:
        return perturbed

    # fallback 2: 전체 풀에서 아무 다른 정답 가져오기 (최후 수단)
    all_answers = [a for answers in type_pool.values() for a in answers if a != answer]
    if all_answers:
        chosen = rng.choice(all_answers)
        logger.debug(f"  Confusing wrong fallback (any pool): '{answer}' → '{chosen}'")
        return chosen

    return "[NO_CONFUSING_AVAILABLE]"


def _numeric_perturbation(answer: str, np_cfg: dict, rng: random.Random) -> str | None:
    """규칙 3: Numeric Perturbation (DATE/NUMBER 전용)."""
    answer_stripped = answer.strip().replace(",", "")

    # 연도 패턴 (4자리 숫자)
    if YEAR_PATTERN.match(answer_stripped):
        year = int(answer_stripped)
        offset = rng.choice(np_cfg["year_offsets"])
        direction = rng.choice([-1, 1])
        return str(year + direction * offset)

    # 일반 숫자
    try:
        num = float(answer_stripped)
        factor = rng.choice(np_cfg["number_factors"])
        result = num * factor
        # 정수였으면 정수로 반환
        if num == int(num):
            return str(int(result))
        return f"{result:.2f}"
    except ValueError:
        pass

    # 숫자가 포함된 복합 문자열 (예: "42 million", "3rd")
    match = re.search(r"(\d+)", answer)
    if match:
        num = int(match.group(1))
        offset = rng.choice(np_cfg["year_offsets"])
        direction = rng.choice([-1, 1])
        perturbed_num = max(0, num + direction * offset)
        return answer.replace(match.group(1), str(perturbed_num))

    return None


# =============================================================
# 4. 저장
# =============================================================
def save_dataset(questions: list[dict], output_name: str = "evaluation_set.jsonl") -> Path:
    """
    전처리된 데이터셋을 JSONL로 저장.
    각 문항 × 3 답변유형 = 평가 세트로 펼쳐서 저장.
    """
    output_path = DATA_PROCESSED / output_name

    # 기존 파일 삭제 (재실행 시 중복 방지)
    if output_path.exists():
        output_path.unlink()

    count = 0
    for q in questions:
        base = {
            "question_id": q["question_id"],
            "question": q["question"],
            "context": q["context"],
            "ground_truth": q["answer"],
            "answer_type_ner": q["answer_type"],
            "question_length": len(q["question"].split()),
            "context_length": len(q["context"].split()),
        }

        # 정답 세트
        append_jsonl(output_path, {**base, "answer": q["answer"], "answer_category": "correct"})
        count += 1

        # 명백한 오답 세트
        append_jsonl(output_path, {**base, "answer": q["obvious_wrong"], "answer_category": "obvious_wrong"})
        count += 1

        # 헷갈리는 오답 세트
        append_jsonl(output_path, {**base, "answer": q["confusing_wrong"], "answer_category": "confusing_wrong"})
        count += 1

    logger.info(f"Saved {count} evaluation sets to {output_path}")
    return output_path


def save_questions_raw(questions: list[dict], output_name: str = "questions_200.jsonl") -> Path:
    """중간 결과: 문항 + Answer Type + 오답 포함 원본 저장."""
    output_path = DATA_PROCESSED / output_name
    if output_path.exists():
        output_path.unlink()

    for q in questions:
        append_jsonl(output_path, q)

    logger.info(f"Saved {len(questions)} raw questions to {output_path}")
    return output_path


# =============================================================
# 5. 통계 출력
# =============================================================
def print_summary(questions: list[dict]) -> None:
    """데이터셋 요약 통계 출력."""
    logger.info("=" * 60)
    logger.info("Dataset Summary")
    logger.info("=" * 60)
    logger.info(f"  Total questions: {len(questions)}")
    logger.info(f"  Total evaluation sets: {len(questions) * 3}")

    # Answer Type 분포
    type_dist: dict[str, int] = {}
    for q in questions:
        t = q["answer_type"]
        type_dist[t] = type_dist.get(t, 0) + 1
    logger.info("  Answer Type distribution:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        logger.info(f"    {t:12s}: {c:3d} ({c / len(questions) * 100:.1f}%)")

    # 오답 생성 실패 체크
    n_fail_obvious = sum(1 for q in questions if q["obvious_wrong"].startswith("[NO_"))
    n_fail_confusing = sum(1 for q in questions if q["confusing_wrong"].startswith("[NO_"))
    if n_fail_obvious > 0:
        logger.warning(f"  ⚠ Obvious wrong fallback: {n_fail_obvious}")
    if n_fail_confusing > 0:
        logger.warning(f"  ⚠ Confusing wrong fallback: {n_fail_confusing}")

    # 샘플 출력
    logger.info("-" * 60)
    logger.info("Sample (first 3 questions):")
    for q in questions[:3]:
        logger.info(f"  Q: {q['question'][:80]}...")
        logger.info(f"    Answer:          {q['answer']} ({q['answer_type']})")
        logger.info(f"    Obvious Wrong:   {q['obvious_wrong']}")
        logger.info(f"    Confusing Wrong: {q['confusing_wrong']}")
        logger.info("")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 1: Prepare TriviaQA evaluation dataset")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (5 questions)")
    parser.add_argument("--config", default="experiment.yaml", help="Config file name")
    args = parser.parse_args()

    config = load_config(args.config)

    # 스모크 테스트 모드
    if args.smoke:
        config["dataset"]["n_questions"] = config["smoke_test"]["n_questions"]
        logger.info("[SMOKE TEST MODE] n_questions = %d", config["dataset"]["n_questions"])

    # Step 1.1: TriviaQA 로드 및 서브셋 추출
    questions = load_triviaqa(config)

    # Step 1.2: spaCy NER로 Answer Type 분류
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    questions = classify_answer_types(questions, nlp)

    # Step 1.3: Rule-based 오답 생성
    questions = generate_wrong_answers(questions, config)

    # Step 1.4: 저장
    suffix = "_smoke" if args.smoke else ""
    save_questions_raw(questions, f"questions{suffix}.jsonl")
    save_dataset(questions, f"evaluation_set{suffix}.jsonl")

    # 요약 출력
    print_summary(questions)
    logger.info("Done! Next step: python -m src.run_judge" + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()