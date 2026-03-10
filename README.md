# LLM-Judge Semantic Uncertainty 관측 연구

## Overview

LLM-as-a-Judge의 판단 불안정성(Semantic Uncertainty)을 정량적으로 관측하는 실험.
TriviaQA 기반 QA 태스크에서 GPT-4o-mini Judge의 반복 판단 분산을 Semantic Entropy로 측정한다.

## Research Questions

* **RQ1**: 판단 불안정성은 결정론적 환경(T=0, prompt 고정)에서도 발생하는가? *(Phase A — 완료)*
* **RQ2**: 프롬프트 구조(verdict 가짓수, few-shot, PKI 억제 방식)가 semantic entropy에 어떤 영향을 미치는가? *(Phase B — 현재)*
* **RQ3**: 높은 semantic entropy는 높은 judgement 오류율과 관련 있는가?

## Branch Structure

| Branch | 내용 |
|--------|------|
| `main` | Phase B (B1-1: 3-Prompt Comparison) |
| `phase-a` | Phase A (Baseline, 200문항 × 1프롬프트 × 30회) |

## Project Structure

```
llm-judge-uncertainty/
├── configs/
│   ├── experiment.yaml           # Phase A config
│   └── experiment_b1.yaml        # Phase B B1-1 config
├── src/
│   ├── __init__.py
│   ├── prompts.py                # 3-Prompt definitions + parsers
│   ├── prepare_data.py           # Step 1: 층화 추출 + evidence-aware context
│   ├── run_judge_batch.py        # Step 2: 3-Prompt Batch API 실행
│   ├── analyze.py                # Step 3: H/H_norm + cross-prompt comparison
│   └── utils.py                  # 공통 유틸리티
├── data/processed/               # 전처리된 데이터셋
├── results/analysis/             # 분석 결과
├── tests/
└── README.md
```

## Phase B Setup

### 1. 환경 생성

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

### 3. 데이터 준비 (spaCy native NER 태깅 + 층화 추출)

```bash
# 스모크 테스트 (태그당 2개 → ~30문항)
python -m src.prepare_data --config experiment_b1.yaml --smoke

# 풀 실험 (태그당 20개 → ~300문항)
python -m src.prepare_data --config experiment_b1.yaml
```

### 4. Judge 실험 실행 (3-Prompt × 30회)

```bash
# 스모크 테스트
python -m src.run_judge_batch auto --config experiment_b1.yaml --smoke

# 풀 실험
python -m src.run_judge_batch auto --config experiment_b1.yaml
```

### 5. 분석

```bash
python -m src.analyze --config experiment_b1.yaml
```

## Phase B B1-1: 3-Prompt Comparison

3개 프롬프트로 동일 문항을 평가하여 비교:

| Prompt | Source | Verdict K | Type |
|--------|--------|-----------|------|
| P-Thakur | Thakur et al. (2024) | 2 (CORRECT/INCORRECT) | Zero-shot |
| P-CLEV | Choudhary et al. (2024) | 3 (True/False/under review) | Zero-shot role-playing |
| P-Lee-Standard | Lee et al. (2026) | 3 (CORRECT/INCORRECT/NOT_ATTEMPTED) | Few-shot |

**Key Metrics**:
- **H (raw SE)**: 프롬프트 내 비교용
- **H_norm = H / ln(K)**: 프롬프트 간 비교용 (verdict 가짓수 정규화)

## Phase A (Baseline) — 완료

* T=0, seed 고정, prompt 고정
* 200문항 × 3답변유형 × 30회 반복
* `phase-a` 브랜치에서 확인 가능
