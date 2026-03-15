# LLM-Judge Semantic Uncertainty 관측 연구

## Overview

LLM-as-a-Judge의 판단 불안정성(Semantic Uncertainty)을 정량적으로 관측하는 실험.
TriviaQA 기반 QA 태스크에서 GPT-4o-mini Judge의 반복 판단 분산을 Semantic Entropy로 측정한다.

## Branch Structure

| Branch | 내용 |
|--------|------|
| `main` | Phase B (B-1 Evidence-Present / B-2 Evidence-Absent) |
| `phase-a` | Phase A (Baseline, 200문항 × 1프롬프트 × 30회) |

## Phase B Structure

```
Phase B
├── B-1: Evidence-Present (original context)
│   ├── B1-1: 3-Prompt Comparison (P-Thakur, P-CLEV, P-Lee-Standard)
│   └── B1-2: Attention Dilution (distractor scaling) [향후]
│
└── B-2: Evidence-Absent (swapped context)
    ├── B2-1: Same-Type Swap → PKI 유도
    └── B2-2: Cross-Type Swap → 통제 조건
```

## Project Structure

```
llm-judge-uncertainty/
├── configs/
│   └── experiment_b.yaml        # Phase B 통합 config
├── src/
│   ├── prompts.py               # 3-Prompt definitions + parsers
│   ├── prepare_data.py          # 층화추출 + evidence-aware + context swap
│   ├── run_judge_batch.py       # Experiment-aware Batch API
│   ├── analyze.py               # B1-1 분석 + B2 PKI 분석
│   └── utils.py                 # 공통 유틸리티
├── data/processed/
├── results/analysis/
└── tests/
```

## Quick Start

```bash
# 환경 설정
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # OPENAI_API_KEY 입력

# 데이터 준비 (B-1 + B-2 모두 생성)
python -m src.prepare_data --config experiment_b.yaml --smoke

# B1-1: 3-Prompt 비교
python -m src.run_judge_batch auto --experiment b1-1 --config experiment_b.yaml --smoke
python -m src.analyze --experiment b1-1 --config experiment_b.yaml --smoke

# B2-1: Same-Type Swap (PKI 검증)
python -m src.run_judge_batch auto --experiment b2-1 --config experiment_b.yaml --smoke
python -m src.analyze --experiment b2-1 --config experiment_b.yaml --smoke

# B2-2: Cross-Type Swap (통제 조건)
python -m src.run_judge_batch auto --experiment b2-2 --config experiment_b.yaml --smoke
python -m src.analyze --experiment b2-2 --config experiment_b.yaml --smoke

# B-1 vs B-2 Paired Comparison
python -m src.analyze --experiment b2-compare --config experiment_b.yaml --smoke
```

## 3-Prompt Design

| Prompt | Source | K | Type | SE Norm |
|--------|--------|---|------|---------|
| P-Thakur | Thakur et al. (2024) | 2 | Zero-shot | H/ln(2) |
| P-CLEV | Choudhary et al. (2024) | 3 | Zero-shot role-playing | H/ln(3) |
| P-Lee-Standard | Lee et al. (2026) | 3 | Few-shot 3-way | H/ln(3) |
