# LLM-Judge Semantic Uncertainty 관측 연구

## Overview
LLM-as-a-Judge의 판단 불안정성(Semantic Uncertainty)을 정량적으로 관측하는 실험.
TriviaQA 기반 QA 태스크에서 GPT-4o-mini Judge의 반복 판단 분산을 Semantic Entropy로 측정한다.

## Research Questions
- **RQ1**: 판단 불안정성은 결정론적 환경(T=0, prompt 고정)에서도 발생하는가?
- **RQ2**: 어떤 변화가 semantic entropy에 가장 큰 영향을 끼치는가? *(Phase B)*
- **RQ3**: 높은 semantic entropy는 높은 judgement 오류율과 관련 있는가?

## Project Structure
```
llm-judge-uncertainty/
├── configs/
│   └── experiment.yaml        # 실험 하이퍼파라미터 및 설정
├── src/
│   ├── __init__.py
│   ├── prepare_data.py        # Step 1: TriviaQA 서브셋 추출 + 오답 생성
│   ├── run_judge.py           # Step 2: Judge 반복 샘플링 (API 호출)
│   ├── analyze.py             # Step 3: Semantic Entropy 계산 + 분석
│   └── utils.py               # 공통 유틸리티
├── notebooks/
│   └── 03_analysis.ipynb      # 인터랙티브 분석 및 시각화
├── data/
│   ├── raw/                   # TriviaQA 원본 데이터
│   └── processed/             # 전처리된 서브셋 + 오답 포함
├── results/
│   ├── logs/                  # API 호출 로그 (JSONL)
│   └── analysis/              # 분석 결과 (CSV, 그래프)
├── tests/
│   └── test_smoke.py          # 소규모 스모크 테스트
├── .env.example               # 환경변수 템플릿
├── .gitignore
├── pyproject.toml             # 프로젝트 메타데이터 + 의존성
├── requirements.txt           # pip 의존성
└── README.md
```

## Setup

### 1. 환경 생성
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

### 3. 데이터 준비
```bash
python -m src.prepare_data
```

### 4. Judge 실험 실행
```bash
# 스모크 테스트 (5문항 × 3답변 × 3회 반복 = 45 API calls)
python -m src.run_judge --smoke

# 풀 실험 (200문항 × 3답변 × 30회 반복 = 18,000 API calls)
python -m src.run_judge
```

### 5. 분석
```bash
python -m src.analyze
# 또는 notebooks/03_analysis.ipynb 실행
```

## Phase A (Baseline) - 현재
- T=0, seed 고정, prompt 고정
- Simple Clustering (verdict 기반)
- 200문항 × 3답변유형 × 30회 반복

## Phase B (Stress Test) - 예정
- Temperature > 0
- Paraphrasing / 어순 변경
- Alias Mutation 오답
- 다른 Judge 모델 비교
