# LLM-Judge Semantic Uncertainty 관측 연구

> **LLM-as-a-Judge의 판단 불안정성(Semantic Uncertainty)을 정량적으로 관측하는 실험 프레임워크**

## Research Questions

| ID | 연구 질문 | Phase |
|----|----------|-------|
| RQ1 | 결정론적 환경(T=0, seed 고정)에서도 판단 불안정성이 나타나는가? | **A (완료)** |
| RQ2 | 어떤 프롬프트 perturbation이 가장 큰 semantic entropy 증가를 유발하는가? | B (예정) |
| RQ3 | 높은 semantic entropy는 높은 판단 오류율과 연관되는가? | **A (완료)** |

## Phase A 주요 결과

TriviaQA 200문항 × 3답변유형 × 30회 반복 = **18,000회** GPT-4o-mini Judge 호출.

| 발견 | 수치 |
|------|------|
| T=0에서도 불안정한 세트 | **35 / 600 (5.8%)** |
| 정답(correct) 카테고리 불안정 비율 | **31 / 200 (15.5%)** |
| 오답 카테고리 불안정 비율 | 1\~3 / 200 (0.5\~1.5%) |
| Entropy 차이 (correct vs wrong) | **15~20배** (p < 10⁻¹¹) |

**핵심 발견**: Judge는 오답을 "틀렸다"고 판단하는 데는 안정적이나, 정답을 "맞다"고 확인하는 데서 판단이 흔들린다 (비대칭적 불안정성).

→ 상세 분석: [`PhaseA_Baseline_Report.md`](PhaseA_Baseline_Report.md)

## Project Structure

```
llm-judge-uncertainty/
├── configs/
│   └── experiment.yaml            # 실험 하이퍼파라미터 및 설정
├── src/
│   ├── prepare_data.py            # Step 1: TriviaQA 서브셋 + NER 분류 + 오답 생성
│   ├── run_judge.py               # Step 2a: 실시간 async API 호출
│   ├── run_judge_batch.py         # Step 2b: OpenAI Batch API (권장)
│   ├── analyze.py                 # Step 3: Semantic Entropy + 통계 검정 + 시각화
│   └── utils.py                   # 공통 유틸리티
├── data/processed/                # 전처리된 데이터셋
│   ├── questions.jsonl            # 200문항 메타데이터
│   └── evaluation_set.jsonl       # 600 평가 세트 (200 × 3)
├── results/analysis/              # Phase A 분석 결과
│   ├── entropy_results.csv        # 600 세트 × entropy/flip rate
│   ├── statistical_tests.json     # 통계 검정 결과
│   └── fig*.png                   # 시각화
├── tests/test_smoke.py
├── PhaseA_Baseline_Report.md      # Phase A 실험 보고서
├── requirements.txt
└── setup.sh                       # 원클릭 환경 설정
```

## Setup & Reproduction

```bash
# 1. 환경 설정
bash setup.sh
source .venv/bin/activate

# 2. API 키 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 3. 데이터 준비 (200문항, 600 eval sets)
python -m src.prepare_data

# 4. Judge 반복 샘플링 (Batch API 권장)
python -m src.run_judge_batch auto          # 18,000 API calls
# 또는 스모크 테스트: python -m src.run_judge_batch auto --smoke

# 5. 분석 + 시각화
python -m src.analyze
```

## Measurement

- **Semantic Entropy**: H = −Σ pᵢ ln(pᵢ) — verdict 분포(CORRECT/INCORRECT/UNSURE) 기반
- **Normalized SE**: H / ln(K) — 활성 클러스터 수로 정규화 (0~1)
- **Flip Rate**: (N − N_majority) / N — 다수결과 다른 판정 비율

## Phase B (예정)

- Paraphrase perturbation (동의어 치환)
- Context order shuffling
- Instruction rephrasing
- Few-shot example injection

## Environment

| 항목 | 사양 |
|------|------|
| Python | ≥ 3.10 |
| Judge Model | GPT-4o-mini (T=0, seed=42) |
| API | OpenAI Batch API |
| 주요 패키지 | openai, datasets, spacy, pandas, scipy, matplotlib |