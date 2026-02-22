"""
Step 3: Semantic Uncertainty 분석 + 심층 분석
- Verdict 분포 집계
- Semantic Entropy 계산 (Simple Clustering, verdict 기반)
- Normalized Semantic Entropy / Flip Rate
- 통계 검정 (Kruskal-Wallis, Mann-Whitney U)
- 시각화 (entropy vs category, correctness, NER type)
- 심층 분류 (stable / unstable / mismatch / both)
- Unstable(SE>0) + Mismatch(expected≠majority) 상세 추출
- 대표 사례 콘솔 출력

Usage:
    python -m src.analyze
    python -m src.analyze --smoke
    python -m src.analyze --skip-deep    # 심층 분석 건너뛰기
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DATA_PROCESSED,
    RESULTS_ANALYSIS,
    RESULTS_LOGS,
    load_config,
    read_jsonl,
    setup_logger,
)

logger = setup_logger("analyze")

# 스타일 설정
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================
# 1. 데이터 로드 및 집계
# =============================================================
def load_results(smoke: bool = False) -> pd.DataFrame:
    """Judge 결과 JSONL을 DataFrame으로 로드."""
    suffix = "_smoke" if smoke else ""
    log_path = RESULTS_LOGS / f"judge_results{suffix}.jsonl"

    if not log_path.exists():
        logger.error(f"File not found: {log_path}")
        logger.error("Run 'python -m src.run_judge' first.")
        sys.exit(1)

    records = read_jsonl(log_path)
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records from {log_path}")

    # 기본 필터: API_ERROR, PARSE_ERROR 제외
    n_errors = df[df["verdict"].isin(["API_ERROR", "PARSE_ERROR"])].shape[0]
    if n_errors > 0:
        logger.warning(f"  Excluding {n_errors} error records")
        df = df[~df["verdict"].isin(["API_ERROR", "PARSE_ERROR"])].copy()

    return df


def load_evaluation_metadata(smoke: bool = False) -> pd.DataFrame:
    """evaluation_set.jsonl에서 메타데이터(answer_type_ner, lengths 등) 로드."""
    suffix = "_smoke" if smoke else ""
    filepath = DATA_PROCESSED / f"evaluation_set{suffix}.jsonl"
    records = read_jsonl(filepath)
    meta = pd.DataFrame(records)
    # question_id + answer_category 기준으로 중복 제거
    meta = meta.drop_duplicates(subset=["question_id", "answer_category"])
    return meta[["question_id", "answer_category", "answer_type_ner",
                  "question_length", "context_length", "ground_truth", "answer"]]


def load_evaluation_full(smoke: bool = False) -> pd.DataFrame:
    """evaluation_set.jsonl 전체 로드 (심층 분석용: question, context 포함)."""
    suffix = "_smoke" if smoke else ""
    filepath = DATA_PROCESSED / f"evaluation_set{suffix}.jsonl"
    records = read_jsonl(filepath)
    return pd.DataFrame(records)


# =============================================================
# 2. Verdict 분포 집계
# =============================================================
def compute_verdict_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 (question_id × answer_category) 세트의 verdict 분포 계산.
    Returns: DataFrame with p_correct, p_incorrect, p_unsure per set.
    """
    grouped = df.groupby(["question_id", "answer_category"])

    rows = []
    for (qid, acat), group in grouped:
        n = len(group)
        counts = Counter(group["verdict"])

        rows.append({
            "question_id": qid,
            "answer_category": acat,
            "n_trials": n,
            "n_correct": counts.get("CORRECT", 0),
            "n_incorrect": counts.get("INCORRECT", 0),
            "n_unsure": counts.get("UNSURE", 0),
            "p_correct": counts.get("CORRECT", 0) / n,
            "p_incorrect": counts.get("INCORRECT", 0) / n,
            "p_unsure": counts.get("UNSURE", 0) / n,
            "majority_verdict": counts.most_common(1)[0][0],
        })

    result = pd.DataFrame(rows)
    logger.info(f"Computed verdict distribution for {len(result)} evaluation sets")
    return result


# =============================================================
# 3. Semantic Entropy 계산
# =============================================================
def compute_semantic_entropy(verdict_df: pd.DataFrame, base: str = "natural") -> pd.DataFrame:
    """
    Simple Clustering (verdict 기반) Semantic Entropy 계산.
    H_semantic = -Σ p_i × log(p_i)   (p_i > 0인 클러스터만)

    Also computes:
    - H_norm = H / log(K)  (K = 활성 클러스터 수)
    - flip_rate = (다수결과 다른 판정 수) / N
    """
    log_fn = np.log if base == "natural" else np.log2

    rows = []
    for _, row in verdict_df.iterrows():
        probs = [row["p_correct"], row["p_incorrect"], row["p_unsure"]]
        active_probs = [p for p in probs if p > 0]
        k = len(active_probs)

        # Semantic Entropy
        h = -sum(p * log_fn(p) for p in active_probs)

        # Normalized Semantic Entropy
        h_norm = h / log_fn(k) if k > 1 else 0.0

        # Flip Rate
        n = row["n_trials"]
        majority_count = max(row["n_correct"], row["n_incorrect"], row["n_unsure"])
        flip_rate = (n - majority_count) / n

        rows.append({
            "question_id": row["question_id"],
            "answer_category": row["answer_category"],
            "semantic_entropy": h,
            "normalized_entropy": h_norm,
            "active_clusters": k,
            "flip_rate": flip_rate,
            "majority_verdict": row["majority_verdict"],
            "n_trials": n,
            "p_correct": row["p_correct"],
            "p_incorrect": row["p_incorrect"],
            "p_unsure": row["p_unsure"],
        })

    result = pd.DataFrame(rows)
    logger.info(f"Semantic Entropy computed: mean={result['semantic_entropy'].mean():.4f}, "
                f"flip_rate mean={result['flip_rate'].mean():.4f}")
    return result


# =============================================================
# 4. 통계 분석
# =============================================================
def run_statistical_analysis(entropy_df: pd.DataFrame) -> dict:
    """핵심 관측 항목에 대한 통계 분석."""
    results = {}

    # --- RQ1: entropy vs answer_category ---
    categories = entropy_df["answer_category"].unique()
    groups = {cat: entropy_df[entropy_df["answer_category"] == cat]["semantic_entropy"]
              for cat in categories}

    if len(groups) >= 2:
        group_values = [g.values for g in groups.values()]
        if all(len(g) > 1 for g in group_values):
            stat, p_val = stats.kruskal(*group_values)
            results["entropy_vs_answer_category"] = {
                "test": "Kruskal-Wallis",
                "statistic": float(stat),
                "p_value": float(p_val),
                "group_means": {cat: float(g.mean()) for cat, g in groups.items()},
                "group_medians": {cat: float(g.median()) for cat, g in groups.items()},
            }
            logger.info(f"  RQ1 (entropy vs answer_category): H={stat:.2f}, p={p_val:.4f}")

    # --- RQ1: flip_rate vs answer_category ---
    flip_groups = {cat: entropy_df[entropy_df["answer_category"] == cat]["flip_rate"]
                   for cat in categories}
    if len(flip_groups) >= 2:
        flip_values = [g.values for g in flip_groups.values()]
        if all(len(g) > 1 for g in flip_values):
            stat, p_val = stats.kruskal(*flip_values)
            results["flip_rate_vs_answer_category"] = {
                "test": "Kruskal-Wallis",
                "statistic": float(stat),
                "p_value": float(p_val),
                "group_means": {cat: float(g.mean()) for cat, g in flip_groups.items()},
            }
            logger.info(f"  RQ1 (flip_rate vs answer_category): H={stat:.2f}, p={p_val:.4f}")

    # --- RQ3: entropy vs correctness (majority_verdict 기준) ---
    correct_entropy = entropy_df[entropy_df["majority_verdict"] == "CORRECT"]["semantic_entropy"]
    incorrect_entropy = entropy_df[entropy_df["majority_verdict"] == "INCORRECT"]["semantic_entropy"]

    if len(correct_entropy) > 1 and len(incorrect_entropy) > 1:
        stat, p_val = stats.mannwhitneyu(correct_entropy, incorrect_entropy, alternative="two-sided")
        results["entropy_vs_correctness"] = {
            "test": "Mann-Whitney U",
            "statistic": float(stat),
            "p_value": float(p_val),
            "correct_mean": float(correct_entropy.mean()),
            "incorrect_mean": float(incorrect_entropy.mean()),
        }
        logger.info(f"  RQ3 (entropy: correct vs incorrect): U={stat:.2f}, p={p_val:.4f}")

    return results


# =============================================================
# 5. 시각화
# =============================================================
def create_visualizations(
    entropy_df: pd.DataFrame, suffix: str = "", save_dir: Path = RESULTS_ANALYSIS
):
    """핵심 시각화 생성."""

    # --- Figure 1: Entropy by Answer Category ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    order = ["correct", "obvious_wrong", "confusing_wrong"]
    available_order = [o for o in order if o in entropy_df["answer_category"].values]
    sns.boxplot(data=entropy_df, x="answer_category", y="semantic_entropy",
                order=available_order, ax=axes[0])
    axes[0].set_title("Semantic Entropy by Answer Category")
    axes[0].set_xlabel("Answer Category")
    axes[0].set_ylabel("Semantic Entropy (nats)")

    sns.boxplot(data=entropy_df, x="answer_category", y="flip_rate",
                order=available_order, ax=axes[1])
    axes[1].set_title("Flip Rate by Answer Category")
    axes[1].set_xlabel("Answer Category")
    axes[1].set_ylabel("Flip Rate")

    sns.histplot(data=entropy_df, x="normalized_entropy", hue="answer_category",
                 hue_order=available_order, bins=20, ax=axes[2], stat="density", common_norm=False)
    axes[2].set_title("Normalized Entropy Distribution")
    axes[2].set_xlabel("Normalized Entropy")

    plt.tight_layout()
    fig.savefig(save_dir / f"fig1_entropy_by_category{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved fig1_entropy_by_category{suffix}.png")

    # --- Figure 2: Entropy vs Correctness ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    verdicts_present = [v for v in ["CORRECT", "INCORRECT", "UNSURE"]
                        if v in entropy_df["majority_verdict"].values]
    sns.boxplot(data=entropy_df, x="majority_verdict", y="semantic_entropy",
                order=verdicts_present, ax=axes[0])
    axes[0].set_title("Entropy by Majority Verdict (RQ3)")
    axes[0].set_xlabel("Majority Verdict")
    axes[0].set_ylabel("Semantic Entropy (nats)")

    sns.scatterplot(data=entropy_df, x="semantic_entropy", y="flip_rate",
                    hue="answer_category", hue_order=available_order, alpha=0.6, ax=axes[1])
    axes[1].set_title("Flip Rate vs Semantic Entropy")
    axes[1].set_xlabel("Semantic Entropy (nats)")
    axes[1].set_ylabel("Flip Rate")

    plt.tight_layout()
    fig.savefig(save_dir / f"fig2_entropy_vs_correctness{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved fig2_entropy_vs_correctness{suffix}.png")

    # --- Figure 3: NER type별 ---
    if "answer_type_ner" in entropy_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        type_order = (entropy_df.groupby("answer_type_ner")["semantic_entropy"]
                      .mean().sort_values(ascending=False).index.tolist())
        sns.boxplot(data=entropy_df, x="answer_type_ner", y="semantic_entropy",
                    order=type_order, ax=ax)
        ax.set_title("Semantic Entropy by Answer NER Type")
        ax.set_xlabel("Answer Type (NER)")
        ax.set_ylabel("Semantic Entropy (nats)")
        plt.tight_layout()
        fig.savefig(save_dir / f"fig3_entropy_by_ner_type{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved fig3_entropy_by_ner_type{suffix}.png")


# =============================================================
# 6. 결과 저장 및 요약
# =============================================================
def save_results(
    entropy_df: pd.DataFrame,
    stat_results: dict,
    suffix: str = "",
):
    """분석 결과를 CSV + JSON으로 저장."""
    csv_path = RESULTS_ANALYSIS / f"entropy_results{suffix}.csv"
    entropy_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path}")

    stats_path = RESULTS_ANALYSIS / f"statistical_tests{suffix}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {stats_path}")


def print_summary(entropy_df: pd.DataFrame, stat_results: dict):
    """콘솔에 요약 출력."""
    logger.info("=" * 60)
    logger.info("Analysis Summary")
    logger.info("=" * 60)

    logger.info(f"  Total evaluation sets: {len(entropy_df)}")
    logger.info(f"  Mean Semantic Entropy: {entropy_df['semantic_entropy'].mean():.4f}")
    logger.info(f"  Mean Flip Rate: {entropy_df['flip_rate'].mean():.4f}")

    unstable = entropy_df[entropy_df["flip_rate"] > 0]
    logger.info(f"  Unstable sets (flip_rate > 0): {len(unstable)} / {len(entropy_df)} "
                f"({len(unstable) / len(entropy_df) * 100:.1f}%)")

    logger.info("-" * 60)
    logger.info("  Per-category summary:")
    for cat in ["correct", "obvious_wrong", "confusing_wrong"]:
        subset = entropy_df[entropy_df["answer_category"] == cat]
        if len(subset) == 0:
            continue
        logger.info(f"    {cat:20s}: entropy={subset['semantic_entropy'].mean():.4f}, "
                    f"flip_rate={subset['flip_rate'].mean():.4f}, "
                    f"unstable={sum(subset['flip_rate'] > 0)}/{len(subset)}")

    if stat_results:
        logger.info("-" * 60)
        logger.info("  Statistical tests:")
        for name, res in stat_results.items():
            logger.info(f"    {name}: {res['test']}, p={res['p_value']:.4f}")


# =============================================================
# 7. 심층 분류 (Deep Analysis)
# =============================================================
def classify_questions(entropy_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 eval set를 4분류:
    - stable_correct: 안정 + 정확 판정
    - unstable_only:  SE>0 흔들렸지만 다수결은 기대와 일치
    - mismatch_only:  SE=0 안정적이지만 기대와 불일치 (체계적 오판)
    - both:           SE>0 흔들리면서 기대와 불일치
    """
    df = entropy_df.copy()

    df["expected_verdict"] = df["answer_category"].apply(
        lambda cat: "CORRECT" if cat == "correct" else "INCORRECT"
    )
    df["is_unstable"] = df["semantic_entropy"] > 0
    df["is_mismatch"] = df["majority_verdict"] != df["expected_verdict"]

    conditions = []
    for _, row in df.iterrows():
        if row["is_unstable"] and row["is_mismatch"]:
            conditions.append("both")
        elif row["is_unstable"]:
            conditions.append("unstable_only")
        elif row["is_mismatch"]:
            conditions.append("mismatch_only")
        else:
            conditions.append("stable_correct")
    df["issue_type"] = conditions

    return df


def summarize_classification(classified_df: pd.DataFrame):
    """분류 결과 요약 출력."""
    logger.info("=" * 60)
    logger.info("Deep Analysis: 문항 분류 요약")
    logger.info("=" * 60)

    for t in ["stable_correct", "unstable_only", "mismatch_only", "both"]:
        n = sum(classified_df["issue_type"] == t)
        logger.info(f"  {t:20s}: {n:4d} ({n/len(classified_df)*100:.1f}%)")

    logger.info("")
    for cat in ["correct", "obvious_wrong", "confusing_wrong"]:
        subset = classified_df[classified_df["answer_category"] == cat]
        logger.info(f"  [{cat}] (n={len(subset)})")
        for t in ["stable_correct", "unstable_only", "mismatch_only", "both"]:
            n = sum(subset["issue_type"] == t)
            if n > 0:
                logger.info(f"    {t:20s}: {n:4d} ({n/len(subset)*100:.1f}%)")


# =============================================================
# 8. Unstable / Mismatch 상세 추출
# =============================================================
def extract_unstable_details(
    classified_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    judge_df: pd.DataFrame,
) -> pd.DataFrame:
    """SE > 0 문항의 상세 정보 추출 (question, answer, rationale 포함)."""
    unstable = classified_df[classified_df["is_unstable"]].copy()
    unstable = unstable.sort_values("semantic_entropy", ascending=False)

    rows = []
    for _, row in unstable.iterrows():
        qid = row["question_id"]
        acat = row["answer_category"]

        eval_row = eval_df[
            (eval_df["question_id"] == qid) & (eval_df["answer_category"] == acat)
        ]
        if eval_row.empty:
            continue
        eval_row = eval_row.iloc[0]

        trials = judge_df[
            (judge_df["question_id"] == qid) & (judge_df["answer_category"] == acat)
        ]
        verdict_counts = Counter(trials["verdict"])

        sample_rationales = {}
        for verdict in ["CORRECT", "INCORRECT", "UNSURE"]:
            v_trials = trials[trials["verdict"] == verdict]
            if len(v_trials) > 0:
                sample_rationales[verdict] = v_trials.iloc[0].get("rationale", "")

        rows.append({
            "question_id": qid,
            "answer_category": acat,
            "issue_type": row["issue_type"],
            "semantic_entropy": row["semantic_entropy"],
            "flip_rate": row["flip_rate"],
            "majority_verdict": row["majority_verdict"],
            "expected_verdict": row["expected_verdict"],
            "answer_type_ner": row.get("answer_type_ner", ""),
            "n_correct": verdict_counts.get("CORRECT", 0),
            "n_incorrect": verdict_counts.get("INCORRECT", 0),
            "n_unsure": verdict_counts.get("UNSURE", 0),
            "question": eval_row.get("question", ""),
            "answer": eval_row.get("answer", ""),
            "ground_truth": eval_row.get("ground_truth", ""),
            "context_preview": str(eval_row.get("context", ""))[:300],
            "rationale_CORRECT": sample_rationales.get("CORRECT", ""),
            "rationale_INCORRECT": sample_rationales.get("INCORRECT", ""),
            "rationale_UNSURE": sample_rationales.get("UNSURE", ""),
        })

    result = pd.DataFrame(rows)
    logger.info(f"  Unstable 문항 (SE > 0): {len(result)}건")
    return result


def extract_mismatch_details(
    classified_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    judge_df: pd.DataFrame,
) -> pd.DataFrame:
    """Category-Verdict 불일치 문항의 상세 정보 추출."""
    mismatch = classified_df[classified_df["is_mismatch"]].copy()
    mismatch = mismatch.sort_values("semantic_entropy", ascending=False)

    rows = []
    for _, row in mismatch.iterrows():
        qid = row["question_id"]
        acat = row["answer_category"]

        eval_row = eval_df[
            (eval_df["question_id"] == qid) & (eval_df["answer_category"] == acat)
        ]
        if eval_row.empty:
            continue
        eval_row = eval_row.iloc[0]

        trials = judge_df[
            (judge_df["question_id"] == qid) & (judge_df["answer_category"] == acat)
        ]
        verdict_counts = Counter(trials["verdict"])

        majority_v = row["majority_verdict"]
        majority_trials = trials[trials["verdict"] == majority_v]
        sample_rationale = majority_trials.iloc[0].get("rationale", "") if len(majority_trials) > 0 else ""

        rows.append({
            "question_id": qid,
            "answer_category": acat,
            "issue_type": row["issue_type"],
            "semantic_entropy": row["semantic_entropy"],
            "flip_rate": row["flip_rate"],
            "expected_verdict": row["expected_verdict"],
            "majority_verdict": row["majority_verdict"],
            "answer_type_ner": row.get("answer_type_ner", ""),
            "n_correct": verdict_counts.get("CORRECT", 0),
            "n_incorrect": verdict_counts.get("INCORRECT", 0),
            "n_unsure": verdict_counts.get("UNSURE", 0),
            "question": eval_row.get("question", ""),
            "answer": eval_row.get("answer", ""),
            "ground_truth": eval_row.get("ground_truth", ""),
            "context_preview": str(eval_row.get("context", ""))[:300],
            "majority_rationale": sample_rationale,
        })

    result = pd.DataFrame(rows)
    logger.info(f"  Mismatch 문항 (expected ≠ majority): {len(result)}건")
    return result


# =============================================================
# 9. 대표 사례 출력 + 심층 분석 결과 저장
# =============================================================
def print_representative_cases(unstable_df: pd.DataFrame, mismatch_df: pd.DataFrame):
    """대표 사례 콘솔 출력."""

    # --- Unstable Top 5 ---
    logger.info("\n" + "=" * 60)
    logger.info("대표 사례: Unstable Top 5 (SE 높은 순)")
    logger.info("=" * 60)
    for i, (_, row) in enumerate(unstable_df.head(5).iterrows()):
        logger.info(f"\n--- [{i+1}] {row['question_id']} / {row['answer_category']} ---")
        logger.info(f"  Question: {row['question']}")
        logger.info(f"  Answer: {row['answer']} (GT: {row['ground_truth']})")
        logger.info(f"  NER Type: {row['answer_type_ner']}")
        logger.info(f"  SE: {row['semantic_entropy']:.4f}, Flip Rate: {row['flip_rate']:.2f}")
        logger.info(f"  Verdicts: C={row['n_correct']}, I={row['n_incorrect']}, U={row['n_unsure']}")
        logger.info(f"  Majority: {row['majority_verdict']} (Expected: {row['expected_verdict']})")
        if row.get("rationale_CORRECT"):
            logger.info(f"  [CORRECT 근거]: {row['rationale_CORRECT'][:150]}")
        if row.get("rationale_INCORRECT"):
            logger.info(f"  [INCORRECT 근거]: {row['rationale_INCORRECT'][:150]}")
        if row.get("rationale_UNSURE"):
            logger.info(f"  [UNSURE 근거]: {row['rationale_UNSURE'][:150]}")

    # --- Mismatch: correct → INCORRECT (SE=0, 체계적 오판) ---
    systematic = mismatch_df[
        (mismatch_df["answer_category"] == "correct")
        & (mismatch_df["majority_verdict"] == "INCORRECT")
        & (mismatch_df["semantic_entropy"] == 0)
    ]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"대표 사례: 체계적 오판 (correct→INCORRECT, SE=0) — {len(systematic)}건")
    logger.info("=" * 60)
    for i, (_, row) in enumerate(systematic.head(5).iterrows()):
        logger.info(f"\n--- [{i+1}] {row['question_id']} ---")
        logger.info(f"  Question: {row['question']}")
        logger.info(f"  Answer (정답): {row['answer']}")
        logger.info(f"  NER Type: {row['answer_type_ner']}")
        logger.info(f"  30회 모두 INCORRECT 판정")
        logger.info(f"  [Judge 근거]: {row['majority_rationale'][:200]}")
        logger.info(f"  Context: {row['context_preview'][:200]}...")

    # --- Mismatch: correct → UNSURE ---
    unsure_cases = mismatch_df[
        (mismatch_df["answer_category"] == "correct")
        & (mismatch_df["majority_verdict"] == "UNSURE")
    ]
    if len(unsure_cases) > 0:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"대표 사례: correct→UNSURE — {len(unsure_cases)}건")
        logger.info("=" * 60)
        for i, (_, row) in enumerate(unsure_cases.head(3).iterrows()):
            logger.info(f"\n--- [{i+1}] {row['question_id']} ---")
            logger.info(f"  Question: {row['question']}")
            logger.info(f"  Answer (정답): {row['answer']}")
            logger.info(f"  SE: {row['semantic_entropy']:.4f}")
            logger.info(f"  Verdicts: C={row['n_correct']}, I={row['n_incorrect']}, U={row['n_unsure']}")
            logger.info(f"  [Judge 근거]: {row['majority_rationale'][:200]}")


def save_deep_results(
    classified_df: pd.DataFrame,
    unstable_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    suffix: str = "",
):
    """심층 분석 결과를 CSV + JSON으로 저장."""
    classified_path = RESULTS_ANALYSIS / f"classified_questions{suffix}.csv"
    classified_df.to_csv(classified_path, index=False)
    logger.info(f"  Saved {classified_path}")

    unstable_path = RESULTS_ANALYSIS / f"unstable_details{suffix}.csv"
    unstable_df.to_csv(unstable_path, index=False)
    logger.info(f"  Saved {unstable_path}")

    mismatch_path = RESULTS_ANALYSIS / f"mismatch_details{suffix}.csv"
    mismatch_df.to_csv(mismatch_path, index=False)
    logger.info(f"  Saved {mismatch_path}")

    # JSON 요약
    summary = {
        "total_eval_sets": len(classified_df),
        "issue_type_distribution": classified_df["issue_type"].value_counts().to_dict(),
        "unstable_count": int(classified_df["is_unstable"].sum()),
        "mismatch_count": int(classified_df["is_mismatch"].sum()),
        "mismatch_subtypes": {
            "correct→INCORRECT": {
                "total": int(((classified_df["answer_category"] == "correct")
                              & (classified_df["majority_verdict"] == "INCORRECT")).sum()),
                "SE=0": int(((classified_df["answer_category"] == "correct")
                             & (classified_df["majority_verdict"] == "INCORRECT")
                             & (classified_df["semantic_entropy"] == 0)).sum()),
                "SE>0": int(((classified_df["answer_category"] == "correct")
                             & (classified_df["majority_verdict"] == "INCORRECT")
                             & (classified_df["semantic_entropy"] > 0)).sum()),
            },
            "correct→UNSURE": {
                "total": int(((classified_df["answer_category"] == "correct")
                              & (classified_df["majority_verdict"] == "UNSURE")).sum()),
                "SE=0": int(((classified_df["answer_category"] == "correct")
                             & (classified_df["majority_verdict"] == "UNSURE")
                             & (classified_df["semantic_entropy"] == 0)).sum()),
                "SE>0": int(((classified_df["answer_category"] == "correct")
                             & (classified_df["majority_verdict"] == "UNSURE")
                             & (classified_df["semantic_entropy"] > 0)).sum()),
            },
        },
        "per_category": {},
    }
    for cat in ["correct", "obvious_wrong", "confusing_wrong"]:
        sub = classified_df[classified_df["answer_category"] == cat]
        summary["per_category"][cat] = {
            "total": len(sub),
            "stable_correct": int(((~sub["is_unstable"]) & (~sub["is_mismatch"])).sum()),
            "unstable_only": int((sub["is_unstable"] & (~sub["is_mismatch"])).sum()),
            "mismatch_only": int(((~sub["is_unstable"]) & sub["is_mismatch"]).sum()),
            "both": int((sub["is_unstable"] & sub["is_mismatch"]).sum()),
        }

    summary_path = RESULTS_ANALYSIS / f"deep_analysis_summary{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {summary_path}")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 3: Analyze Judge results")
    parser.add_argument("--smoke", action="store_true", help="Analyze smoke test results")
    parser.add_argument("--skip-deep", action="store_true", help="Skip deep analysis (Steps 7-9)")
    parser.add_argument("--config", default="experiment.yaml", help="Config file name")
    args = parser.parse_args()

    config = load_config(args.config)
    suffix = "_smoke" if args.smoke else ""

    # ── Steps 1-6: Entropy 분석 ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Steps 1-6: Entropy Analysis")
    logger.info("=" * 60)

    # 1. 결과 로드
    df = load_results(smoke=args.smoke)

    # 2. 메타데이터 병합
    meta = load_evaluation_metadata(smoke=args.smoke)
    df = df.merge(meta[["question_id", "answer_category", "answer_type_ner",
                         "question_length", "context_length"]],
                  on=["question_id", "answer_category"], how="left")

    # 3. Verdict 분포
    verdict_df = compute_verdict_distribution(df)

    # 4. Semantic Entropy + Flip Rate
    entropy_base = config["analysis"]["entropy_base"]
    entropy_df = compute_semantic_entropy(verdict_df, base=entropy_base)

    # 메타데이터 병합
    entropy_df = entropy_df.merge(
        meta[["question_id", "answer_category", "answer_type_ner",
              "question_length", "context_length"]],
        on=["question_id", "answer_category"], how="left"
    )

    # 5. 통계 분석
    stat_results = run_statistical_analysis(entropy_df)

    # 6. 시각화 + 저장 + 요약
    logger.info("Creating visualizations...")
    create_visualizations(entropy_df, suffix=suffix)
    save_results(entropy_df, stat_results, suffix=suffix)
    print_summary(entropy_df, stat_results)

    # ── Steps 7-9: 심층 분석 ─────────────────────────────────
    if not args.skip_deep:
        logger.info("\n" + "=" * 60)
        logger.info("Steps 7-9: Deep Analysis")
        logger.info("=" * 60)

        # 7. 분류
        classified_df = classify_questions(entropy_df)
        summarize_classification(classified_df)

        # 8. 상세 추출 (evaluation_set 전체 + judge_results 필요)
        eval_full_df = load_evaluation_full(smoke=args.smoke)
        judge_df = pd.DataFrame(read_jsonl(
            RESULTS_LOGS / f"judge_results{suffix}.jsonl"
        ))
        judge_df = judge_df[~judge_df["verdict"].isin(["API_ERROR", "PARSE_ERROR"])].copy()

        unstable_df = extract_unstable_details(classified_df, eval_full_df, judge_df)
        mismatch_df = extract_mismatch_details(classified_df, eval_full_df, judge_df)

        # 9. 대표 사례 + 저장
        print_representative_cases(unstable_df, mismatch_df)
        save_deep_results(classified_df, unstable_df, mismatch_df, suffix=suffix)

    logger.info("\nDone! Results saved to results/analysis/")


if __name__ == "__main__":
    main()