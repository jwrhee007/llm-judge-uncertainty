"""
Step 3: Semantic Uncertainty 분석
- Verdict 분포 집계
- Semantic Entropy 계산 (Simple Clustering, verdict 기반)
- Normalized Semantic Entropy
- Flip Rate 계산
- 시각화 (entropy vs answer_type, flip_rate 등)

Usage:
    python -m src.analyze
    python -m src.analyze --smoke
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
    from src.utils import DATA_PROCESSED

    suffix = "_smoke" if smoke else ""
    filepath = DATA_PROCESSED / f"evaluation_set{suffix}.jsonl"
    records = read_jsonl(filepath)
    meta = pd.DataFrame(records)
    # question_id + answer_category 기준으로 중복 제거
    meta = meta.drop_duplicates(subset=["question_id", "answer_category"])
    return meta[["question_id", "answer_category", "answer_type_ner",
                  "question_length", "context_length", "ground_truth", "answer"]]


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

    # 1a: Semantic Entropy boxplot
    order = ["correct", "obvious_wrong", "confusing_wrong"]
    available_order = [o for o in order if o in entropy_df["answer_category"].values]
    sns.boxplot(data=entropy_df, x="answer_category", y="semantic_entropy",
                order=available_order, ax=axes[0])
    axes[0].set_title("Semantic Entropy by Answer Category")
    axes[0].set_xlabel("Answer Category")
    axes[0].set_ylabel("Semantic Entropy (nats)")

    # 1b: Flip Rate boxplot
    sns.boxplot(data=entropy_df, x="answer_category", y="flip_rate",
                order=available_order, ax=axes[1])
    axes[1].set_title("Flip Rate by Answer Category")
    axes[1].set_xlabel("Answer Category")
    axes[1].set_ylabel("Flip Rate")

    # 1c: Normalized Entropy distribution
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

    # 2a: Entropy by majority verdict
    verdicts_present = [v for v in ["CORRECT", "INCORRECT", "UNSURE"]
                        if v in entropy_df["majority_verdict"].values]
    sns.boxplot(data=entropy_df, x="majority_verdict", y="semantic_entropy",
                order=verdicts_present, ax=axes[0])
    axes[0].set_title("Entropy by Majority Verdict (RQ3)")
    axes[0].set_xlabel("Majority Verdict")
    axes[0].set_ylabel("Semantic Entropy (nats)")

    # 2b: Flip Rate vs Entropy scatter
    sns.scatterplot(data=entropy_df, x="semantic_entropy", y="flip_rate",
                    hue="answer_category", hue_order=available_order, alpha=0.6, ax=axes[1])
    axes[1].set_title("Flip Rate vs Semantic Entropy")
    axes[1].set_xlabel("Semantic Entropy (nats)")
    axes[1].set_ylabel("Flip Rate")

    plt.tight_layout()
    fig.savefig(save_dir / f"fig2_entropy_vs_correctness{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved fig2_entropy_vs_correctness{suffix}.png")

    # --- Figure 3: 보조 관측 (NER type별) ---
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
    # CSV
    csv_path = RESULTS_ANALYSIS / f"entropy_results{suffix}.csv"
    entropy_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved {csv_path}")

    # 통계 결과 JSON
    stats_path = RESULTS_ANALYSIS / f"statistical_tests{suffix}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {stats_path}")


def print_summary(entropy_df: pd.DataFrame, stat_results: dict):
    """콘솔에 요약 출력."""
    logger.info("=" * 60)
    logger.info("Analysis Summary")
    logger.info("=" * 60)

    # 전체 통계
    logger.info(f"  Total evaluation sets: {len(entropy_df)}")
    logger.info(f"  Mean Semantic Entropy: {entropy_df['semantic_entropy'].mean():.4f}")
    logger.info(f"  Mean Flip Rate: {entropy_df['flip_rate'].mean():.4f}")

    # 불안정한 세트 비율
    unstable = entropy_df[entropy_df["flip_rate"] > 0]
    logger.info(f"  Unstable sets (flip_rate > 0): {len(unstable)} / {len(entropy_df)} "
                f"({len(unstable) / len(entropy_df) * 100:.1f}%)")

    # 카테고리별 요약
    logger.info("-" * 60)
    logger.info("  Per-category summary:")
    for cat in ["correct", "obvious_wrong", "confusing_wrong"]:
        subset = entropy_df[entropy_df["answer_category"] == cat]
        if len(subset) == 0:
            continue
        logger.info(f"    {cat:20s}: entropy={subset['semantic_entropy'].mean():.4f}, "
                    f"flip_rate={subset['flip_rate'].mean():.4f}, "
                    f"unstable={sum(subset['flip_rate'] > 0)}/{len(subset)}")

    # 통계 검정 결과
    if stat_results:
        logger.info("-" * 60)
        logger.info("  Statistical tests:")
        for name, res in stat_results.items():
            logger.info(f"    {name}: {res['test']}, p={res['p_value']:.4f}")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 3: Analyze Judge results")
    parser.add_argument("--smoke", action="store_true", help="Analyze smoke test results")
    parser.add_argument("--config", default="experiment.yaml", help="Config file name")
    args = parser.parse_args()

    config = load_config(args.config)
    suffix = "_smoke" if args.smoke else ""

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

    # 6. 시각화
    logger.info("Creating visualizations...")
    create_visualizations(entropy_df, suffix=suffix)

    # 7. 저장
    save_results(entropy_df, stat_results, suffix=suffix)

    # 8. 요약
    print_summary(entropy_df, stat_results)


if __name__ == "__main__":
    main()