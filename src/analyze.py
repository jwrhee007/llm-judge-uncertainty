"""
Phase B Step 3: 3-Prompt Comparison Analysis
- 프롬프트별 Verdict 분포 + Semantic Entropy (H, H_norm)
- Cross-prompt H_norm 비교
- Negativity Bias 분석
- spaCy native NER 태그별 Content Effect
- 심층 분류 + 상세 추출

Usage:
    python -m src.analyze --config experiment_b1.yaml
    python -m src.analyze --config experiment_b1.yaml --smoke
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import DATA_PROCESSED, RESULTS_ANALYSIS, RESULTS_LOGS, load_config, read_jsonl, setup_logger
from src.prompts import POSITIVE, NEGATIVE, UNCERTAIN, PARSE_ERROR as LABEL_PARSE_ERROR

logger = setup_logger("analyze")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# === 1. 데이터 로드 ===
def load_results(smoke: bool = False) -> pd.DataFrame:
    sfx = "_smoke" if smoke else ""
    path = RESULTS_LOGS / f"judge_results_b1{sfx}.jsonl"
    if not path.exists():
        logger.error(f"Not found: {path}"); sys.exit(1)
    df = pd.DataFrame(read_jsonl(path))
    logger.info(f"Loaded {len(df)} records")
    n_err = df[df["verdict"].isin(["API_ERROR", LABEL_PARSE_ERROR])].shape[0]
    if n_err:
        logger.warning(f"  Excluding {n_err} errors")
        df = df[~df["verdict"].isin(["API_ERROR", LABEL_PARSE_ERROR])].copy()
    return df

def load_eval_meta(smoke: bool = False) -> pd.DataFrame:
    sfx = "_smoke" if smoke else ""
    path = DATA_PROCESSED / f"evaluation_set_b1{sfx}.jsonl"
    meta = pd.DataFrame(read_jsonl(path))
    meta = meta.drop_duplicates(subset=["question_id", "answer_category"])
    return meta

# === 2. Verdict 분포 (프롬프트별) ===
def compute_verdict_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """(prompt_id, question_id, answer_category)별 verdict 분포."""
    verdict_k_map = config["prompts"]["verdict_k"]
    rows = []
    for (pid, qid, acat), grp in df.groupby(["prompt_id", "question_id", "answer_category"]):
        n = len(grp)
        counts = Counter(grp["verdict"])
        k = verdict_k_map.get(pid, 3)
        rows.append({
            "prompt_id": pid, "question_id": qid, "answer_category": acat,
            "n_trials": n, "verdict_k": k,
            "n_positive": counts.get(POSITIVE, 0),
            "n_negative": counts.get(NEGATIVE, 0),
            "n_uncertain": counts.get(UNCERTAIN, 0),
            "p_positive": counts.get(POSITIVE, 0) / n,
            "p_negative": counts.get(NEGATIVE, 0) / n,
            "p_uncertain": counts.get(UNCERTAIN, 0) / n,
            "majority_verdict": counts.most_common(1)[0][0],
        })
    return pd.DataFrame(rows)

# === 3. Semantic Entropy (H + H_norm) ===
def compute_semantic_entropy(vdf: pd.DataFrame) -> pd.DataFrame:
    """H = -Σ p*ln(p), H_norm = H / ln(K)."""
    rows = []
    for _, r in vdf.iterrows():
        probs = [r["p_positive"], r["p_negative"]]
        if r["verdict_k"] >= 3:
            probs.append(r["p_uncertain"])
        active = [p for p in probs if p > 0]
        k_active = len(active)
        h = -sum(p * np.log(p) for p in active) if active else 0.0
        k = r["verdict_k"]
        h_norm = h / np.log(k) if k > 1 and h > 0 else 0.0
        majority_count = max(r["n_positive"], r["n_negative"],
                             r["n_uncertain"] if r["verdict_k"] >= 3 else 0)
        flip_rate = (r["n_trials"] - majority_count) / r["n_trials"]
        rows.append({
            **{c: r[c] for c in ["prompt_id","question_id","answer_category","n_trials",
                                  "verdict_k","p_positive","p_negative","p_uncertain",
                                  "majority_verdict"]},
            "semantic_entropy": h, "h_norm": h_norm,
            "active_clusters": k_active, "flip_rate": flip_rate,
        })
    result = pd.DataFrame(rows)
    logger.info(f"SE computed: {len(result)} sets, mean H={result['semantic_entropy'].mean():.4f}, "
                f"mean H_norm={result['h_norm'].mean():.4f}")
    return result

# === 4. 통계 분석 ===
def run_statistical_analysis(edf: pd.DataFrame, config: dict) -> dict:
    results = {}
    prompt_ids = sorted(edf["prompt_id"].unique())

    # --- Cross-prompt H_norm comparison ---
    if len(prompt_ids) >= 2:
        groups = [edf[edf["prompt_id"] == pid]["h_norm"].values for pid in prompt_ids]
        if all(len(g) > 1 for g in groups):
            stat, p = stats.kruskal(*groups)
            results["h_norm_vs_prompt"] = {
                "test": "Kruskal-Wallis", "statistic": float(stat), "p_value": float(p),
                "group_means": {pid: float(edf[edf["prompt_id"]==pid]["h_norm"].mean()) for pid in prompt_ids},
            }
            logger.info(f"  H_norm vs prompt: H={stat:.2f}, p={p:.6f}")

    # --- Per-prompt: entropy vs answer_category ---
    for pid in prompt_ids:
        sub = edf[edf["prompt_id"] == pid]
        cats = sub["answer_category"].unique()
        groups = [sub[sub["answer_category"]==c]["semantic_entropy"].values for c in cats]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            stat, p = stats.kruskal(*groups)
            results[f"entropy_vs_category_{pid}"] = {
                "test": "Kruskal-Wallis", "statistic": float(stat), "p_value": float(p),
                "group_means": {c: float(sub[sub["answer_category"]==c]["semantic_entropy"].mean()) for c in cats},
            }

    # --- Per-prompt: NER tag effect (다빈도 태그만) ---
    min_for_stats = config["sampling"]["min_for_stats"]
    for pid in prompt_ids:
        sub = edf[edf["prompt_id"] == pid]
        if "spacy_label" not in sub.columns:
            continue
        tag_counts = sub["spacy_label"].value_counts()
        valid_tags = tag_counts[tag_counts >= min_for_stats].index.tolist()
        if len(valid_tags) < 2:
            continue
        groups = [sub[sub["spacy_label"]==t]["semantic_entropy"].values for t in valid_tags]
        if all(len(g) > 1 for g in groups):
            stat, p = stats.kruskal(*groups)
            results[f"entropy_vs_ner_{pid}"] = {
                "test": "Kruskal-Wallis", "statistic": float(stat), "p_value": float(p),
                "tags_tested": valid_tags,
                "group_means": {t: float(sub[sub["spacy_label"]==t]["semantic_entropy"].mean()) for t in valid_tags},
            }

    return results

# === 5. Negativity Bias 분석 ===
def analyze_negativity_bias(edf: pd.DataFrame) -> dict:
    """correct 답변에 대한 오판 방향 분석 (프롬프트별)."""
    results = {}
    for pid in sorted(edf["prompt_id"].unique()):
        sub = edf[(edf["prompt_id"] == pid) & (edf["answer_category"] == "correct")]
        total = len(sub)
        if total == 0:
            continue
        n_pos = (sub["majority_verdict"] == POSITIVE).sum()
        n_neg = (sub["majority_verdict"] == NEGATIVE).sum()
        n_unc = (sub["majority_verdict"] == UNCERTAIN).sum()
        k = sub["verdict_k"].iloc[0]
        results[pid] = {
            "total_correct": int(total), "verdict_k": int(k),
            "majority_POSITIVE": int(n_pos), "majority_NEGATIVE": int(n_neg),
            "majority_UNCERTAIN": int(n_unc),
            "misclassification_rate": float((n_neg + n_unc) / total),
            "negative_bias_ratio": float(n_neg / (n_neg + n_unc)) if (n_neg + n_unc) > 0 else None,
        }
        logger.info(f"  [{pid}] correct→POS:{n_pos} NEG:{n_neg} UNC:{n_unc} "
                     f"(neg_bias={results[pid]['negative_bias_ratio']})")
    return results

# === 6. 시각화 ===
def create_visualizations(edf: pd.DataFrame, sfx: str = ""):
    prompt_ids = sorted(edf["prompt_id"].unique())
    n_prompts = len(prompt_ids)

    # Fig 1: H_norm by prompt (cross-prompt comparison)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=edf, x="prompt_id", y="h_norm", order=prompt_ids, ax=ax)
    ax.set_title("Normalized Semantic Entropy by Prompt (H_norm)")
    ax.set_ylabel("H_norm (0~1)")
    plt.tight_layout()
    fig.savefig(RESULTS_ANALYSIS / f"fig1_hnorm_by_prompt{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: SE by answer_category (per prompt)
    fig, axes = plt.subplots(1, n_prompts, figsize=(6*n_prompts, 5), sharey=True)
    if n_prompts == 1: axes = [axes]
    cat_order = ["correct", "obvious_wrong", "confusing_wrong"]
    for ax, pid in zip(axes, prompt_ids):
        sub = edf[edf["prompt_id"] == pid]
        avail = [c for c in cat_order if c in sub["answer_category"].values]
        sns.boxplot(data=sub, x="answer_category", y="semantic_entropy", order=avail, ax=ax)
        ax.set_title(f"{pid}")
        ax.set_xlabel(""); ax.set_ylabel("SE (nats)" if ax == axes[0] else "")
    fig.suptitle("Semantic Entropy by Answer Category (per Prompt)", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_ANALYSIS / f"fig2_entropy_by_category{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: SE by NER tag (가장 안정적인 프롬프트 기준)
    if "spacy_label" in edf.columns:
        # 전체 프롬프트 평균 h_norm이 가장 낮은 프롬프트 선택
        best_pid = edf.groupby("prompt_id")["h_norm"].mean().idxmin()
        sub = edf[edf["prompt_id"] == best_pid]
        tag_order = sub.groupby("spacy_label")["semantic_entropy"].mean().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(max(10, len(tag_order)*0.8), 5))
        sns.boxplot(data=sub, x="spacy_label", y="semantic_entropy", order=tag_order, ax=ax)
        ax.set_title(f"SE by NER Tag [{best_pid}]")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(RESULTS_ANALYSIS / f"fig3_entropy_by_ner{sfx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Fig 4: Negativity Bias — correct 카테고리 verdict 분포
    correct_sub = edf[edf["answer_category"] == "correct"]
    fig, axes = plt.subplots(1, n_prompts, figsize=(5*n_prompts, 4))
    if n_prompts == 1: axes = [axes]
    for ax, pid in zip(axes, prompt_ids):
        sub = correct_sub[correct_sub["prompt_id"] == pid]
        vc = sub["majority_verdict"].value_counts()
        vc.plot.bar(ax=ax, color=["#2ecc71","#e74c3c","#f39c12"][:len(vc)])
        ax.set_title(f"{pid}\n(correct answers)")
        ax.set_ylabel("Count"); ax.set_xlabel("")
    fig.suptitle("Majority Verdict Distribution for Correct Answers", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_ANALYSIS / f"fig4_negativity_bias{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Saved 4 figures to results/analysis/")

# === 7. 결과 저장 ===
def save_results(edf, stat_results, neg_results, sfx=""):
    edf.to_csv(RESULTS_ANALYSIS / f"entropy_results_b1{sfx}.csv", index=False)
    with open(RESULTS_ANALYSIS / f"statistical_tests_b1{sfx}.json", "w") as f:
        json.dump(stat_results, f, indent=2, ensure_ascii=False)
    with open(RESULTS_ANALYSIS / f"negativity_bias_b1{sfx}.json", "w") as f:
        json.dump(neg_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Results saved")

# === 8. 심층 분류 ===
def classify_questions(edf: pd.DataFrame) -> pd.DataFrame:
    df = edf.copy()
    df["expected_verdict"] = df["answer_category"].apply(lambda c: POSITIVE if c == "correct" else NEGATIVE)
    df["is_unstable"] = df["semantic_entropy"] > 0
    df["is_mismatch"] = df["majority_verdict"] != df["expected_verdict"]
    df["issue_type"] = df.apply(
        lambda r: "both" if r["is_unstable"] and r["is_mismatch"]
        else "unstable_only" if r["is_unstable"]
        else "mismatch_only" if r["is_mismatch"]
        else "stable_correct", axis=1)
    return df

def save_deep_results(cdf, sfx=""):
    cdf.to_csv(RESULTS_ANALYSIS / f"classified_b1{sfx}.csv", index=False)
    summary = {"total": len(cdf)}
    for pid in sorted(cdf["prompt_id"].unique()):
        sub = cdf[cdf["prompt_id"] == pid]
        summary[pid] = {
            "total": len(sub),
            **sub["issue_type"].value_counts().to_dict(),
            "per_category": {}
        }
        for cat in ["correct", "obvious_wrong", "confusing_wrong"]:
            cs = sub[sub["answer_category"] == cat]
            summary[pid]["per_category"][cat] = cs["issue_type"].value_counts().to_dict()
    with open(RESULTS_ANALYSIS / f"deep_summary_b1{sfx}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  Deep analysis saved")

# === 9. 요약 출력 ===
def print_summary(edf, stat_results):
    logger.info("=" * 60)
    logger.info("Phase B B1-1 Analysis Summary")
    logger.info("=" * 60)
    for pid in sorted(edf["prompt_id"].unique()):
        sub = edf[edf["prompt_id"] == pid]
        logger.info(f"\n  [{pid}] (K={sub['verdict_k'].iloc[0]})")
        logger.info(f"    mean H={sub['semantic_entropy'].mean():.4f}, "
                     f"mean H_norm={sub['h_norm'].mean():.4f}, "
                     f"mean flip_rate={sub['flip_rate'].mean():.4f}")
        unstable = (sub["flip_rate"] > 0).sum()
        logger.info(f"    unstable: {unstable}/{len(sub)} ({unstable/len(sub)*100:.1f}%)")
        for cat in ["correct", "obvious_wrong", "confusing_wrong"]:
            cs = sub[sub["answer_category"] == cat]
            if len(cs):
                logger.info(f"    {cat:20s}: H={cs['semantic_entropy'].mean():.4f}, "
                             f"flip={cs['flip_rate'].mean():.4f}")

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--config", default="experiment_b1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    sfx = "_smoke" if args.smoke else ""

    # Load
    df = load_results(smoke=args.smoke)
    meta = load_eval_meta(smoke=args.smoke)

    # Verdict distribution
    vdf = compute_verdict_distribution(df, config)

    # Semantic Entropy
    edf = compute_semantic_entropy(vdf)

    # Merge metadata
    merge_cols = ["question_id", "answer_category"]
    meta_cols = [c for c in ["spacy_label","evidence_present","question_length","context_length"] if c in meta.columns]
    edf = edf.merge(meta[merge_cols + meta_cols].drop_duplicates(subset=merge_cols),
                     on=merge_cols, how="left")

    # Stats
    stat_results = run_statistical_analysis(edf, config)
    neg_results = analyze_negativity_bias(edf)

    # Visualize + Save
    create_visualizations(edf, sfx=sfx)
    save_results(edf, stat_results, neg_results, sfx=sfx)
    print_summary(edf, stat_results)

    # Deep analysis
    if not args.skip_deep:
        cdf = classify_questions(edf)
        save_deep_results(cdf, sfx=sfx)

    logger.info("\nDone!")

if __name__ == "__main__":
    main()
