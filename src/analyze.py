"""
Phase B Step 3: Analysis — Experiment-Aware
- --experiment b1-1: 3-Prompt SE comparison + negativity bias + content effect
- --experiment b2-1: PKI analysis (same-type swap)
- --experiment b2-2: PKI analysis (cross-type swap, control)
- --experiment b2-compare: B-1 vs B-2 paired comparison

Usage:
    python -m src.analyze --experiment b1-1 --config experiment_b.yaml
    python -m src.analyze --experiment b2-1 --config experiment_b.yaml
    python -m src.analyze --experiment b2-compare --config experiment_b.yaml
"""
from __future__ import annotations
import argparse, json, re, sys
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


# =============================================================
# Common: Load + SE computation
# =============================================================
def load_results(experiment: str, smoke: bool) -> pd.DataFrame:
    sfx = "_smoke" if smoke else ""
    exp_tag = experiment.replace("-", "")
    path = RESULTS_LOGS / f"judge_results_{exp_tag}{sfx}.jsonl"
    if not path.exists():
        logger.error(f"Not found: {path}"); sys.exit(1)
    df = pd.DataFrame(read_jsonl(path))
    logger.info(f"Loaded {len(df)} records from {path.name}")
    n_err = df[df["verdict"].isin(["API_ERROR", LABEL_PARSE_ERROR])].shape[0]
    if n_err:
        logger.warning(f"  Excluding {n_err} errors")
        df = df[~df["verdict"].isin(["API_ERROR", LABEL_PARSE_ERROR])].copy()
    return df


def load_eval_meta(eval_set_name: str, smoke: bool) -> pd.DataFrame:
    sfx = "_smoke" if smoke else ""
    name = eval_set_name.replace(".jsonl", f"{sfx}.jsonl")
    path = DATA_PROCESSED / name
    meta = pd.DataFrame(read_jsonl(path))
    meta = meta.drop_duplicates(subset=["question_id", "answer_category"])
    return meta


def compute_verdict_distribution(df: pd.DataFrame, verdict_k_map: dict) -> pd.DataFrame:
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


def compute_semantic_entropy(vdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in vdf.iterrows():
        probs = [r["p_positive"], r["p_negative"]]
        if r["verdict_k"] >= 3:
            probs.append(r["p_uncertain"])
        active = [p for p in probs if p > 0]
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
            "active_clusters": len(active), "flip_rate": flip_rate,
        })
    return pd.DataFrame(rows)


def merge_metadata(edf: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["question_id", "answer_category"]
    available = [c for c in ["spacy_label","evidence_present","question_length","context_length",
                             "swap_type","swap_source_qid","swap_source_tag"] if c in meta.columns]
    return edf.merge(meta[merge_cols + available].drop_duplicates(subset=merge_cols),
                     on=merge_cols, how="left")


# =============================================================
# B1-1 Analysis
# =============================================================
def analyze_b11(config: dict, smoke: bool):
    logger.info("=== B1-1: 3-Prompt Comparison ===")
    sfx = "_smoke" if smoke else ""
    df = load_results("b1-1", smoke)
    meta = load_eval_meta("evaluation_set_b1.jsonl", smoke)
    vdf = compute_verdict_distribution(df, config["prompts"]["verdict_k"])
    edf = compute_semantic_entropy(vdf)
    edf = merge_metadata(edf, meta)
    logger.info(f"SE: {len(edf)} sets, mean H={edf['semantic_entropy'].mean():.4f}, "
                f"mean H_norm={edf['h_norm'].mean():.4f}")

    # Stats
    stat_results = _b11_statistics(edf, config)
    neg_results = _negativity_bias(edf)

    # Visualize
    _b11_visualizations(edf, sfx)

    # Deep classify
    cdf = _classify_questions(edf)

    # Save
    edf.to_csv(RESULTS_ANALYSIS / f"entropy_b11{sfx}.csv", index=False)
    cdf.to_csv(RESULTS_ANALYSIS / f"classified_b11{sfx}.csv", index=False)
    _save_json(stat_results, f"stats_b11{sfx}.json")
    _save_json(neg_results, f"negativity_b11{sfx}.json")
    _save_deep_summary(cdf, f"deep_b11{sfx}.json")

    # Print
    _print_b11_summary(edf, stat_results, neg_results)


def _b11_statistics(edf, config):
    results = {}
    pids = sorted(edf["prompt_id"].unique())
    # Cross-prompt H_norm
    if len(pids) >= 2:
        groups = [edf[edf["prompt_id"]==p]["h_norm"].values for p in pids]
        if all(len(g) > 1 for g in groups):
            s, p = stats.kruskal(*groups)
            results["h_norm_vs_prompt"] = {"test": "Kruskal-Wallis", "H": float(s), "p": float(p),
                "means": {pid: float(edf[edf["prompt_id"]==pid]["h_norm"].mean()) for pid in pids}}
            logger.info(f"  H_norm vs prompt: H={s:.2f}, p={p:.6f}")
    # Per-prompt: category
    for pid in pids:
        sub = edf[edf["prompt_id"]==pid]
        cats = sub["answer_category"].unique()
        groups = [sub[sub["answer_category"]==c]["semantic_entropy"].values for c in cats]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            s, p = stats.kruskal(*groups)
            results[f"SE_vs_cat_{pid}"] = {"test": "Kruskal-Wallis", "H": float(s), "p": float(p),
                "means": {c: float(sub[sub["answer_category"]==c]["semantic_entropy"].mean()) for c in cats}}
    # Per-prompt: NER
    ms = config["sampling"]["min_for_stats"]
    for pid in pids:
        sub = edf[edf["prompt_id"]==pid]
        if "spacy_label" not in sub.columns: continue
        valid = sub["spacy_label"].value_counts()
        valid = valid[valid >= ms].index.tolist()
        if len(valid) < 2: continue
        groups = [sub[sub["spacy_label"]==t]["semantic_entropy"].values for t in valid]
        if all(len(g) > 1 for g in groups):
            s, p = stats.kruskal(*groups)
            results[f"SE_vs_ner_{pid}"] = {"test": "Kruskal-Wallis", "H": float(s), "p": float(p),
                "tags": valid,
                "means": {t: float(sub[sub["spacy_label"]==t]["semantic_entropy"].mean()) for t in valid}}
    return results


def _negativity_bias(edf):
    results = {}
    for pid in sorted(edf["prompt_id"].unique()):
        sub = edf[(edf["prompt_id"]==pid) & (edf["answer_category"]=="correct")]
        n = len(sub)
        if not n: continue
        np_ = int((sub["majority_verdict"]==POSITIVE).sum())
        nn = int((sub["majority_verdict"]==NEGATIVE).sum())
        nu = int((sub["majority_verdict"]==UNCERTAIN).sum())
        results[pid] = {"total": n, "POSITIVE": np_, "NEGATIVE": nn, "UNCERTAIN": nu,
                        "misclass_rate": float((nn+nu)/n),
                        "neg_bias_ratio": float(nn/(nn+nu)) if (nn+nu) > 0 else None}
        logger.info(f"  [{pid}] correct -> POS:{np_} NEG:{nn} UNC:{nu}")
    return results


def _b11_visualizations(edf, sfx):
    pids = sorted(edf["prompt_id"].unique())
    n_p = len(pids)

    # Fig 1: H_norm by prompt
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=edf, x="prompt_id", y="h_norm", order=pids, ax=ax)
    ax.set_title("H_norm by Prompt"); ax.set_ylabel("H_norm (0~1)")
    plt.tight_layout()
    fig.savefig(RESULTS_ANALYSIS / f"fig1_hnorm_by_prompt{sfx}.png", dpi=150); plt.close()

    # Fig 2: SE by category per prompt
    fig, axes = plt.subplots(1, n_p, figsize=(6*n_p, 5), sharey=True)
    if n_p == 1: axes = [axes]
    cat_order = ["correct","obvious_wrong","confusing_wrong"]
    for ax, pid in zip(axes, pids):
        sub = edf[edf["prompt_id"]==pid]
        avail = [c for c in cat_order if c in sub["answer_category"].values]
        sns.boxplot(data=sub, x="answer_category", y="semantic_entropy", order=avail, ax=ax)
        ax.set_title(pid); ax.set_xlabel("")
    fig.suptitle("SE by Answer Category", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_ANALYSIS / f"fig2_se_by_category{sfx}.png", dpi=150); plt.close()

    # Fig 3: SE by NER tag
    if "spacy_label" in edf.columns:
        best_pid = edf.groupby("prompt_id")["h_norm"].mean().idxmin()
        sub = edf[edf["prompt_id"]==best_pid]
        tag_order = sub.groupby("spacy_label")["semantic_entropy"].mean().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(max(10, len(tag_order)*0.8), 5))
        sns.boxplot(data=sub, x="spacy_label", y="semantic_entropy", order=tag_order, ax=ax)
        ax.set_title(f"SE by NER Tag [{best_pid}]")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(RESULTS_ANALYSIS / f"fig3_se_by_ner{sfx}.png", dpi=150); plt.close()

    # Fig 4: Negativity bias
    correct_sub = edf[edf["answer_category"]=="correct"]
    fig, axes = plt.subplots(1, n_p, figsize=(5*n_p, 4))
    if n_p == 1: axes = [axes]
    for ax, pid in zip(axes, pids):
        sub = correct_sub[correct_sub["prompt_id"]==pid]
        vc = sub["majority_verdict"].value_counts()
        vc.plot.bar(ax=ax, color=["#2ecc71","#e74c3c","#f39c12"][:len(vc)])
        ax.set_title(f"{pid}\n(correct answers)"); ax.set_ylabel("Count")
    fig.suptitle("Majority Verdict for Correct Answers", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_ANALYSIS / f"fig4_neg_bias{sfx}.png", dpi=150); plt.close()
    logger.info("  Saved B1-1 figures")


# =============================================================
# B2 Analysis (PKI)
# =============================================================
def analyze_b2(experiment: str, config: dict, smoke: bool):
    swap_label = "Same-Type" if experiment == "b2-1" else "Cross-Type"
    logger.info(f"=== {experiment.upper()}: PKI Analysis ({swap_label} Swap) ===")
    sfx = "_smoke" if smoke else ""
    exp_tag = experiment.replace("-","")

    eval_name = config["experiments"][experiment]["eval_set"]
    df = load_results(experiment, smoke)
    meta = load_eval_meta(eval_name, smoke)
    vdf = compute_verdict_distribution(df, config["prompts"]["verdict_k"])
    edf = compute_semantic_entropy(vdf)
    edf = merge_metadata(edf, meta)
    logger.info(f"SE: {len(edf)} sets, mean H={edf['semantic_entropy'].mean():.4f}")

    # PKI analysis: correct 답변이 swapped context에서 POSITIVE로 판정된 비율
    pki_results = _pki_analysis(edf, df, config)

    # Save
    edf.to_csv(RESULTS_ANALYSIS / f"entropy_{exp_tag}{sfx}.csv", index=False)
    _save_json(pki_results, f"pki_{exp_tag}{sfx}.json")

    # Print
    _print_pki_summary(pki_results, experiment, swap_label)


def _pki_analysis(edf, raw_df, config):
    results = {}
    for pid in sorted(edf["prompt_id"].unique()):
        sub = edf[edf["prompt_id"] == pid]

        # PKI rate: correct 답변에서 POSITIVE로 판정된 비율
        correct_sub = sub[sub["answer_category"] == "correct"]
        n_total = len(correct_sub)
        n_pki = int((correct_sub["majority_verdict"] == POSITIVE).sum())
        pki_rate = n_pki / n_total if n_total > 0 else 0.0

        # Wrong answer에서도 POSITIVE로 판정 (심각한 PKI)
        wrong_sub = sub[sub["answer_category"].isin(["obvious_wrong","confusing_wrong"])]
        n_wrong_total = len(wrong_sub)
        n_wrong_pki = int((wrong_sub["majority_verdict"] == POSITIVE).sum())

        # PKI marker 분석 (rationale이 있는 프롬프트에서만)
        pki_markers = config["analysis"].get("pki_markers", [])
        marker_counts = {}
        if pki_markers:
            pid_trials = raw_df[raw_df["prompt_id"] == pid]
            for marker in pki_markers:
                count = int(pid_trials["raw_response"].str.lower().str.contains(
                    re.escape(marker.lower()), na=False).sum())
                if count > 0:
                    marker_counts[marker] = count

        # NER 태그별 PKI rate
        tag_pki = {}
        if "spacy_label" in correct_sub.columns:
            for tag in correct_sub["spacy_label"].unique():
                tag_sub = correct_sub[correct_sub["spacy_label"] == tag]
                if len(tag_sub) > 0:
                    tag_pki[tag] = {
                        "total": len(tag_sub),
                        "pki": int((tag_sub["majority_verdict"] == POSITIVE).sum()),
                        "rate": float((tag_sub["majority_verdict"] == POSITIVE).sum() / len(tag_sub))
                    }

        results[pid] = {
            "pki_rate": float(pki_rate),
            "correct_total": n_total,
            "correct_pki": n_pki,
            "wrong_total": n_wrong_total,
            "wrong_pki": n_wrong_pki,
            "pki_markers": marker_counts,
            "pki_by_tag": tag_pki,
        }
    return results


def _print_pki_summary(pki_results, experiment, swap_label):
    logger.info(f"\n{'='*60}")
    logger.info(f"PKI Summary ({swap_label} Swap)")
    logger.info(f"{'='*60}")
    for pid, r in pki_results.items():
        logger.info(f"\n  [{pid}]")
        logger.info(f"    Correct answers: {r['correct_pki']}/{r['correct_total']} "
                     f"judged POSITIVE (PKI rate={r['pki_rate']:.1%})")
        logger.info(f"    Wrong answers:   {r['wrong_pki']}/{r['wrong_total']} judged POSITIVE")
        if r["pki_markers"]:
            logger.info(f"    PKI markers: {r['pki_markers']}")
        if r["pki_by_tag"]:
            top_tags = sorted(r["pki_by_tag"].items(), key=lambda x: -x[1]["rate"])[:5]
            for tag, info in top_tags:
                if info["pki"] > 0:
                    logger.info(f"    {tag:15s}: {info['pki']}/{info['total']} ({info['rate']:.0%})")


# =============================================================
# B2 Compare (B-1 vs B-2 paired)
# =============================================================
def analyze_b2_compare(config: dict, smoke: bool):
    logger.info("=== B-1 vs B-2 Paired Comparison ===")
    sfx = "_smoke" if smoke else ""

    # Load B1-1
    b1_df = load_results("b1-1", smoke)
    # 단일 프롬프트만 (B-2와 동일한 프롬프트)
    single_pid = config["prompts"]["default_single"]
    b1_df = b1_df[b1_df["prompt_id"] == single_pid]
    vdf_b1 = compute_verdict_distribution(b1_df, config["prompts"]["verdict_k"])
    edf_b1 = compute_semantic_entropy(vdf_b1)
    edf_b1["condition"] = "B1_original"

    results = {"prompt": single_pid}
    for b2_exp in ["b2-1", "b2-2"]:
        try:
            b2_df = load_results(b2_exp, smoke)
        except SystemExit:
            logger.warning(f"  {b2_exp} results not found, skipping")
            continue
        vdf_b2 = compute_verdict_distribution(b2_df, config["prompts"]["verdict_k"])
        edf_b2 = compute_semantic_entropy(vdf_b2)
        swap_label = "same_type" if b2_exp == "b2-1" else "cross_type"
        edf_b2["condition"] = f"B2_{swap_label}"

        # Paired comparison: correct 답변만
        b1_correct = edf_b1[edf_b1["answer_category"]=="correct"].set_index("question_id")
        b2_correct = edf_b2[edf_b2["answer_category"]=="correct"].set_index("question_id")
        common_qids = b1_correct.index.intersection(b2_correct.index)
        if len(common_qids) > 0:
            b1_verdicts = b1_correct.loc[common_qids, "majority_verdict"]
            b2_verdicts = b2_correct.loc[common_qids, "majority_verdict"]
            # Verdict shift matrix
            shift = pd.crosstab(b1_verdicts, b2_verdicts,
                               rownames=["B1"], colnames=[f"B2_{swap_label}"])
            logger.info(f"\n  Verdict Shift (B1 → {b2_exp}), n={len(common_qids)}:")
            logger.info(f"\n{shift.to_string()}")

            n_b1_pos = (b1_verdicts == POSITIVE).sum()
            n_b2_pos = (b2_verdicts == POSITIVE).sum()
            results[b2_exp] = {
                "n_paired": len(common_qids),
                "b1_positive": int(n_b1_pos),
                "b2_positive": int(n_b2_pos),
                "verdict_shift": shift.to_dict(),
            }

    _save_json(results, f"b2_compare{sfx}.json")
    logger.info("  Comparison saved")


# =============================================================
# Common helpers
# =============================================================
def _classify_questions(edf):
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


def _save_json(data, filename):
    path = RESULTS_ANALYSIS / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Saved {path.name}")


def _save_deep_summary(cdf, filename):
    summary = {"total": len(cdf)}
    for pid in sorted(cdf["prompt_id"].unique()):
        sub = cdf[cdf["prompt_id"]==pid]
        summary[pid] = {"total": len(sub), **sub["issue_type"].value_counts().to_dict(),
                        "per_category": {}}
        for cat in ["correct","obvious_wrong","confusing_wrong"]:
            cs = sub[sub["answer_category"]==cat]
            if len(cs): summary[pid]["per_category"][cat] = cs["issue_type"].value_counts().to_dict()
    _save_json(summary, filename)


def _print_b11_summary(edf, stat_results, neg_results):
    logger.info(f"\n{'='*60}")
    logger.info("B1-1 Summary")
    logger.info(f"{'='*60}")
    for pid in sorted(edf["prompt_id"].unique()):
        sub = edf[edf["prompt_id"]==pid]
        unstable = (sub["flip_rate"] > 0).sum()
        logger.info(f"\n  [{pid}] K={sub['verdict_k'].iloc[0]}")
        logger.info(f"    H={sub['semantic_entropy'].mean():.4f}, H_norm={sub['h_norm'].mean():.4f}, "
                     f"flip={sub['flip_rate'].mean():.4f}")
        logger.info(f"    unstable: {unstable}/{len(sub)} ({unstable/len(sub)*100:.1f}%)")
        for cat in ["correct","obvious_wrong","confusing_wrong"]:
            cs = sub[sub["answer_category"]==cat]
            if len(cs):
                logger.info(f"    {cat:20s}: H={cs['semantic_entropy'].mean():.4f}")
    if neg_results:
        logger.info(f"\n  Negativity Bias:")
        for pid, r in neg_results.items():
            logger.info(f"    [{pid}] misclass={r['misclass_rate']:.1%}, neg_bias={r['neg_bias_ratio']}")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["b1-1","b2-1","b2-2","b2-compare"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default="experiment_b.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.experiment == "b1-1":
        analyze_b11(config, args.smoke)
    elif args.experiment in ("b2-1", "b2-2"):
        analyze_b2(args.experiment, config, args.smoke)
    elif args.experiment == "b2-compare":
        analyze_b2_compare(config, args.smoke)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
