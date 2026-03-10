"""
Phase B Step 1: 데이터 준비
- TriviaQA 전수 태깅 (spaCy native NER 18태그)
- 태그별 층화 추출 (20개/태그 상한)
- Evidence-Aware Context Selection
- Rule-based 오답 생성

Usage:
    python -m src.prepare_data --config experiment_b1.yaml
    python -m src.prepare_data --config experiment_b1.yaml --smoke
"""
from __future__ import annotations
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import spacy
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import DATA_PROCESSED, append_jsonl, load_config, setup_logger

logger = setup_logger("prepare_data")

SPACY_NATIVE_TAGS = [
    "PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT",
    "WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","PERCENT",
    "MONEY","QUANTITY","ORDINAL","CARDINAL",
]
UNTAGGED = "UNTAGGED"
YEAR_PATTERN = re.compile(r"^\d{4}$")
NUMBER_PATTERN = re.compile(r"^[\d,]+\.?\d*$")

# === 1. TriviaQA 전수 로드 ===
def load_triviaqa_all(config: dict) -> list[dict]:
    ds_cfg = config["dataset"]
    logger.info(f"Loading TriviaQA (subset={ds_cfg['subset']}, split={ds_cfg['split']})...")
    ds = load_dataset("trivia_qa", ds_cfg["subset"], split=ds_cfg["split"])
    logger.info(f"  Raw dataset size: {len(ds)}")
    candidates = []
    for item in ds:
        av = item["answer"]["value"]
        if not av or not av.strip():
            continue
        candidates.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "answer": av.strip(),
            "answer_aliases": item["answer"].get("aliases", []),
            "_wiki_contexts": item.get("entity_pages", {}).get("wiki_context", []),
            "_search_contexts": item.get("search_results", {}).get("search_context", []),
        })
    logger.info(f"  After basic filtering: {len(candidates)} questions")
    return candidates

# === 2. spaCy Native NER 태깅 ===
def tag_answer_types(questions: list[dict], nlp) -> list[dict]:
    logger.info("Tagging answer types with spaCy native NER...")
    for q in questions:
        q["spacy_label"] = _tag_single(q["answer"], q["question"], nlp)
    counts = Counter(q["spacy_label"] for q in questions)
    logger.info(f"  Tag distribution ({len(counts)} tags):")
    for tag, c in counts.most_common():
        logger.info(f"    {tag:15s}: {c:5d}")
    return questions

def _tag_single(answer: str, question: str, nlp) -> str:
    context_text = f"{question} The answer is {answer}."
    doc = nlp(context_text)
    for ent in doc.ents:
        if answer.lower() in ent.text.lower() or ent.text.lower() in answer.lower():
            if ent.label_ in SPACY_NATIVE_TAGS:
                return ent.label_
    doc_answer = nlp(answer)
    for ent in doc_answer.ents:
        if ent.label_ in SPACY_NATIVE_TAGS:
            return ent.label_
    return UNTAGGED

# === 3. Evidence-Aware Context Selection ===
def select_contexts(questions: list[dict], config: dict) -> list[dict]:
    ds_cfg = config["dataset"]
    ev_cfg = config["evidence"]
    min_len, max_len = ds_cfg["min_context_length"], ds_cfg["max_context_length"]
    logger.info("Selecting contexts (evidence-aware)...")
    selected = []
    n_present = n_absent = 0
    for q in questions:
        all_aliases = list({a.strip() for a in [q["answer"]] + q.get("answer_aliases", []) if a.strip()})
        ctx_candidates = []
        for ctx in q["_wiki_contexts"]:
            ctx = _truncate(ctx, max_len)
            if len(ctx.split()) >= min_len:
                ctx_candidates.append(("wiki", ctx))
        for ctx in q["_search_contexts"]:
            ctx = _truncate(ctx, max_len)
            if len(ctx.split()) >= min_len:
                ctx_candidates.append(("search", ctx))
        if not ctx_candidates:
            continue
        best_ctx = best_source = best_span = None
        for source, ctx in ctx_candidates:
            span = _find_evidence_span(ctx, all_aliases, ev_cfg)
            if span and (best_ctx is None or len(ctx) > len(best_ctx)):
                best_ctx, best_source, best_span = ctx, source, span
        if best_ctx:
            q.update(context=best_ctx, context_source=best_source, evidence_present=True,
                     evidence_span_start=best_span["start"], evidence_span_end=best_span["end"],
                     evidence_span_text=best_span["text"])
            n_present += 1
        else:
            longest = max(ctx_candidates, key=lambda x: len(x[1]))
            q.update(context=longest[1], context_source=longest[0], evidence_present=False,
                     evidence_span_start=None, evidence_span_end=None, evidence_span_text=None)
            n_absent += 1
        q.pop("_wiki_contexts", None); q.pop("_search_contexts", None)
        selected.append(q)
    logger.info(f"  Context selected: {len(selected)} (present={n_present}, absent={n_absent})")
    return selected

def _truncate(ctx: str, max_words: int) -> str:
    words = ctx.split()
    return " ".join(words[:max_words]) if len(words) > max_words else ctx

def _find_evidence_span(context: str, aliases: list[str], ev_cfg: dict) -> dict | None:
    ctx_lower = context.lower()
    threshold = ev_cfg.get("short_alias_threshold", 3)
    for alias in sorted(aliases, key=len, reverse=True):
        al = alias.lower().strip()
        if not al:
            continue
        idx = ctx_lower.find(al)
        if idx == -1:
            continue
        if len(al) <= threshold:
            if idx > 0 and context[idx - 1].isalnum():
                continue
            end_idx = idx + len(al)
            if end_idx < len(context) and context[end_idx].isalnum():
                continue
        return {"start": idx, "end": idx + len(al), "text": context[idx:idx + len(al)]}
    return None

# === 4. 층화 추출 ===
def stratified_sample(questions: list[dict], config: dict, smoke: bool = False) -> list[dict]:
    max_per = config["smoke_test"]["max_per_tag"] if smoke else config["sampling"]["max_per_tag"]
    rng = random.Random(config["dataset"]["seed"])
    tag_groups = defaultdict(list)
    for q in questions:
        tag_groups[q["spacy_label"]].append(q)
    sampled = []
    logger.info(f"Stratified sampling (max_per_tag={max_per}):")
    for tag in sorted(tag_groups):
        pool = tag_groups[tag]
        chosen = rng.sample(pool, min(len(pool), max_per))
        sampled.extend(chosen)
        logger.info(f"  {tag:15s}: {len(pool):5d} avail → {len(chosen):3d} sampled")
    rng.shuffle(sampled)
    logger.info(f"  Total sampled: {len(sampled)}")
    return sampled

# === 5. 오답 생성 ===
def generate_wrong_answers(questions: list[dict], config: dict) -> list[dict]:
    logger.info("Generating wrong answers...")
    wa_cfg = config["wrong_answers"]
    rng = random.Random(config["dataset"]["seed"] + 1)
    tag_pool = defaultdict(list)
    for q in questions:
        tag_pool[q["spacy_label"]].append(q["answer"])
    all_tags = list(tag_pool.keys())
    for q in questions:
        a, t = q["answer"], q["spacy_label"]
        q["obvious_wrong"] = _cross_type_swap(a, t, tag_pool, all_tags, rng)
        q["confusing_wrong"] = _confusing_wrong(a, t, tag_pool, wa_cfg, rng)
    logger.info("  Done")
    return questions

def _cross_type_swap(answer, tag, tag_pool, all_tags, rng):
    others = [t for t in all_tags if t != tag and tag_pool[t]]
    if not others: return "[NO_CROSS_TYPE_AVAILABLE]"
    ct = rng.choice(others)
    cands = [a for a in tag_pool[ct] if a != answer]
    return rng.choice(cands) if cands else rng.choice(tag_pool[ct])

def _confusing_wrong(answer, tag, tag_pool, wa_cfg, rng):
    np_cfg = wa_cfg["confusing"]["numeric_perturbation"]
    numeric_tags = {"DATE","TIME","CARDINAL","ORDINAL","QUANTITY","MONEY","PERCENT"}
    if tag in numeric_tags:
        p = _numeric_perturbation(answer, np_cfg, rng)
        if p: return p
    same = [a for a in tag_pool.get(tag, []) if a != answer]
    if same: return rng.choice(same)
    p = _numeric_perturbation(answer, np_cfg, rng)
    if p: return p
    all_a = [a for ans in tag_pool.values() for a in ans if a != answer]
    return rng.choice(all_a) if all_a else "[NO_CONFUSING_AVAILABLE]"

def _numeric_perturbation(answer, np_cfg, rng):
    s = answer.strip().replace(",", "")
    if YEAR_PATTERN.match(s):
        return str(int(s) + rng.choice([-1,1]) * rng.choice(np_cfg["year_offsets"]))
    try:
        n = float(s); r = n * rng.choice(np_cfg["number_factors"])
        return str(int(r)) if n == int(n) else f"{r:.2f}"
    except ValueError: pass
    m = re.search(r"(\d+)", answer)
    if m:
        n = int(m.group(1))
        return answer.replace(m.group(1), str(max(0, n + rng.choice([-1,1])*rng.choice(np_cfg["year_offsets"]))))
    return None

# === 6. 저장 ===
def save_dataset(questions, output_name):
    path = DATA_PROCESSED / output_name
    if path.exists(): path.unlink()
    c = 0
    for q in questions:
        base = {k: q[k] for k in ["question_id","question","context","context_source","spacy_label",
                "evidence_present","evidence_span_start","evidence_span_end","evidence_span_text"]}
        base["ground_truth"] = q["answer"]
        base["question_length"] = len(q["question"].split())
        base["context_length"] = len(q["context"].split())
        for acat, field in [("correct","answer"),("obvious_wrong","obvious_wrong"),("confusing_wrong","confusing_wrong")]:
            append_jsonl(path, {**base, "answer": q[field], "answer_category": acat})
            c += 1
    logger.info(f"Saved {c} eval sets → {path}")
    return path

def save_questions_raw(questions, output_name):
    path = DATA_PROCESSED / output_name
    if path.exists(): path.unlink()
    for q in questions:
        append_jsonl(path, {k: v for k, v in q.items() if not k.startswith("_")})
    logger.info(f"Saved {len(questions)} raw questions → {path}")

# === 7. 요약 ===
def print_summary(questions, config):
    logger.info("=" * 60)
    logger.info(f"Phase B Dataset: {len(questions)} questions, {len(questions)*3} eval sets")
    logger.info("=" * 60)
    td = Counter(q["spacy_label"] for q in questions)
    ms = config["sampling"]["min_for_stats"]
    for t, c in td.most_common():
        logger.info(f"  {t:15s}: {c:3d} {'✓' if c >= ms else '⚠ below threshold'}")
    np_ = sum(1 for q in questions if q["evidence_present"])
    logger.info(f"  Evidence present: {np_}/{len(questions)}")

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default="experiment_b1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    questions = load_triviaqa_all(config)
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    questions = tag_answer_types(questions, nlp)
    questions = select_contexts(questions, config)
    questions = stratified_sample(questions, config, smoke=args.smoke)
    questions = generate_wrong_answers(questions, config)
    sfx = "_smoke" if args.smoke else ""
    save_questions_raw(questions, f"questions_b1{sfx}.jsonl")
    save_dataset(questions, f"evaluation_set_b1{sfx}.jsonl")
    print_summary(questions, config)

if __name__ == "__main__":
    main()
