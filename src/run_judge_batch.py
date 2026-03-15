"""
Phase B Step 2: Judge Batch API — Experiment-Aware
- --experiment b1-1: 3-Prompt × baseline (evidence-present)
- --experiment b2-1: 1-Prompt × same-type swap (PKI verification)
- --experiment b2-2: 1-Prompt × cross-type swap (control)

Usage:
    python -m src.run_judge_batch auto --experiment b1-1 --config experiment_b.yaml
    python -m src.run_judge_batch auto --experiment b2-1 --config experiment_b.yaml
    python -m src.run_judge_batch auto --experiment b2-2 --config experiment_b.yaml
    # 모든 실험 순차 실행:
    python -m src.run_judge_batch auto --experiment all --config experiment_b.yaml
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import DATA_PROCESSED, RESULTS_LOGS, append_jsonl, load_config, load_env, read_jsonl, setup_logger
from src.prompts import get_active_prompts, get_prompt, PARSE_ERROR as LABEL_PARSE_ERROR

logger = setup_logger("run_judge_batch")


# =============================================================
# Experiment → eval_set + prompts 매핑
# =============================================================
def resolve_experiment(experiment: str, config: dict, smoke: bool) -> dict:
    """실험 ID → (eval_set 경로, 프롬프트 목록, 결과 파일명) 반환."""
    sfx = "_smoke" if smoke else ""
    exp_cfg = config["experiments"][experiment]
    eval_file = exp_cfg["eval_set"].replace(".jsonl", f"{sfx}.jsonl")
    eval_path = DATA_PROCESSED / eval_file

    if exp_cfg.get("use_all_prompts", False):
        prompts = get_active_prompts(config)
    else:
        single_id = config["prompts"]["default_single"]
        prompts = [get_prompt(single_id)]

    log_name = f"judge_results_{experiment.replace('-','')}{sfx}.jsonl"
    state_name = f"batch_state_{experiment.replace('-','')}{sfx}.json"

    return {
        "experiment": experiment,
        "eval_path": eval_path,
        "prompts": prompts,
        "log_path": RESULTS_LOGS / log_name,
        "state_path": RESULTS_LOGS / state_name,
        "n_trials": config["smoke_test"]["n_trials"] if smoke else config["judge"]["n_trials"],
    }


# =============================================================
# 1. Batch Input 생성
# =============================================================
def generate_batch_input(exp: dict, config: dict, smoke: bool) -> list[Path]:
    if not exp["eval_path"].exists():
        logger.error(f"Not found: {exp['eval_path']}. Run prepare_data first.")
        sys.exit(1)
    eval_sets = read_jsonl(exp["eval_path"])
    judge_cfg = config["judge"]
    batch_cfg = config["batch"]
    sfx = "_smoke" if smoke else ""
    exp_tag = exp["experiment"].replace("-", "")

    all_requests = []
    for prompt_def in exp["prompts"]:
        pid = prompt_def["id"]
        system_msg = prompt_def["system"].strip()
        user_tpl = prompt_def["user_template"]
        for es in eval_sets:
            user_msg = user_tpl.format(
                context=es["context"], question=es["question"], candidate_answer=es["answer"]
            ).strip()
            for trial in range(1, exp["n_trials"] + 1):
                custom_id = f"{pid}|{es['question_id']}|{es['answer_category']}|{trial}"
                body = {
                    "model": judge_cfg["model"],
                    "messages": ([{"role": "system", "content": system_msg}] if system_msg else [])
                               + [{"role": "user", "content": user_msg}],
                    "temperature": judge_cfg["temperature"],
                    "max_tokens": judge_cfg["max_tokens"],
                    "seed": judge_cfg["seed"],
                    "logprobs": judge_cfg["logprobs"],
                    "top_logprobs": judge_cfg["top_logprobs"],
                }
                all_requests.append({"custom_id": custom_id, "method": "POST",
                                     "url": "/v1/chat/completions", "body": body})

    logger.info(f"[{exp['experiment']}] {len(all_requests)} requests "
                f"({len(exp['prompts'])} prompts x {len(eval_sets)} sets x {exp['n_trials']} trials)")

    max_per = max(50, batch_cfg["max_enqueued_tokens"] // batch_cfg["est_tokens_per_request"])
    n_chunks = max(1, (len(all_requests) + max_per - 1) // max_per)
    if n_chunks > 1:
        logger.info(f"  Splitting into {n_chunks} batches")

    paths = []
    for i in range(n_chunks):
        chunk = all_requests[i * max_per:(i + 1) * max_per]
        if not chunk: break
        p = RESULTS_LOGS / f"batch_input_{exp_tag}{sfx}_part{i+1}.jsonl"
        if p.exists(): p.unlink()
        for req in chunk:
            append_jsonl(p, req)
        logger.info(f"  Part {i+1}: {len(chunk)} requests, {p.stat().st_size/(1024*1024):.1f} MB")
        paths.append(p)
    return paths


# =============================================================
# 2. Batch 제출
# =============================================================
def submit_batch(input_path: Path, exp: dict, smoke: bool, part_index: int) -> str:
    load_env(); client = OpenAI()
    logger.info(f"Uploading {input_path.name}...")
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"experiment": exp["experiment"],
                  "mode": "smoke" if smoke else "full",
                  "part": str(part_index + 1)})
    logger.info(f"  Batch {batch.id} created (status={batch.status})")
    return batch.id


# =============================================================
# 3. 상태 확인
# =============================================================
def check_status(exp: dict):
    load_env(); client = OpenAI()
    state = _load_state(exp["state_path"])
    if not state:
        logger.error("No batch state."); sys.exit(1)
    for entry in state.get("batches", []):
        batch = client.batches.retrieve(entry["batch_id"])
        rc = batch.request_counts
        prog = f"{rc.completed}/{rc.total}" if rc else "?"
        logger.info(f"  Part {entry['part']}: {batch.status} ({prog})")
        entry.update(status=batch.status, output_file_id=batch.output_file_id,
                     error_file_id=batch.error_file_id)
    _save_state(state, exp["state_path"])


# =============================================================
# 4. 결과 다운로드 + 파싱
# =============================================================
def download_results(exp: dict, config: dict):
    load_env(); client = OpenAI()
    state = _load_state(exp["state_path"])
    if not state:
        logger.error("No state."); sys.exit(1)
    log_path = exp["log_path"]
    if log_path.exists(): log_path.unlink()

    prompts_map = {p["id"]: p for p in exp["prompts"]}
    judge_cfg = config["judge"]
    total_s = total_e = 0
    sfx = state.get("suffix", "")
    exp_tag = exp["experiment"].replace("-", "")

    for entry in state["batches"]:
        batch = client.batches.retrieve(entry["batch_id"])
        if batch.status != "completed":
            logger.warning(f"  Part {entry['part']} not completed ({batch.status})")
            continue
        raw_content = client.files.content(batch.output_file_id)
        raw_path = RESULTS_LOGS / f"batch_raw_{exp_tag}{sfx}_{entry['part']}.jsonl"
        raw_path.write_bytes(raw_content.content)
        raw_records = read_jsonl(raw_path)
        s = e = 0
        for rec in raw_records:
            cid = rec.get("custom_id", "")
            parts = cid.split("|")
            if len(parts) != 4:
                e += 1; continue
            prompt_id, qid, acat, trial_str = parts
            trial = int(trial_str)
            resp_body = rec.get("response", {}).get("body", {})
            err = rec.get("error")
            if err:
                append_jsonl(log_path, {
                    "prompt_id": prompt_id, "question_id": qid, "answer_category": acat,
                    "trial_number": trial, "verdict": "API_ERROR", "raw_response": "",
                    "model": judge_cfg["model"],
                    "timestamp": datetime.now(timezone.utc).isoformat()})
                e += 1; continue
            choices = resp_body.get("choices", [])
            raw_text = choices[0].get("message", {}).get("content", "") if choices else ""
            logprobs_data = _extract_logprobs(choices[0]) if choices else None
            prompt_def = prompts_map.get(prompt_id)
            verdict = prompt_def["parse_fn"](raw_text) if prompt_def else LABEL_PARSE_ERROR
            usage = resp_body.get("usage", {})
            append_jsonl(log_path, {
                "prompt_id": prompt_id, "question_id": qid, "answer_category": acat,
                "trial_number": trial, "seed": judge_cfg["seed"],
                "response_id": resp_body.get("id"),
                "system_fingerprint": resp_body.get("system_fingerprint"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": verdict, "raw_response": raw_text,
                "logprobs": logprobs_data,
                "model": resp_body.get("model", judge_cfg["model"]),
                "usage_prompt_tokens": usage.get("prompt_tokens"),
                "usage_completion_tokens": usage.get("completion_tokens"),
            })
            s += 1
        logger.info(f"  Part {entry['part']}: {s} ok, {e} err")
        total_s += s; total_e += e

    all_results = read_jsonl(log_path)
    logger.info(f"\nTotal: {total_s} success, {total_e} errors")
    for pid in sorted({r["prompt_id"] for r in all_results}):
        sub = [r for r in all_results if r["prompt_id"] == pid]
        vc = Counter(r["verdict"] for r in sub)
        logger.info(f"  {pid}: {dict(vc)}")


def _extract_logprobs(choice: dict) -> dict | None:
    lp = choice.get("logprobs")
    if not lp or not lp.get("content"): return None
    tokens = []
    for ti in lp["content"][:20]:
        td = {"token": ti.get("token",""), "logprob": ti.get("logprob")}
        if ti.get("top_logprobs"):
            td["top_logprobs"] = [{"token": tp["token"], "logprob": tp["logprob"]} for tp in ti["top_logprobs"]]
        tokens.append(td)
    return {"tokens": tokens}


# =============================================================
# 5. Auto
# =============================================================
def auto_run(exp: dict, config: dict, smoke: bool, poll_interval: int):
    input_paths = generate_batch_input(exp, config, smoke)
    batch_entries = []
    cooldown = config["batch"]["cooldown_between_batches"]
    for i, path in enumerate(input_paths):
        bid = submit_batch(path, exp, smoke, i)
        batch_entries.append({"batch_id": bid, "part": i + 1, "status": "submitted"})
        if i < len(input_paths) - 1:
            logger.info(f"Cooldown {cooldown}s..."); time.sleep(cooldown)
    sfx = "_smoke" if smoke else ""
    _save_state({"batches": batch_entries, "suffix": sfx, "smoke": smoke}, exp["state_path"])

    load_env(); client = OpenAI()
    while True:
        all_done = True
        for entry in batch_entries:
            if entry["status"] in ("completed","failed","expired","cancelled"): continue
            batch = client.batches.retrieve(entry["batch_id"])
            entry["status"] = batch.status
            entry["output_file_id"] = batch.output_file_id
            entry["error_file_id"] = batch.error_file_id
            rc = batch.request_counts
            prog = f"{rc.completed}/{rc.total}" if rc else "?"
            logger.info(f"  Part {entry['part']}: {batch.status} ({prog})")
            if batch.status not in ("completed","failed","expired","cancelled"):
                all_done = False
        _save_state({"batches": batch_entries, "suffix": sfx, "smoke": smoke}, exp["state_path"])
        if all_done: break
        time.sleep(poll_interval)
    download_results(exp, config)


# === State ===
def _save_state(state, path):
    with open(path, "w") as f: json.dump(state, f, indent=2, ensure_ascii=False)
def _load_state(path):
    if not path.exists(): return None
    with open(path) as f: return json.load(f)


# =============================================================
# Main
# =============================================================
EXPERIMENTS = ["b1-1", "b2-1", "b2-2"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["submit","status","download","auto"])
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS + ["all"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--config", default="experiment_b.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke: logger.info("[SMOKE TEST MODE]")

    exps = EXPERIMENTS if args.experiment == "all" else [args.experiment]

    for exp_id in exps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_id}")
        logger.info(f"{'='*60}")
        exp = resolve_experiment(exp_id, config, args.smoke)

        if args.command == "submit":
            paths = generate_batch_input(exp, config, args.smoke)
            entries = []
            for i, p in enumerate(paths):
                bid = submit_batch(p, exp, args.smoke, i)
                entries.append({"batch_id": bid, "part": i+1, "status": "submitted"})
            sfx = "_smoke" if args.smoke else ""
            _save_state({"batches": entries, "suffix": sfx, "smoke": args.smoke}, exp["state_path"])
        elif args.command == "status":
            check_status(exp)
        elif args.command == "download":
            download_results(exp, config)
        elif args.command == "auto":
            auto_run(exp, config, smoke=args.smoke, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
