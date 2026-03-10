"""
Phase B Step 2: Judge 반복 샘플링 - 3-Prompt Comparison
- 3개 프롬프트 × N문항 × 3답변유형 × 30회를 Batch API로 제출
- 프롬프트별 별도 배치 또는 통합 배치

Usage:
    python -m src.run_judge_batch submit --config experiment_b1.yaml
    python -m src.run_judge_batch submit --config experiment_b1.yaml --smoke
    python -m src.run_judge_batch status --config experiment_b1.yaml
    python -m src.run_judge_batch download --config experiment_b1.yaml
    python -m src.run_judge_batch auto --config experiment_b1.yaml
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import DATA_PROCESSED, RESULTS_LOGS, append_jsonl, load_config, load_env, read_jsonl, setup_logger
from src.prompts import get_active_prompts, PARSE_ERROR as LABEL_PARSE_ERROR

logger = setup_logger("run_judge_batch")
BATCH_STATE_FILE = RESULTS_LOGS / "batch_state_b1.json"

# === 1. Batch Input 생성 ===
def generate_batch_input(config: dict, smoke: bool = False) -> list[Path]:
    sfx = "_smoke" if smoke else ""
    eval_path = DATA_PROCESSED / f"evaluation_set_b1{sfx}.jsonl"
    if not eval_path.exists():
        logger.error(f"Not found: {eval_path}. Run prepare_data first.")
        sys.exit(1)
    eval_sets = read_jsonl(eval_path)
    judge_cfg = config["judge"]
    batch_cfg = config["batch"]
    n_trials = config["smoke_test"]["n_trials"] if smoke else judge_cfg["n_trials"]
    prompts = get_active_prompts(config)

    all_requests = []
    for prompt_def in prompts:
        pid = prompt_def["id"]
        system_msg = prompt_def["system"].strip()
        user_tpl = prompt_def["user_template"]
        for es in eval_sets:
            user_msg = user_tpl.format(
                context=es["context"], question=es["question"], candidate_answer=es["answer"]
            ).strip()
            for trial in range(1, n_trials + 1):
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

    logger.info(f"Total requests: {len(all_requests)} "
                f"({len(prompts)} prompts × {len(eval_sets)} sets × {n_trials} trials)")

    # 분할
    max_per = max(50, batch_cfg["max_enqueued_tokens"] // batch_cfg["est_tokens_per_request"])
    n_chunks = max(1, (len(all_requests) + max_per - 1) // max_per)
    if n_chunks > 1:
        logger.info(f"  Splitting into {n_chunks} batches (~{max_per} req each)")

    paths = []
    for i in range(n_chunks):
        chunk = all_requests[i * max_per:(i + 1) * max_per]
        if not chunk: break
        p = RESULTS_LOGS / f"batch_input_b1{sfx}_part{i+1}.jsonl"
        if p.exists(): p.unlink()
        for req in chunk:
            append_jsonl(p, req)
        logger.info(f"  Part {i+1}: {len(chunk)} requests, {p.stat().st_size/(1024*1024):.1f} MB")
        paths.append(p)
    return paths

# === 2. Batch 제출 ===
def submit_batch(input_path: Path, smoke: bool, part_index: int) -> str:
    load_env(); client = OpenAI()
    logger.info(f"Uploading {input_path.name}...")
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"experiment": "phase-b-b1-1", "mode": "smoke" if smoke else "full", "part": str(part_index + 1)})
    logger.info(f"  Batch {batch.id} created (status={batch.status})")
    return batch.id

# === 3. 상태 확인 ===
def check_status():
    load_env(); client = OpenAI(); state = _load_state()
    if not state:
        logger.error("No batch state. Run submit first."); sys.exit(1)
    for entry in state.get("batches", []):
        batch = client.batches.retrieve(entry["batch_id"])
        rc = batch.request_counts
        logger.info(f"  {entry['batch_id']} [{batch.status}] "
                     f"{rc.completed}/{rc.total} done, {rc.failed} failed" if rc else f"  {entry['batch_id']} [{batch.status}]")
        entry["status"] = batch.status
        entry["output_file_id"] = batch.output_file_id
        entry["error_file_id"] = batch.error_file_id
    _save_state(state)

# === 4. 결과 다운로드 + 파싱 ===
def download_results(config: dict):
    load_env(); client = OpenAI(); state = _load_state()
    if not state:
        logger.error("No state."); sys.exit(1)
    sfx = state.get("suffix", "")
    log_path = RESULTS_LOGS / f"judge_results_b1{sfx}.jsonl"
    if log_path.exists(): log_path.unlink()
    prompts_map = {p["id"]: p for p in get_active_prompts(config)}
    judge_cfg = config["judge"]
    total_success = total_errors = 0

    for entry in state["batches"]:
        batch = client.batches.retrieve(entry["batch_id"])
        if batch.status != "completed":
            logger.warning(f"  {entry['batch_id']} not completed ({batch.status}), skipping")
            continue
        raw_content = client.files.content(batch.output_file_id)
        raw_path = RESULTS_LOGS / f"batch_output_raw_b1{sfx}_{entry['part']}.jsonl"
        raw_path.write_bytes(raw_content.content)
        raw_records = read_jsonl(raw_path)
        success = errors = 0
        for rec in raw_records:
            cid = rec.get("custom_id", "")
            parts = cid.split("|")
            if len(parts) != 4:
                errors += 1; continue
            prompt_id, qid, acat, trial_str = parts
            trial = int(trial_str)
            resp_body = rec.get("response", {}).get("body", {})
            err = rec.get("error")
            if err:
                append_jsonl(log_path, _error_record(prompt_id, qid, acat, trial, judge_cfg, err))
                errors += 1; continue
            choices = resp_body.get("choices", [])
            raw_text = choices[0].get("message", {}).get("content", "") if choices else ""
            logprobs_data = _extract_logprobs(choices[0]) if choices else None
            # 프롬프트별 파서로 verdict 매핑
            prompt_def = prompts_map.get(prompt_id)
            unified_verdict = prompt_def["parse_fn"](raw_text) if prompt_def else LABEL_PARSE_ERROR
            usage = resp_body.get("usage", {})
            append_jsonl(log_path, {
                "prompt_id": prompt_id, "question_id": qid, "answer_category": acat,
                "trial_number": trial, "seed": judge_cfg["seed"],
                "response_id": resp_body.get("id"),
                "system_fingerprint": resp_body.get("system_fingerprint"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": unified_verdict, "raw_response": raw_text,
                "logprobs": logprobs_data,
                "model": resp_body.get("model", judge_cfg["model"]),
                "usage_prompt_tokens": usage.get("prompt_tokens"),
                "usage_completion_tokens": usage.get("completion_tokens"),
            })
            success += 1
        logger.info(f"  Part {entry['part']}: {success} ok, {errors} err")
        total_success += success; total_errors += errors
    # 요약
    all_results = read_jsonl(log_path)
    logger.info(f"\nTotal: {total_success} success, {total_errors} errors")
    for pid in sorted({r["prompt_id"] for r in all_results}):
        sub = [r for r in all_results if r["prompt_id"] == pid]
        vc = Counter(r["verdict"] for r in sub)
        logger.info(f"  {pid}: {dict(vc)}")

def _error_record(pid, qid, acat, trial, judge_cfg, err):
    return {"prompt_id": pid, "question_id": qid, "answer_category": acat,
            "trial_number": trial, "seed": judge_cfg["seed"],
            "verdict": "API_ERROR", "raw_response": "", "logprobs": None,
            "model": judge_cfg["model"], "timestamp": datetime.now(timezone.utc).isoformat(),
            "parse_error": json.dumps(err)}

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

# === 5. Auto (submit → poll → download) ===
def auto_run(config: dict, smoke: bool, poll_interval: int):
    sfx = "_smoke" if smoke else ""
    input_paths = generate_batch_input(config, smoke=smoke)
    batch_entries = []
    cooldown = config["batch"]["cooldown_between_batches"]
    for i, path in enumerate(input_paths):
        bid = submit_batch(path, smoke=smoke, part_index=i)
        batch_entries.append({"batch_id": bid, "part": i + 1, "status": "submitted"})
        if i < len(input_paths) - 1:
            logger.info(f"Cooldown {cooldown}s..."); time.sleep(cooldown)
    _save_state({"batches": batch_entries, "suffix": sfx, "smoke": smoke})
    # 폴링
    load_env(); client = OpenAI()
    while True:
        all_done = True
        for entry in batch_entries:
            if entry["status"] in ("completed", "failed", "expired", "cancelled"): continue
            batch = client.batches.retrieve(entry["batch_id"])
            entry["status"] = batch.status
            entry["output_file_id"] = batch.output_file_id
            entry["error_file_id"] = batch.error_file_id
            rc = batch.request_counts
            prog = f"{rc.completed}/{rc.total}" if rc else "?"
            logger.info(f"  Part {entry['part']}: {batch.status} ({prog})")
            if batch.status not in ("completed", "failed", "expired", "cancelled"):
                all_done = False
        _save_state({"batches": batch_entries, "suffix": sfx, "smoke": smoke})
        if all_done: break
        time.sleep(poll_interval)
    download_results(config)

# === State management ===
def _save_state(state):
    with open(BATCH_STATE_FILE, "w") as f: json.dump(state, f, indent=2, ensure_ascii=False)
def _load_state():
    if not BATCH_STATE_FILE.exists(): return None
    with open(BATCH_STATE_FILE) as f: return json.load(f)

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["submit","status","download","auto"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--config", default="experiment_b1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke: logger.info("[SMOKE TEST MODE]")
    if args.command == "submit":
        paths = generate_batch_input(config, smoke=args.smoke)
        entries = []
        for i, p in enumerate(paths):
            bid = submit_batch(p, smoke=args.smoke, part_index=i)
            entries.append({"batch_id": bid, "part": i+1, "status": "submitted"})
        sfx = "_smoke" if args.smoke else ""
        _save_state({"batches": entries, "suffix": sfx, "smoke": args.smoke})
    elif args.command == "status": check_status()
    elif args.command == "download": download_results(config)
    elif args.command == "auto": auto_run(config, smoke=args.smoke, poll_interval=args.poll_interval)

if __name__ == "__main__":
    main()
