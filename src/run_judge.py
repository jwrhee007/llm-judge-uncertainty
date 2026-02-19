"""
Step 2 (Batch): Judge 반복 샘플링 - OpenAI Batch API 버전
- 18,000건을 JSONL로 묶어서 한번에 제출
- Rate limit 없음, 비용 50% 할인
- 결과는 보통 수십 분 내 반환 (최대 24시간)

Usage:
    # 1단계: 배치 생성 + 제출
    python -m src.run_judge_batch submit
    python -m src.run_judge_batch submit --smoke

    # 2단계: 상태 확인
    python -m src.run_judge_batch status

    # 3단계: 결과 다운로드 + 파싱
    python -m src.run_judge_batch download

    # 원스텝: 제출 → 폴링 → 다운로드 (자동 대기)
    python -m src.run_judge_batch auto
    python -m src.run_judge_batch auto --smoke
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DATA_PROCESSED,
    RESULTS_LOGS,
    append_jsonl,
    load_config,
    load_env,
    read_jsonl,
    setup_logger,
)

logger = setup_logger("run_judge_batch")

# 배치 상태 파일 (batch_id 등 저장)
BATCH_STATE_FILE = RESULTS_LOGS / "batch_state.json"


# =============================================================
# 1. Batch Input JSONL 생성
# =============================================================
def generate_batch_input(config: dict, smoke: bool = False) -> list[Path]:
    """
    evaluation_set.jsonl을 읽어 Batch API 입력 JSONL 생성.
    200MB 한도를 넘으면 자동으로 분할.
    Returns: list of input file paths.
    """
    suffix = "_smoke" if smoke else ""
    eval_path = DATA_PROCESSED / f"evaluation_set{suffix}.jsonl"

    if not eval_path.exists():
        logger.error(f"File not found: {eval_path}")
        logger.error("Run 'python -m src.prepare_data' first.")
        sys.exit(1)

    eval_sets = read_jsonl(eval_path)
    judge_cfg = config["judge"]
    prompt_cfg = config["judge_prompt"]
    n_trials = config["smoke_test"]["n_trials"] if smoke else judge_cfg["n_trials"]

    system_msg = prompt_cfg["system"].strip()

    # 모든 request를 먼저 생성
    all_requests = []
    for eval_set in eval_sets:
        user_msg = prompt_cfg["user_template"].format(
            context=eval_set["context"],
            question=eval_set["question"],
            answer=eval_set["answer"],
        ).strip()

        for trial in range(1, n_trials + 1):
            custom_id = f"{eval_set['question_id']}|{eval_set['answer_category']}|{trial}"

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": judge_cfg["model"],
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": judge_cfg["temperature"],
                    "max_tokens": judge_cfg["max_tokens"],
                    "seed": judge_cfg["seed"],
                    "logprobs": judge_cfg["logprobs"],
                    "top_logprobs": judge_cfg["top_logprobs"],
                },
            }
            all_requests.append(request)

    logger.info(f"Total requests: {len(all_requests)}")

    # 토큰 기반 분할 (enqueued token limit 대응)
    # 한도 2M tokens, 보수적으로 30%만 사용 → ~200 requests/batch
    MAX_ENQUEUED_TOKENS = 600_000
    EST_TOKENS_PER_REQUEST = 3_000
    max_requests_per_batch = max(50, MAX_ENQUEUED_TOKENS // EST_TOKENS_PER_REQUEST)
    n_chunks = max(1, (len(all_requests) + max_requests_per_batch - 1) // max_requests_per_batch)

    if n_chunks > 1:
        logger.info(f"  Enqueued token limit: splitting into {n_chunks} batches "
                     f"(~{max_requests_per_batch} requests each)")

    output_paths = []

    for i in range(n_chunks):
        chunk = all_requests[i * max_requests_per_batch : (i + 1) * max_requests_per_batch]
        if not chunk:
            break

        if n_chunks == 1:
            output_path = RESULTS_LOGS / f"batch_input{suffix}.jsonl"
        else:
            output_path = RESULTS_LOGS / f"batch_input{suffix}_part{i+1}.jsonl"

        if output_path.exists():
            output_path.unlink()

        for req in chunk:
            append_jsonl(output_path, req)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Part {i+1}: {len(chunk)} requests, {size_mb:.1f} MB → {output_path.name}")
        output_paths.append(output_path)

    return output_paths


# =============================================================
# 2. Batch 제출
# =============================================================
def submit_batch(input_path: Path, smoke: bool = False, part_index: int = 0) -> str:
    """Batch API에 JSONL 업로드 + 배치 생성."""
    load_env()
    client = OpenAI()

    # 파일 업로드
    logger.info(f"Uploading {input_path.name}...")
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    logger.info(f"  Uploaded file_id: {uploaded.id}")

    # 배치 생성
    logger.info("Creating batch...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "llm-judge-semantic-uncertainty",
            "phase": "baseline",
            "mode": "smoke" if smoke else "full",
            "part": str(part_index + 1),
        },
    )

    logger.info(f"  Batch created: {batch.id}")
    logger.info(f"  Status: {batch.status}")

    return batch.id


# =============================================================
# 3. 상태 확인
# =============================================================
def check_status() -> dict:
    """현재 배치 상태 확인."""
    load_env()
    client = OpenAI()
    state = _load_state()

    if not state:
        logger.error("No batch state found. Run 'submit' first.")
        sys.exit(1)

    batch = client.batches.retrieve(state["batch_id"])

    logger.info(f"Batch ID: {batch.id}")
    logger.info(f"  Status: {batch.status}")
    logger.info(f"  Created: {batch.created_at}")

    if batch.request_counts:
        total = batch.request_counts.total
        completed = batch.request_counts.completed
        failed = batch.request_counts.failed
        logger.info(f"  Progress: {completed}/{total} completed, {failed} failed")

    if batch.output_file_id:
        logger.info(f"  Output file: {batch.output_file_id}")

    if batch.error_file_id:
        logger.info(f"  Error file: {batch.error_file_id}")

    # 상태 업데이트
    state["status"] = batch.status
    state["output_file_id"] = batch.output_file_id
    state["error_file_id"] = batch.error_file_id
    _save_state(state)

    return state


# =============================================================
# 4. 결과 다운로드 + 파싱
# =============================================================
def download_results(config: dict) -> Path:
    """배치 결과 다운로드 → judge_results.jsonl 형식으로 파싱."""
    load_env()
    client = OpenAI()
    state = _load_state()

    if not state:
        logger.error("No batch state found. Run 'submit' first.")
        sys.exit(1)

    # 최신 상태 확인
    batch = client.batches.retrieve(state["batch_id"])
    if batch.status != "completed":
        logger.error(f"Batch not completed yet. Status: {batch.status}")
        if batch.status == "failed":
            logger.error("Batch failed. Check error file.")
        sys.exit(1)

    smoke = state.get("smoke", False)
    suffix = "_smoke" if smoke else ""

    # 결과 다운로드
    logger.info(f"Downloading results (output_file_id: {batch.output_file_id})...")
    result_content = client.files.content(batch.output_file_id)
    raw_path = RESULTS_LOGS / f"batch_output_raw{suffix}.jsonl"
    raw_path.write_bytes(result_content.content)
    logger.info(f"  Raw output saved: {raw_path}")

    # 에러 파일 다운로드 (있으면)
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_path = RESULTS_LOGS / f"batch_errors{suffix}.jsonl"
        error_path.write_bytes(error_content.content)
        logger.info(f"  Error file saved: {error_path}")

    # 파싱: run_judge.py와 동일한 출력 형식으로 변환
    log_path = RESULTS_LOGS / f"judge_results{suffix}.jsonl"
    if log_path.exists():
        log_path.unlink()

    raw_records = read_jsonl(raw_path)
    judge_cfg = config["judge"]
    success_count = 0
    error_count = 0

    for record in raw_records:
        custom_id = record.get("custom_id", "")
        parts = custom_id.split("|")
        if len(parts) != 3:
            logger.warning(f"  Invalid custom_id: {custom_id}")
            error_count += 1
            continue

        question_id, answer_category, trial_str = parts
        trial_number = int(trial_str)

        response_body = record.get("response", {}).get("body", {})
        error_info = record.get("error")

        if error_info:
            # 에러 응답
            parsed_record = {
                "question_id": question_id,
                "answer_category": answer_category,
                "trial_number": trial_number,
                "seed": judge_cfg["seed"],
                "response_id": None,
                "system_fingerprint": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": "API_ERROR",
                "evidence_span": "",
                "rationale": "",
                "logprobs": None,
                "raw_response": "",
                "parse_error": json.dumps(error_info),
                "model": judge_cfg["model"],
                "usage_prompt_tokens": None,
                "usage_completion_tokens": None,
            }
            error_count += 1
        else:
            # 성공 응답 파싱
            choices = response_body.get("choices", [])
            raw_content = ""
            logprobs_data = None
            sys_fingerprint = response_body.get("system_fingerprint")
            resp_id = response_body.get("id")
            usage = response_body.get("usage", {})

            if choices:
                choice = choices[0]
                raw_content = choice.get("message", {}).get("content", "")
                logprobs_data = _extract_logprobs_from_dict(choice)

            parsed_json = _parse_judge_response(raw_content)

            parsed_record = {
                "question_id": question_id,
                "answer_category": answer_category,
                "trial_number": trial_number,
                "seed": judge_cfg["seed"],
                "response_id": resp_id,
                "system_fingerprint": sys_fingerprint,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": parsed_json.get("verdict", "PARSE_ERROR"),
                "evidence_span": parsed_json.get("evidence_span", ""),
                "rationale": parsed_json.get("brief_rationale", ""),
                "logprobs": logprobs_data,
                "raw_response": raw_content,
                "parse_error": parsed_json.get("_parse_error"),
                "model": response_body.get("model", judge_cfg["model"]),
                "usage_prompt_tokens": usage.get("prompt_tokens"),
                "usage_completion_tokens": usage.get("completion_tokens"),
            }
            success_count += 1

        append_jsonl(log_path, parsed_record)

    logger.info(f"Parsed {success_count} success, {error_count} errors → {log_path}")

    # 간단한 verdict 분포 확인
    all_results = read_jsonl(log_path)
    verdict_counts = {}
    for r in all_results:
        v = r.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    logger.info(f"  Verdict distribution: {json.dumps(verdict_counts, indent=2)}")

    # 실제 비용 계산
    total_prompt = sum(r.get("usage_prompt_tokens", 0) or 0 for r in all_results)
    total_completion = sum(r.get("usage_completion_tokens", 0) or 0 for r in all_results)
    # Batch API는 50% 할인
    cost = (total_prompt * 0.075 + total_completion * 0.30) / 1_000_000
    logger.info(f"  Actual cost (batch 50% off): ~${cost:.2f}")
    logger.info(f"  Total tokens: {total_prompt + total_completion:,} "
                f"(prompt: {total_prompt:,}, completion: {total_completion:,})")

    logger.info(f"\nDone! Next step: python -m src.analyze" + (" --smoke" if smoke else ""))
    return log_path


# =============================================================
# 5. 자동 모드: 제출 → 폴링 → 다운로드
# =============================================================
def auto_run(config: dict, smoke: bool = False, poll_interval: int = 30):
    """원스텝 실행: 배치 생성 → 제출 → 완료 대기 → 결과 다운로드."""
    load_env()
    client = OpenAI()
    suffix = "_smoke" if smoke else ""

    # 1. 배치 입력 생성 (자동 분할)
    input_paths = generate_batch_input(config, smoke=smoke)
    logger.info(f"Total {len(input_paths)} batch(es) to process")

    # 최종 결과 파일 - 이미 존재하면 resume
    log_path = RESULTS_LOGS / f"judge_results{suffix}.jsonl"
    start_part = 0

    if log_path.exists():
        from src.utils import read_jsonl
        existing = read_jsonl(log_path)
        existing_count = len([r for r in existing if r.get("verdict") != "API_ERROR"])
        if existing_count > 0:
            # 이미 완료된 파트 수 추정
            judge_cfg = config["judge"]
            n_trials = config["smoke_test"]["n_trials"] if smoke else judge_cfg["n_trials"]
            MAX_ENQUEUED_TOKENS = 600_000
            EST_TOKENS_PER_REQUEST = 3_000
            max_requests_per_batch = max(50, MAX_ENQUEUED_TOKENS // EST_TOKENS_PER_REQUEST)
            start_part = existing_count // max_requests_per_batch
            logger.info(f"Resuming: {existing_count} records exist, skipping parts 1~{start_part}")

    if start_part >= len(input_paths):
        logger.info("All parts already completed!")
        return

    all_batch_ids = []

    # 2. 각 파트별 제출 → 대기 → 다운로드
    for i in range(start_part, len(input_paths)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing batch part {i+1}/{len(input_paths)}")
        logger.info(f"{'='*60}")

        # 제출
        batch_id = submit_batch(input_paths[i], smoke=smoke, part_index=i)
        all_batch_ids.append(batch_id)

        # 상태 저장
        state = {
            "batch_ids": all_batch_ids,
            "current_part": i + 1,
            "total_parts": len(input_paths),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "smoke": smoke,
        }
        _save_state(state)

        # 폴링 대기
        logger.info(f"Waiting for batch {batch_id} (part {i+1})...")
        start_time = time.time()

        while True:
            batch = client.batches.retrieve(batch_id)
            elapsed = time.time() - start_time

            progress = ""
            if batch.request_counts:
                total = batch.request_counts.total
                completed = batch.request_counts.completed
                failed = batch.request_counts.failed
                pct = (completed / total * 100) if total > 0 else 0
                progress = f" [{completed}/{total} = {pct:.0f}%, {failed} failed]"

            logger.info(f"  [{elapsed / 60:.1f}m] Status: {batch.status}{progress}")

            if batch.status == "completed":
                logger.info(f"Batch part {i+1} completed!")
                break
            elif batch.status == "failed":
                # enqueued token limit 에러인지 확인
                error_msg = ""
                if batch.error_file_id:
                    error_content = client.files.content(batch.error_file_id)
                    error_msg = error_content.text[:500]

                if "nqueued token limit" in error_msg or "nqueued token limit" in str(batch.errors):
                    retry_wait = 180
                    logger.warning(f"Enqueued token limit hit. Waiting {retry_wait}s and resubmitting part {i+1}...")
                    time.sleep(retry_wait)
                    # 재제출
                    batch_id = submit_batch(input_paths[i], smoke=smoke, part_index=i)
                    start_time = time.time()
                    continue
                else:
                    logger.error(f"Batch part {i+1} failed!")
                    logger.error(f"  Errors: {error_msg}")
                    sys.exit(1)
            elif batch.status in ("expired", "cancelled"):
                logger.error(f"Batch part {i+1} {batch.status}!")
                sys.exit(1)

            time.sleep(poll_interval)

        # 결과 다운로드 + 파싱 (기존 log_path에 append)
        logger.info(f"Downloading results for part {i+1}...")
        _download_and_append(client, batch, config, log_path, suffix, i + 1)

        # 배치 간 쿨다운 (enqueued token limit 리셋 대기)
        if i < len(input_paths) - 1:
            cooldown = 90
            logger.info(f"Cooldown {cooldown}s before next batch...")
            time.sleep(cooldown)

    # 최종 요약
    all_results = read_jsonl(log_path)
    verdict_counts = {}
    for r in all_results:
        v = r.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    logger.info(f"\n{'='*60}")
    logger.info(f"All batches complete!")
    logger.info(f"{'='*60}")
    logger.info(f"  Total records: {len(all_results)}")
    logger.info(f"  Verdict distribution: {json.dumps(verdict_counts, indent=2)}")

    total_prompt = sum(r.get("usage_prompt_tokens", 0) or 0 for r in all_results)
    total_completion = sum(r.get("usage_completion_tokens", 0) or 0 for r in all_results)
    cost = (total_prompt * 0.075 + total_completion * 0.30) / 1_000_000
    logger.info(f"  Actual cost (batch 50% off): ~${cost:.2f}")
    logger.info(f"\nDone! Next step: python -m src.analyze" + (" --smoke" if smoke else ""))


def _download_and_append(
    client: OpenAI, batch, config: dict, log_path: Path, suffix: str, part_num: int
):
    """단일 배치 결과를 다운로드하여 log_path에 append."""
    result_content = client.files.content(batch.output_file_id)
    raw_path = RESULTS_LOGS / f"batch_output_raw{suffix}_part{part_num}.jsonl"
    raw_path.write_bytes(result_content.content)

    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_path = RESULTS_LOGS / f"batch_errors{suffix}_part{part_num}.jsonl"
        error_path.write_bytes(error_content.content)

    raw_records = read_jsonl(raw_path)
    judge_cfg = config["judge"]
    success = 0
    errors = 0

    for record in raw_records:
        custom_id = record.get("custom_id", "")
        parts = custom_id.split("|")
        if len(parts) != 3:
            errors += 1
            continue

        question_id, answer_category, trial_str = parts
        trial_number = int(trial_str)
        response_body = record.get("response", {}).get("body", {})
        error_info = record.get("error")

        if error_info:
            parsed_record = {
                "question_id": question_id,
                "answer_category": answer_category,
                "trial_number": trial_number,
                "seed": judge_cfg["seed"],
                "response_id": None,
                "system_fingerprint": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": "API_ERROR",
                "evidence_span": "",
                "rationale": "",
                "logprobs": None,
                "raw_response": "",
                "parse_error": json.dumps(error_info),
                "model": judge_cfg["model"],
                "usage_prompt_tokens": None,
                "usage_completion_tokens": None,
            }
            errors += 1
        else:
            choices = response_body.get("choices", [])
            raw_content = ""
            logprobs_data = None

            if choices:
                choice = choices[0]
                raw_content = choice.get("message", {}).get("content", "")
                logprobs_data = _extract_logprobs_from_dict(choice)

            parsed_json = _parse_judge_response(raw_content)

            usage = response_body.get("usage", {})
            parsed_record = {
                "question_id": question_id,
                "answer_category": answer_category,
                "trial_number": trial_number,
                "seed": judge_cfg["seed"],
                "response_id": response_body.get("id"),
                "system_fingerprint": response_body.get("system_fingerprint"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verdict": parsed_json.get("verdict", "PARSE_ERROR"),
                "evidence_span": parsed_json.get("evidence_span", ""),
                "rationale": parsed_json.get("brief_rationale", ""),
                "logprobs": logprobs_data,
                "raw_response": raw_content,
                "parse_error": parsed_json.get("_parse_error"),
                "model": response_body.get("model", judge_cfg["model"]),
                "usage_prompt_tokens": usage.get("prompt_tokens"),
                "usage_completion_tokens": usage.get("completion_tokens"),
            }
            success += 1

        append_jsonl(log_path, parsed_record)

    logger.info(f"  Part {part_num}: {success} success, {errors} errors")


# =============================================================
# Helper: JSON 파싱 (run_judge.py와 동일 로직)
# =============================================================
def _parse_judge_response(raw: str) -> dict:
    """Judge 응답 JSON 파싱."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        inner = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        cleaned = inner.strip()

    try:
        parsed = json.loads(cleaned)
        verdict = parsed.get("verdict", "").upper().strip()
        if verdict not in ("CORRECT", "INCORRECT", "UNSURE"):
            parsed["_parse_error"] = f"Invalid verdict: {verdict}"
            parsed["verdict"] = "PARSE_ERROR"
        else:
            parsed["verdict"] = verdict
            parsed["_parse_error"] = None
        return parsed
    except json.JSONDecodeError as e:
        verdict = _extract_verdict_fallback(raw)
        return {
            "verdict": verdict,
            "evidence_span": "",
            "brief_rationale": "",
            "_parse_error": f"JSON parse failed: {e}",
        }


def _extract_verdict_fallback(raw: str) -> str:
    upper = raw.upper()
    for v in ("INCORRECT", "CORRECT", "UNSURE"):
        if v in upper:
            return v
    return "PARSE_ERROR"


def _extract_logprobs_from_dict(choice: dict) -> dict | None:
    """dict 형태의 choice에서 logprobs 추출."""
    logprobs = choice.get("logprobs")
    if not logprobs or not logprobs.get("content"):
        return None

    tokens = []
    for token_info in logprobs["content"][:20]:
        token_data = {
            "token": token_info.get("token", ""),
            "logprob": token_info.get("logprob"),
        }
        if token_info.get("top_logprobs"):
            token_data["top_logprobs"] = [
                {"token": tp.get("token", ""), "logprob": tp.get("logprob")}
                for tp in token_info["top_logprobs"]
            ]
        tokens.append(token_data)

    return {"tokens": tokens}


# =============================================================
# Helper: 상태 파일 관리
# =============================================================
def _save_state(state: dict):
    with open(BATCH_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _load_state() -> dict | None:
    if not BATCH_STATE_FILE.exists():
        return None
    with open(BATCH_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 2 (Batch): Run LLM Judge via Batch API")
    parser.add_argument(
        "command",
        choices=["submit", "status", "download", "auto"],
        help="submit: 배치 제출 | status: 상태 확인 | download: 결과 다운로드 | auto: 전체 자동",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--poll-interval", type=int, default=30, help="폴링 간격 (초)")
    parser.add_argument("--config", default="experiment.yaml", help="Config file name")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.smoke:
        logger.info("[SMOKE TEST MODE]")

    if args.command == "submit":
        input_paths = generate_batch_input(config, smoke=args.smoke)
        for i, path in enumerate(input_paths):
            submit_batch(path, smoke=args.smoke, part_index=i)

    elif args.command == "status":
        check_status()

    elif args.command == "download":
        download_results(config)

    elif args.command == "auto":
        auto_run(config, smoke=args.smoke, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()