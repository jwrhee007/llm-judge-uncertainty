"""
Step 2: Judge 반복 샘플링
- GPT-4o-mini에 (질문, 지문, 답변) 세트를 반복 제출
- T=0, seed 고정 (pseudo-deterministic baseline)
- JSONL로 로그 저장 (checkpoint/resume 지원)
- asyncio + Semaphore로 rate limit 준수

Usage:
    python -m src.run_judge             # 풀 실험 (18,000 API calls)
    python -m src.run_judge --smoke     # 스모크 테스트 (45 calls)
    python -m src.run_judge --resume    # 중단된 실험 이어서 실행
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    DATA_PROCESSED,
    RESULTS_LOGS,
    append_jsonl,
    get_completed_keys,
    load_config,
    load_env,
    read_jsonl,
    setup_logger,
)

logger = setup_logger("run_judge")


# =============================================================
# 1. 평가 세트 로드
# =============================================================
def load_evaluation_set(config: dict, smoke: bool = False) -> list[dict]:
    """evaluation_set.jsonl에서 평가 세트 로드."""
    suffix = "_smoke" if smoke else ""
    filepath = DATA_PROCESSED / f"evaluation_set{suffix}.jsonl"

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        logger.error("Run 'python -m src.prepare_data' first.")
        sys.exit(1)

    records = read_jsonl(filepath)
    logger.info(f"Loaded {len(records)} evaluation sets from {filepath}")
    return records


# =============================================================
# 2. 단일 API 호출
# =============================================================
async def call_judge(
    client: AsyncOpenAI,
    eval_set: dict,
    trial_number: int,
    config: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    단일 Judge 판단 API 호출.
    retry with exponential backoff 포함.
    """
    judge_cfg = config["judge"]
    prompt_cfg = config["judge_prompt"]
    api_cfg = config["api"]

    # 프롬프트 구성
    system_msg = prompt_cfg["system"].strip()
    user_msg = prompt_cfg["user_template"].format(
        context=eval_set["context"],
        question=eval_set["question"],
        answer=eval_set["answer"],
    ).strip()

    # Retry loop
    last_error = None
    for attempt in range(api_cfg["max_retries"]):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=judge_cfg["model"],
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=judge_cfg["temperature"],
                    max_tokens=judge_cfg["max_tokens"],
                    seed=judge_cfg["seed"],
                    logprobs=judge_cfg["logprobs"],
                    top_logprobs=judge_cfg["top_logprobs"],
                )

                # 응답 파싱
                choice = response.choices[0]
                raw_content = choice.message.content or ""

                # JSON 파싱 시도
                parsed = _parse_judge_response(raw_content)

                # logprobs 추출
                logprobs_data = _extract_logprobs(choice)

                return {
                    "question_id": eval_set["question_id"],
                    "answer_category": eval_set["answer_category"],
                    "trial_number": trial_number,
                    "seed": judge_cfg["seed"],
                    "response_id": response.id,
                    "system_fingerprint": response.system_fingerprint,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "verdict": parsed.get("verdict", "PARSE_ERROR"),
                    "evidence_span": parsed.get("evidence_span", ""),
                    "rationale": parsed.get("brief_rationale", ""),
                    "logprobs": logprobs_data,
                    "raw_response": raw_content,
                    "parse_error": parsed.get("_parse_error"),
                    "model": response.model,
                    "usage_prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "usage_completion_tokens": response.usage.completion_tokens if response.usage else None,
                }

            except RateLimitError as e:
                last_error = str(e)
                wait = api_cfg["retry_base_delay"] * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)

            except APITimeoutError as e:
                last_error = str(e)
                wait = api_cfg["retry_base_delay"] * (2 ** attempt)
                logger.warning(f"Timeout, retrying in {wait:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)

            except APIError as e:
                last_error = str(e)
                wait = api_cfg["retry_base_delay"] * (2 ** attempt)
                logger.warning(f"API error: {e}, retrying in {wait:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)

    # 모든 재시도 실패
    logger.error(
        f"All retries failed for {eval_set['question_id']}/"
        f"{eval_set['answer_category']}/trial_{trial_number}: {last_error}"
    )
    return {
        "question_id": eval_set["question_id"],
        "answer_category": eval_set["answer_category"],
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
        "parse_error": last_error,
        "model": judge_cfg["model"],
        "usage_prompt_tokens": None,
        "usage_completion_tokens": None,
    }


def _parse_judge_response(raw: str) -> dict:
    """
    Judge 응답에서 JSON 파싱.
    ```json ... ``` 블록이나 raw JSON 모두 처리.
    """
    # markdown 코드블록 제거
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # ```json\n...\n``` 형태
        lines = cleaned.split("\n")
        # 첫 줄과 마지막 줄 제거
        inner = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        cleaned = inner.strip()

    try:
        parsed = json.loads(cleaned)
        # verdict 정규화
        verdict = parsed.get("verdict", "").upper().strip()
        if verdict not in ("CORRECT", "INCORRECT", "UNSURE"):
            parsed["_parse_error"] = f"Invalid verdict: {verdict}"
            parsed["verdict"] = "PARSE_ERROR"
        else:
            parsed["verdict"] = verdict
            parsed["_parse_error"] = None
        return parsed

    except json.JSONDecodeError as e:
        # JSON 파싱 실패 시 verdict만이라도 추출 시도
        verdict = _extract_verdict_fallback(raw)
        return {
            "verdict": verdict,
            "evidence_span": "",
            "brief_rationale": "",
            "_parse_error": f"JSON parse failed: {e}. Raw: {raw[:200]}",
        }


def _extract_verdict_fallback(raw: str) -> str:
    """JSON 파싱 실패 시 텍스트에서 verdict 키워드 추출."""
    upper = raw.upper()
    for v in ("INCORRECT", "CORRECT", "UNSURE"):  # INCORRECT를 먼저 체크 (CORRECT 포함 방지)
        if v in upper:
            return v
    return "PARSE_ERROR"


def _extract_logprobs(choice) -> dict | None:
    """응답에서 logprobs 정보 추출."""
    if not choice.logprobs or not choice.logprobs.content:
        return None

    # 처음 몇 토큰의 logprob 수집 (verdict 관련)
    tokens = []
    for token_info in choice.logprobs.content[:20]:  # 처음 20토큰
        token_data = {
            "token": token_info.token,
            "logprob": token_info.logprob,
        }
        if token_info.top_logprobs:
            token_data["top_logprobs"] = [
                {"token": tp.token, "logprob": tp.logprob}
                for tp in token_info.top_logprobs
            ]
        tokens.append(token_data)

    return {"tokens": tokens}


# =============================================================
# 3. 배치 실행 (async)
# =============================================================
async def run_experiment(config: dict, smoke: bool = False, resume: bool = True):
    """전체 실험 실행: 평가 세트 × N회 반복."""
    load_env()
    client = AsyncOpenAI()

    judge_cfg = config["judge"]
    api_cfg = config["api"]

    # 스모크 테스트 설정
    n_trials = config["smoke_test"]["n_trials"] if smoke else judge_cfg["n_trials"]

    # 평가 세트 로드
    eval_sets = load_evaluation_set(config, smoke=smoke)

    # 로그 파일 경로
    suffix = "_smoke" if smoke else ""
    log_path = RESULTS_LOGS / f"judge_results{suffix}.jsonl"

    # 이미 완료된 항목 확인 (resume 지원)
    completed = set()
    if resume and log_path.exists():
        completed = get_completed_keys(log_path)
        if completed:
            logger.info(f"Resuming: {len(completed)} calls already completed")

    # 실행할 태스크 목록 생성
    tasks_to_run = []
    for eval_set in eval_sets:
        for trial in range(1, n_trials + 1):
            key = f"{eval_set['question_id']}_{eval_set['answer_category']}_{trial}"
            if key not in completed:
                tasks_to_run.append((eval_set, trial))

    total_calls = len(tasks_to_run)
    total_expected = len(eval_sets) * n_trials
    logger.info(f"Total API calls: {total_calls} / {total_expected} (skipped: {total_expected - total_calls})")

    if total_calls == 0:
        logger.info("All calls already completed. Nothing to do.")
        return log_path

    # 예상 비용 출력
    est_input = total_calls * 500  # ~500 tokens/call
    est_output = total_calls * 100  # ~100 tokens/call
    est_cost = (est_input * 0.15 + est_output * 0.60) / 1_000_000
    logger.info(f"Estimated cost: ~${est_cost:.2f}")

    # Rate limiter: RPM 기반 동시 요청 제한
    # RPM / 60 = 초당 허용 요청 수, 안전 마진 80%
    max_concurrent = max(1, int(api_cfg["requests_per_minute"] / 60 * 0.8))
    semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Max concurrent requests: {max_concurrent}")

    # 진행률 바
    pbar = tqdm(total=total_calls, desc="Judge API calls", unit="call")

    # 배치 실행 (checkpoint_interval 단위로 끊어서 실행)
    checkpoint_interval = api_cfg["checkpoint_interval"]
    errors = 0
    start_time = time.time()

    for batch_start in range(0, total_calls, checkpoint_interval):
        batch = tasks_to_run[batch_start:batch_start + checkpoint_interval]

        # 비동기 태스크 생성
        coros = [
            call_judge(client, eval_set, trial, config, semaphore)
            for eval_set, trial in batch
        ]

        # 비동기 실행
        results = await asyncio.gather(*coros, return_exceptions=True)

        # 결과 저장
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected exception: {result}")
                errors += 1
                pbar.update(1)
                continue

            append_jsonl(log_path, result)

            if result["verdict"] == "API_ERROR":
                errors += 1
            if result.get("parse_error"):
                logger.debug(f"Parse issue: {result['parse_error'][:100]}")

            pbar.update(1)

        # 배치 완료 로그
        elapsed = time.time() - start_time
        done = min(batch_start + checkpoint_interval, total_calls)
        rate = done / elapsed if elapsed > 0 else 0
        logger.info(
            f"Checkpoint: {done}/{total_calls} calls "
            f"({elapsed:.0f}s elapsed, {rate:.1f} calls/s, {errors} errors)"
        )

    pbar.close()

    # 최종 요약
    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Experiment Complete")
    logger.info("=" * 60)
    logger.info(f"  Total calls: {total_calls}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Time: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}m)")
    logger.info(f"  Rate: {total_calls / elapsed_total:.1f} calls/s")
    logger.info(f"  Log file: {log_path}")

    # 간단한 verdict 분포 확인
    all_results = read_jsonl(log_path)
    verdict_counts = {}
    for r in all_results:
        v = r.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    logger.info(f"  Verdict distribution: {json.dumps(verdict_counts, indent=2)}")

    logger.info(f"\nDone! Next step: python -m src.analyze" + (" --smoke" if smoke else ""))
    return log_path


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 2: Run LLM Judge repeated sampling")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore existing logs)")
    parser.add_argument("--config", default="experiment.yaml", help="Config file name")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.smoke:
        logger.info("[SMOKE TEST MODE]")

    if args.no_resume:
        # 기존 로그 삭제
        suffix = "_smoke" if args.smoke else ""
        log_path = RESULTS_LOGS / f"judge_results{suffix}.jsonl"
        if log_path.exists():
            log_path.unlink()
            logger.info(f"Cleared existing log: {log_path}")

    asyncio.run(run_experiment(config, smoke=args.smoke, resume=not args.no_resume))


if __name__ == "__main__":
    main()