"""
공통 유틸리티: 설정 로드, 로깅, 경로 관리, 체크포인트
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# =============================================================
# Paths
# =============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"
RESULTS_ANALYSIS = PROJECT_ROOT / "results" / "analysis"

# 디렉토리 보장
for d in [DATA_RAW, DATA_PROCESSED, RESULTS_LOGS, RESULTS_ANALYSIS]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================
# Environment
# =============================================================
def load_env() -> None:
    """Load .env file and validate required keys."""
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Copy .env.example → .env and fill in your key.")


# =============================================================
# Config
# =============================================================
def load_config(config_name: str = "experiment.yaml") -> dict[str, Any]:
    """Load YAML config file."""
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================
# Logging
# =============================================================
def setup_logger(name: str = "llm-judge", level: str | None = None) -> logging.Logger:
    """Configure and return a logger."""
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger


# =============================================================
# JSONL I/O (for checkpoint/resume)
# =============================================================
def append_jsonl(filepath: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(filepath: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    if not filepath.exists():
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_completed_keys(log_path: Path) -> set[str]:
    """
    이미 성공적으로 완료된 (question_id, answer_category, trial_number) 조합을 반환.
    API_ERROR, PARSE_ERROR는 제외하여 재시도 대상이 되도록 함.
    checkpoint/resume에 사용.
    """
    records = read_jsonl(log_path)
    return {
        f"{r['question_id']}_{r['answer_category']}_{r['trial_number']}"
        for r in records
        if "question_id" in r and "answer_category" in r and "trial_number" in r
        and r.get("verdict") not in ("API_ERROR", "PARSE_ERROR")
    }