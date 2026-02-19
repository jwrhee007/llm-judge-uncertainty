"""
스모크 테스트: 환경 세팅이 올바른지 검증
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_imports():
    """핵심 패키지 import 확인"""
    import openai
    import datasets
    import yaml
    import pandas
    import numpy
    import spacy
    import tqdm
    print("✓ All core packages imported successfully")


def test_config_load():
    """설정 파일 로드 확인"""
    from src.utils import load_config
    config = load_config()
    assert config["dataset"]["name"] == "triviaqa"
    assert config["judge"]["model"] == "gpt-4o-mini"
    assert config["judge"]["n_trials"] == 30
    print("✓ Config loaded and validated")


def test_paths():
    """프로젝트 경로 존재 확인"""
    from src.utils import DATA_RAW, DATA_PROCESSED, RESULTS_LOGS
    assert DATA_RAW.exists()
    assert DATA_PROCESSED.exists()
    assert RESULTS_LOGS.exists()
    print("✓ All directories exist")


def test_api_key():
    """API 키 설정 확인"""
    from src.utils import load_env
    load_env()
    key = os.getenv("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-your"):
        print("✓ OPENAI_API_KEY is set")
    else:
        print("⚠ OPENAI_API_KEY not set (skip API tests)")


def test_spacy_model():
    """spaCy 영어 모델 로드 확인"""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("William Shakespeare was born in 1564 in Stratford-upon-Avon.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        assert len(entities) > 0
        print(f"✓ spaCy NER working: {entities}")
    except OSError:
        print("⚠ spaCy model not found. Run: python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    test_imports()
    test_config_load()
    test_paths()
    test_api_key()
    test_spacy_model()
    print("\n=== All smoke tests passed ===")
