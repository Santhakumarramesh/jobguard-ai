"""Shared configuration for JobGuard AI."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_THRESHOLD = 0.47
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

JOB_TEXT_COLUMNS = (
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
)

META_FEATURES = (
    "text_length",
    "word_count",
    "has_email",
    "has_url",
    "exclamation_count",
    "caps_ratio",
    "telecommuting",
    "has_company_logo",
    "has_questions",
)

DEFAULT_DATASET_PATHS = (
    DATA_DIR / "jobguard-augmented.csv",
    DATA_DIR / "jobguard-combined.csv",
    PROJECT_ROOT / "jobguard-dataset.csv",
    DATA_DIR / "jobguard-dataset.csv",
    PROJECT_ROOT / "dataset" / "jobguard-dataset.csv",
    PROJECT_ROOT.parent / "jobguard-dataset.csv",
)

ARTIFACT_FILENAMES = {
    "classifier": "classifier.pkl",
    "vectorizer": "tfidf_vectorizer.pkl",
    "scaler": "meta_scaler.pkl",
    "config": "model_config.json",
}
