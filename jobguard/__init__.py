"""JobGuard AI reusable package."""

from .config import (
    DEFAULT_DATASET_PATHS,
    DEFAULT_THRESHOLD,
    META_FEATURES,
    MODEL_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from .detector import JobFraudDetector, load_detector
from .heuristics import RuleBasedJobFraudDetector, score_job_text
from .api import create_app
from .text import compose_job_text, extract_meta_features_from_text, preprocess_text

__all__ = [
    "DEFAULT_DATASET_PATHS",
    "DEFAULT_THRESHOLD",
    "META_FEATURES",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "TFIDF_MAX_FEATURES",
    "TFIDF_NGRAM_RANGE",
    "JobFraudDetector",
    "RuleBasedJobFraudDetector",
    "load_detector",
    "score_job_text",
    "create_app",
    "compose_job_text",
    "extract_meta_features_from_text",
    "preprocess_text",
]
