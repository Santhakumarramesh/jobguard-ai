"""Text normalization and feature extraction helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from .config import JOB_TEXT_COLUMNS

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return DEFAULT_STOPWORDS


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer

        return WordNetLemmatizer()
    except Exception:
        return None


def compose_job_text(
    title: str | None = None,
    company_profile: str | None = None,
    description: str | None = None,
    requirements: str | None = None,
    benefits: str | None = None,
) -> str:
    """Combine standard job posting fields into a single text blob."""

    parts = [
        title,
        company_profile,
        description,
        requirements,
        benefits,
    ]
    cleaned = [str(part).strip() for part in parts if part and str(part).strip()]
    return " ".join(cleaned).strip()


def compose_job_text_from_mapping(mapping: dict, columns: Iterable[str] = JOB_TEXT_COLUMNS) -> str:
    """Compose a single text string from a mapping or dataframe row."""

    company_profile = mapping.get("company_profile") or mapping.get("company")
    values = [
        mapping.get("title"),
        company_profile,
        mapping.get("description"),
        mapping.get("requirements"),
        mapping.get("benefits"),
    ]
    return compose_job_text(*values)


def preprocess_text(text: str | None) -> str:
    """Normalize raw text for vectorization."""

    if text is None:
        return "empty"

    text = str(text)
    if not text.strip():
        return "empty"

    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "empty"

    try:
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    stopwords = _stopwords()
    lemmatizer = _lemmatizer()
    cleaned = []
    for token in tokens:
        token = token.strip()
        if len(token) <= 1 or token in stopwords:
            continue
        if lemmatizer is not None:
            try:
                token = lemmatizer.lemmatize(token)
            except Exception:
                pass
        cleaned.append(token)

    return " ".join(cleaned) if cleaned else "empty"


def extract_meta_features_from_text(text: str | None) -> dict[str, float]:
    """Extract the numeric meta-features used by the notebook pipeline."""

    text = "" if text is None else str(text)
    text_len = len(text)
    return {
        "text_length": float(text_len),
        "word_count": float(len(text.split())),
        "has_email": float(bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))),
        "has_url": float(bool(re.search(r"https?://|www\.", text))),
        "exclamation_count": float(text.count("!")),
        "caps_ratio": float(sum(1 for c in text if c.isupper()) / max(text_len, 1)),
        "telecommuting": 0.0,
        "has_company_logo": 0.0,
        "has_questions": 0.0,
    }
