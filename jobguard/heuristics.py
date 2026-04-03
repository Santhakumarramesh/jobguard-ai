"""Rule-based fallback detector used when model artifacts are unavailable."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_THRESHOLD, META_FEATURES
from .text import compose_job_text, compose_job_text_from_mapping, extract_meta_features_from_text

FRAUD_PATTERNS = [
    re.compile(r"\b(ssn|social security|bank account|bank details|wire transfer|western union|moneygram)\b", re.I),
    re.compile(r"\b(registration fee|application fee|processing fee|upfront fee)\b", re.I),
    re.compile(r"\b(send (your|us) (ssn|bank|account|details))\b", re.I),
    re.compile(r"\$(\d{1,3},?\d{3})\s*(per\s*)?(day|week|month)\s*(guaranteed|guarantee)", re.I),
    re.compile(r"\b(earn\s+\$|make\s+\$|guaranteed\s+\$)\d+", re.I),
    re.compile(r"\b(no experience needed|no experience required|no skills needed)\b", re.I),
    re.compile(r"\b(act now|apply now|limited time|urgent|immediately)\b", re.I),
    re.compile(r"\b(work from home|work at home|wfh)\s*[-–—]\s*(earn|make|get)\s+\$", re.I),
    re.compile(r"\b(pay (a |the )?fee|pay (for )?training)\b", re.I),
    re.compile(r"!!!|\.{3,}|\b(cash|money)\s+(reward|bonus)\b", re.I),
]

LEGIT_PATTERNS = [
    re.compile(r"\b(apply at|careers\.|linkedin|indeed|glassdoor)\b", re.I),
    re.compile(r"\b(equity|benefits|401k|health insurance)\b", re.I),
    re.compile(r"\b(years?\s+experience|experience required)\b", re.I),
    re.compile(r"\b(agile|scrum|sprint|roadmap)\b", re.I),
    re.compile(r"\b(full-?time|part-?time|contract|remote)\b", re.I),
]


def _risk_bucket(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    if prob >= 0.50:
        return "MEDIUM"
    if prob >= 0.20:
        return "LOW"
    return "VERY LOW"


def score_job_text(text: str | None, has_company_logo: bool = False, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Score a job posting with a deterministic rule-based heuristic."""

    text = "" if text is None else str(text)
    normalized = text.lower().strip()
    if not normalized:
        return {
            "prediction": 0,
            "label": "LEGITIMATE",
            "fraud_probability": 25.0,
            "confidence": 75.0,
            "risk_level": "VERY LOW",
            "threshold_used": threshold,
            "signals": [],
            "source": "heuristic",
        }

    meta = extract_meta_features_from_text(text)
    word_count = int(meta["word_count"])
    text_len = int(meta["text_length"])

    score = 0.35
    signals: list[dict[str, str]] = []

    if has_company_logo:
        score -= 0.12
    else:
        signals.append({"type": "r", "text": "No company logo"})

    if word_count < 100:
        score += 0.15
        signals.append({"type": "r", "text": f"Very short description ({word_count} words)"})
    elif word_count > 350:
        score -= 0.08

    if text_len < 500:
        score += 0.08
        signals.append({"type": "r", "text": f"Short text ({text_len} chars)"})

    for idx, pattern in enumerate(FRAUD_PATTERNS):
        if pattern.search(text):
            score += 0.08 + idx * 0.01
            signals.append({"type": "r", "text": "Fraud phrase detected"})

    for pattern in LEGIT_PATTERNS:
        if pattern.search(text):
            score -= 0.05
            signals.append({"type": "g", "text": "Legit signal detected"})

    caps_ratio = float(meta["caps_ratio"])
    if caps_ratio > 0.15:
        score += 0.1
        signals.append({"type": "r", "text": "Excessive caps"})

    exclamation_count = int(meta["exclamation_count"])
    if exclamation_count > 3:
        score += 0.08
        signals.append({"type": "r", "text": f"{exclamation_count} exclamation marks"})

    fraud_prob = max(0.05, min(0.95, score))
    prediction = int(fraud_prob >= threshold)
    label = "FRAUDULENT" if prediction else "LEGITIMATE"
    confidence = fraud_prob if prediction else 1 - fraud_prob

    return {
        "prediction": prediction,
        "label": label,
        "fraud_probability": round(fraud_prob * 100, 2),
        "confidence": round(confidence * 100, 2),
        "risk_level": _risk_bucket(fraud_prob),
        "threshold_used": threshold,
        "signals": signals,
        "source": "heuristic",
    }


@dataclass
class RuleBasedJobFraudDetector:
    """Simple detector used as a fallback when artifacts are missing."""

    threshold: float = DEFAULT_THRESHOLD
    source: str = "heuristic"

    def predict(self, job_text: str) -> dict:
        return score_job_text(job_text, threshold=self.threshold)

    def predict_batch(self, text_list) -> pd.DataFrame:
        rows = []
        for item in text_list:
            result = self.predict(item)
            rows.append({"text": str(item), **result})
        return pd.DataFrame(rows)

    def predict_dataframe(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df_copy = input_df.copy()
        for col in ("title", "company_profile", "description", "requirements", "benefits"):
            if col not in df_copy.columns:
                df_copy[col] = ""

        if "has_company_logo" not in df_copy.columns:
            df_copy["has_company_logo"] = 0

        results = []
        for _, row in df_copy.iterrows():
            text = compose_job_text_from_mapping(row.to_dict())
            results.append(
                {
                    **score_job_text(
                        text,
                        has_company_logo=bool(row.get("has_company_logo", 0)),
                        threshold=self.threshold,
                    ),
                    "text": text,
                }
            )

        result_df = pd.DataFrame(results)
        return pd.concat([df_copy.reset_index(drop=True), result_df], axis=1)
