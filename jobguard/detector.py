"""Model-backed detector and artifact loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from .config import ARTIFACT_FILENAMES, DEFAULT_THRESHOLD, META_FEATURES, MODEL_DIR
from .heuristics import RuleBasedJobFraudDetector
from .text import compose_job_text, extract_meta_features_from_text, preprocess_text


class JobFraudDetector:
    """Production-ready wrapper around the trained job fraud model."""

    def __init__(self, model, vectorizer, scaler, meta_feature_names, threshold: float = DEFAULT_THRESHOLD, source: str = "artifacts"):
        self.model = model
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.meta_names = list(meta_feature_names)
        self.threshold = float(threshold)
        self.source = source

    @staticmethod
    def _risk_bucket(prob: float) -> str:
        if prob >= 0.80:
            return "HIGH"
        if prob >= 0.50:
            return "MEDIUM"
        if prob >= 0.20:
            return "LOW"
        return "VERY LOW"

    def _extract_meta_from_text(self, text: str | None) -> np.ndarray:
        meta = extract_meta_features_from_text(text)
        return np.array([[meta.get(name, 0.0) for name in self.meta_names]], dtype=float)

    def _score(self, raw_text: str) -> float:
        clean = preprocess_text(raw_text)
        X_text = self.vectorizer.transform([clean])
        X_meta = self.scaler.transform(self._extract_meta_from_text(raw_text))
        X_in = hstack([X_text, csr_matrix(X_meta)])

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(X_in)[0, 1])
        if hasattr(self.model, "decision_function"):
            from scipy.special import expit

            return float(expit(self.model.decision_function(X_in))[0])
        return float(self.model.predict(X_in)[0])

    def predict(self, job_text: str) -> dict:
        if not isinstance(job_text, str) or not job_text.strip():
            return {"error": "Input text is empty or invalid."}

        prob = self._score(job_text)
        pred = int(prob >= self.threshold)
        label = "FRAUDULENT" if pred else "LEGITIMATE"
        confidence = prob if pred else 1 - prob

        return {
            "prediction": pred,
            "label": label,
            "fraud_probability": round(prob * 100, 2),
            "confidence": round(confidence * 100, 2),
            "risk_level": self._risk_bucket(prob),
            "threshold_used": self.threshold,
            "source": self.source,
        }

    def predict_batch(self, text_list) -> pd.DataFrame:
        if not isinstance(text_list, (list, tuple)):
            raise TypeError("text_list must be a list or tuple of strings")

        rows = []
        for txt in text_list:
            out = self.predict(str(txt))
            rows.append({"text": str(txt), **out})
        return pd.DataFrame(rows)

    def predict_dataframe(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("input_df must be a pandas DataFrame")

        df_copy = input_df.copy()
        text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
        for col in text_cols:
            if col not in df_copy.columns:
                df_copy[col] = ""

        if "company_profile" not in input_df.columns and "company" in df_copy.columns:
            df_copy["company_profile"] = df_copy["company"].fillna("").astype(str)

        if "has_company_logo" not in df_copy.columns:
            df_copy["has_company_logo"] = 0

        def _series_or_default(column: str, default: int = 0) -> pd.Series:
            if column in df_copy.columns:
                return pd.to_numeric(df_copy[column], errors="coerce").fillna(default).astype(int)
            return pd.Series(np.full(len(df_copy), default), index=df_copy.index, dtype=int)

        combined = (
            df_copy["title"].fillna("").astype(str)
            + " "
            + df_copy["company_profile"].fillna("").astype(str)
            + " "
            + df_copy["description"].fillna("").astype(str)
            + " "
            + df_copy["requirements"].fillna("").astype(str)
            + " "
            + df_copy["benefits"].fillna("").astype(str)
        ).str.replace(r"\s+", " ", regex=True).str.strip()

        cleaned = combined.apply(preprocess_text)
        X_text = self.vectorizer.transform(cleaned)

        meta_df = pd.DataFrame(
            {
                "text_length": combined.str.len(),
                "word_count": combined.str.split().str.len().fillna(0).astype(int),
                "has_email": combined.str.contains(r"[\w.+-]+@[\w-]+\.[\w.-]+", regex=True).astype(int),
                "has_url": combined.str.contains(r"https?://|www\.", regex=True).astype(int),
                "exclamation_count": combined.str.count(r"!").fillna(0).astype(int),
                "caps_ratio": combined.apply(lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)),
                "telecommuting": _series_or_default("telecommuting", 0),
                "has_company_logo": _series_or_default("has_company_logo", 0),
                "has_questions": _series_or_default("has_questions", 0),
            }
        )

        for col in self.meta_names:
            if col not in meta_df.columns:
                meta_df[col] = 0

        X_meta = self.scaler.transform(meta_df[self.meta_names].fillna(0).astype(float))
        X_in = hstack([X_text, csr_matrix(X_meta)])

        probs = self.model.predict_proba(X_in)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        out = df_copy.copy()
        out["fraud_probability"] = np.round(probs * 100, 2)
        out["prediction"] = preds
        out["label"] = np.where(preds == 1, "FRAUDULENT", "LEGITIMATE")
        out["confidence"] = np.round(np.where(preds == 1, probs, 1 - probs) * 100, 2)
        out["risk_level"] = [self._risk_bucket(p) for p in probs]
        out["threshold_used"] = self.threshold
        out["source"] = self.source
        return out

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path = MODEL_DIR) -> "JobFraudDetector":
        artifact_dir = Path(artifact_dir)
        config_path = artifact_dir / ARTIFACT_FILENAMES["config"]
        model_path = artifact_dir / ARTIFACT_FILENAMES["classifier"]
        vectorizer_path = artifact_dir / ARTIFACT_FILENAMES["vectorizer"]
        scaler_path = artifact_dir / ARTIFACT_FILENAMES["scaler"]

        if not config_path.exists() or not model_path.exists() or not vectorizer_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model artifacts not found in {artifact_dir}")

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        scaler = joblib.load(scaler_path)
        threshold = float(config.get("optimal_threshold", DEFAULT_THRESHOLD))
        meta_features = config.get("meta_features", list(META_FEATURES))
        return cls(model, vectorizer, scaler, meta_features, threshold=threshold, source="artifacts")


def load_detector(artifact_dir: str | Path = MODEL_DIR, allow_fallback: bool = True):
    """Load the trained detector or fall back to the rule-based detector."""

    try:
        return JobFraudDetector.from_artifacts(artifact_dir)
    except Exception as exc:
        if not allow_fallback:
            raise

        detector = RuleBasedJobFraudDetector()
        detector.source = f"heuristic fallback ({exc})"
        return detector
