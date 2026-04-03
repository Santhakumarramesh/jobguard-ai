"""Training and evaluation helpers for the JobGuard notebook."""

from __future__ import annotations

import importlib.util
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler

from .config import (
    ARTIFACT_FILENAMES,
    DEFAULT_DATASET_PATHS,
    DEFAULT_THRESHOLD,
    META_FEATURES,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    JOB_TEXT_COLUMNS,
)
from .text import compose_job_text, extract_meta_features_from_text, preprocess_text

SMOTE_AVAILABLE = True  # Notebook compatibility flag for the balancing branch.
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
WC_AVAILABLE = importlib.util.find_spec("wordcloud") is not None

if WC_AVAILABLE:  # pragma: no cover - optional dependency
    from wordcloud import WordCloud
else:  # pragma: no cover - optional dependency
    WordCloud = None


def savefig(name: str) -> Path:
    """Save the current matplotlib figure into the shared outputs directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=100, bbox_inches="tight")
    return path


def load_dataset(search_paths: Iterable[Path] | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the primary dataset from the first available search path."""

    paths = [Path(p) for p in (search_paths or DEFAULT_DATASET_PATHS)]
    for path in paths:
        if path.exists():
            return pd.read_csv(path), path

    raise FileNotFoundError(
        "Dataset not found. Download from Kaggle and save as 'jobguard-dataset.csv':\n"
        "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n"
        "Run: python3 scripts/download_datasets.py && python3 scripts/augment_fraud_data.py"
    )


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str], fill_value="") -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col not in frame.columns:
            frame[col] = fill_value
    return frame


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create the text and meta-feature columns used by the notebook."""

    frame = _ensure_columns(df, JOB_TEXT_COLUMNS, "")
    for col in ("telecommuting", "has_company_logo", "has_questions"):
        if col not in frame.columns:
            frame[col] = 0

    frame["text_data"] = frame.apply(
        lambda row: compose_job_text(
            row.get("title"),
            row.get("company_profile"),
            row.get("description"),
            row.get("requirements"),
            row.get("benefits"),
        ),
        axis=1,
    ).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    frame["clean_text"] = frame["text_data"].map(preprocess_text)

    meta_df = frame["text_data"].map(extract_meta_features_from_text).apply(pd.Series)
    for feature in META_FEATURES:
        if feature not in meta_df.columns:
            meta_df[feature] = 0.0

    frame = frame.reset_index(drop=True)
    for feature in META_FEATURES:
        frame[feature] = meta_df[feature].reset_index(drop=True)

    for feature in {"telecommuting", "has_company_logo", "has_questions"}:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0).astype(int)

    return frame


def build_feature_matrix(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer | None = None,
    scaler: MaxAbsScaler | None = None,
    *,
    fit: bool = True,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
) -> tuple:
    """Vectorize cleaned text and append the meta-features."""

    frame = df.copy()
    if "clean_text" not in frame.columns or not set(META_FEATURES).issubset(frame.columns):
        frame = build_training_frame(frame)

    vectorizer = vectorizer or TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=3,
        max_df=0.90,
        strip_accents="unicode",
        analyzer="word",
    )

    if fit or not hasattr(vectorizer, "vocabulary_"):
        X_tfidf = vectorizer.fit_transform(frame["clean_text"])
    else:
        X_tfidf = vectorizer.transform(frame["clean_text"])

    scaler = scaler or MaxAbsScaler()
    meta_frame = frame[list(META_FEATURES)].fillna(0).astype(float)
    if fit or not hasattr(scaler, "n_features_in_"):
        X_meta = scaler.fit_transform(meta_frame)
    else:
        X_meta = scaler.transform(meta_frame)

    X = hstack([X_tfidf, csr_matrix(X_meta)])
    y = frame["fraudulent"].astype(int).to_numpy() if "fraudulent" in frame.columns else None
    feature_names = vectorizer.get_feature_names_out()
    return X, y, vectorizer, scaler, feature_names


def split_and_balance(
    X,
    y,
    *,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    use_smote: bool = True,
    k_neighbors: int = 5,
):
    """Split the dataset and optionally balance the training set with oversampling."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_res, y_train_res = X_train, y_train
    if use_smote:
        y_train_arr = np.asarray(y_train).astype(int)
        class_counts = np.bincount(y_train_arr)
        if len(class_counts) >= 2 and class_counts.min() > 0 and class_counts.min() < class_counts.max():
            rng = np.random.default_rng(random_state)
            majority_class = int(np.argmax(class_counts))
            minority_class = int(np.argmin(class_counts))
            majority_count = int(class_counts[majority_class])
            minority_idx = np.where(y_train_arr == minority_class)[0]
            extra_idx = rng.choice(
                minority_idx,
                size=majority_count - int(class_counts[minority_class]),
                replace=True,
            )
            resample_idx = np.concatenate([np.arange(len(y_train_arr)), extra_idx])
            X_train_res = X_train[resample_idx]
            y_train_res = y_train_arr[resample_idx]

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res


def build_model_suite(y_train, *, random_state: int = RANDOM_STATE, include_optional: bool = True) -> dict:
    """Construct the candidate models used by the notebook."""

    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if include_optional:
        if XGBOOST_AVAILABLE:
            try:
                from xgboost import XGBClassifier
            except Exception:  # pragma: no cover - optional dependency
                XGBClassifier = None

        if XGBOOST_AVAILABLE and XGBClassifier is not None:
            y_arr = np.asarray(y_train).astype(int)
            scale_pos_weight = int((y_arr == 0).sum() / max((y_arr == 1).sum(), 1))
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            models["SGD Logistic"] = SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=1000,
                tol=1e-3,
                class_weight="balanced",
                random_state=random_state,
            )

    return models


def _predict_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        return expit(np.asarray(model.decision_function(X)))
    return np.asarray(model.predict(X), dtype=float)


def _cv_strategy(y, max_splits: int = 5, random_state: int = RANDOM_STATE):
    y_arr = np.asarray(y).astype(int)
    class_counts = np.bincount(y_arr)
    if len(class_counts) < 2 or class_counts.min() < 2:
        return None
    splits = min(max_splits, int(class_counts.min()))
    if splits < 2:
        return None
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


def train_and_evaluate_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    X_train_res=None,
    y_train_res=None,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    verbose: bool = True,
):
    """Train each candidate model and collect evaluation metrics."""

    X_train_res = X_train if X_train_res is None else X_train_res
    y_train_res = y_train if y_train_res is None else y_train_res

    cv = _cv_strategy(y_train, max_splits=cv_splits, random_state=random_state)
    trained_models = {}
    results = {}

    if verbose:
        print(f"{'Model':<26} {'CV F1 Mean':>12} {'CV F1 Std':>10} {'Holdout F1':>12} {'Time (s)':>9}")
        print("-" * 72)

    for name, model in models.items():
        start = pd.Timestamp.utcnow()

        if cv is not None:
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=n_jobs)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Cross-validation failed for {name}: {exc}")
                cv_scores = np.array([np.nan])
        else:
            cv_scores = np.array([np.nan])

        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        y_prob = _predict_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        elapsed = (pd.Timestamp.utcnow() - start).total_seconds()

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": auc(fpr, tpr),
            "avg_prec": average_precision_score(y_test, y_prob),
            "fpr": fpr,
            "tpr": tpr,
            "prec_curve": prec_curve,
            "rec_curve": rec_curve,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cv_f1_mean": float(np.nanmean(cv_scores)),
            "cv_f1_std": float(np.nanstd(cv_scores)),
            "train_sec": elapsed,
        }
        trained_models[name] = model

        if verbose:
            print(
                f"{name:<26} {results[name]['cv_f1_mean']:>12.4f} "
                f"{results[name]['cv_f1_std']:>10.4f} {results[name]['f1']:>12.4f} {elapsed:>9.1f}"
            )

    if verbose:
        print("-" * 72)
        print("Training complete.")

    return results, trained_models


def select_best_model(results: dict, trained_models: dict, metric: str = "f1") -> tuple[str, object]:
    """Return the best model by metric."""

    best_name = max(results, key=lambda name: results[name][metric])
    return best_name, trained_models[best_name]


def optimize_threshold(y_true, y_prob, *, num_thresholds: int = 200) -> dict:
    """Find the threshold that maximizes F1 on the holdout set."""

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    f1_scores = [
        f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    optimal_threshold = float(thresholds[int(np.argmax(f1_scores))])
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    precision = precision_score(y_true, y_pred_opt, zero_division=0)
    recall = recall_score(y_true, y_pred_opt, zero_division=0)

    return {
        "thresholds": thresholds,
        "f1_scores": np.asarray(f1_scores),
        "optimal_threshold": optimal_threshold,
        "optimal_f1": float(max(f1_scores)),
        "y_pred_opt": y_pred_opt,
        "precision": float(precision),
        "recall": float(recall),
    }


def save_model_artifacts(
    model,
    vectorizer,
    scaler,
    meta_features: Iterable[str] = META_FEATURES,
    *,
    output_dir: Path = MODEL_DIR,
    config: dict | None = None,
) -> dict[str, Path]:
    """Persist the trained model bundle to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        ARTIFACT_FILENAMES["classifier"]: model,
        ARTIFACT_FILENAMES["vectorizer"]: vectorizer,
        ARTIFACT_FILENAMES["scaler"]: scaler,
    }

    saved = {}
    for filename, obj in artifacts.items():
        path = output_dir / filename
        joblib.dump(obj, path)
        saved[filename] = path

    model_config = {
        "optimal_threshold": DEFAULT_THRESHOLD,
        "meta_features": list(meta_features),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
    }
    if config:
        model_config.update(config)

    config_path = output_dir / ARTIFACT_FILENAMES["config"]
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)
    saved[ARTIFACT_FILENAMES["config"]] = config_path

    return saved
