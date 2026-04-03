from __future__ import annotations

import pandas as pd

from jobguard.pipeline import (
    build_feature_matrix,
    build_model_suite,
    build_training_frame,
    optimize_threshold,
    select_best_model,
    split_and_balance,
    train_and_evaluate_models,
)


def test_pipeline_smoke_train_and_score():
    df = pd.DataFrame(
        [
            {
                "title": "Senior Data Engineer",
                "company_profile": "We build reliable data systems.",
                "description": "Python data engineer role focused on pipelines and analytics.",
                "requirements": "Python, SQL, data engineering, spark.",
                "benefits": "Health insurance and equity.",
                "telecommuting": 0,
                "has_company_logo": 1,
                "has_questions": 1,
                "fraudulent": 0,
            },
            {
                "title": "Platform Engineer",
                "company_profile": "We build reliable data systems.",
                "description": "Python data engineer role focused on pipelines and analytics.",
                "requirements": "Python, SQL, data engineering, spark.",
                "benefits": "Health insurance and equity.",
                "telecommuting": 0,
                "has_company_logo": 1,
                "has_questions": 1,
                "fraudulent": 0,
            },
            {
                "title": "Machine Learning Engineer",
                "company_profile": "We build reliable data systems.",
                "description": "Python data engineer role focused on pipelines and analytics.",
                "requirements": "Python, SQL, data engineering, spark.",
                "benefits": "Health insurance and equity.",
                "telecommuting": 0,
                "has_company_logo": 1,
                "has_questions": 1,
                "fraudulent": 0,
            },
            {
                "title": "Earn Money Fast",
                "company_profile": "No real company info.",
                "description": "Earn money weekly and pay a small training fee now.",
                "requirements": "No experience needed, act now.",
                "benefits": "Cash bonus and guaranteed income.",
                "telecommuting": 1,
                "has_company_logo": 0,
                "has_questions": 0,
                "fraudulent": 1,
            },
            {
                "title": "Work From Home",
                "company_profile": "No real company info.",
                "description": "Earn money weekly and pay a small training fee now.",
                "requirements": "No experience needed, act now.",
                "benefits": "Cash bonus and guaranteed income.",
                "telecommuting": 1,
                "has_company_logo": 0,
                "has_questions": 0,
                "fraudulent": 1,
            },
            {
                "title": "Data Entry Clerk",
                "company_profile": "No real company info.",
                "description": "Earn money weekly and pay a small training fee now.",
                "requirements": "No experience needed, act now.",
                "benefits": "Cash bonus and guaranteed income.",
                "telecommuting": 1,
                "has_company_logo": 0,
                "has_questions": 0,
                "fraudulent": 1,
            },
        ]
    )

    frame = build_training_frame(df)
    X, y, tfidf, scaler, feature_names = build_feature_matrix(frame)
    assert X.shape[0] == len(df)
    assert len(feature_names) > 0

    X_train, X_test, y_train, y_test, X_train_res, y_train_res = split_and_balance(
        X,
        y,
        test_size=0.33,
        use_smote=False,
    )

    models = build_model_suite(y_train, include_optional=False)
    results, trained_models = train_and_evaluate_models(
        {"Naive Bayes": models["Naive Bayes"]},
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_res=X_train_res,
        y_train_res=y_train_res,
        cv_splits=2,
        n_jobs=1,
        verbose=False,
    )

    best_name, best_model = select_best_model(results, trained_models)
    assert best_name == "Naive Bayes"
    assert best_model is not None
    assert 0.0 <= results[best_name]["f1"] <= 1.0

    threshold_result = optimize_threshold(y_test, results[best_name]["y_prob"])
    assert 0.0 < threshold_result["optimal_threshold"] < 1.0
    assert threshold_result["y_pred_opt"].shape[0] == len(y_test)
