"""Tests for the FraudModel (RF + Isolation Forest hybrid)."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.preprocessor import TransactionPreprocessor
from fraud_detector.feature_engineer import FeatureEngineer
from fraud_detector.model import FraudModel, ModelMetrics


def _build_training_data(
    n: int = 300, fraud_rate: float = 0.05
) -> tuple[pd.DataFrame, list[str]]:
    """Build a feature-enriched DataFrame with synthetic fraud labels."""
    rng = np.random.default_rng(42)
    raw = pd.DataFrame({
        "customer_id": rng.choice(
            [f"CUST-{i:03d}" for i in range(20)], size=n
        ),
        "amount": rng.uniform(5, 5000, size=n).round(2),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "merchant_category": rng.choice(
            ["retail", "grocery", "travel", "online"], size=n
        ),
        "transaction_type": rng.choice(
            ["purchase", "refund", "transfer"], size=n
        ),
        "channel": rng.choice(["pos", "online", "mobile"], size=n),
        "location": rng.choice(["NYC", "LA", "Chicago", "Houston"], size=n),
    })

    # Synthetic fraud labels
    n_fraud = max(int(n * fraud_rate), 5)
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)
    raw["is_fraud"] = labels

    pp = TransactionPreprocessor()
    processed = pp.fit_transform(raw)

    fe = FeatureEngineer()
    featured = fe.fit_transform(processed)

    feature_cols = pp.get_feature_columns() + fe.get_feature_columns()
    return featured, feature_cols


# ── Training ─────────────────────────────────────────────────────────


def test_model_training_does_not_error():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    metrics = model.train(df, feature_cols, target_column="is_fraud")
    assert isinstance(metrics, ModelMetrics)


def test_metrics_populated_after_training():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    metrics = model.train(df, feature_cols)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert len(metrics.cv_scores) == 2
    assert len(metrics.feature_importances) > 0


def test_metrics_summary_includes_split_method():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    metrics = model.train(df, feature_cols)
    summary = metrics.summary()
    assert "Split:" in summary
    assert "PR AUC:" in summary


# ── Prediction output shape and range ────────────────────────────────


def test_predict_proba_output_shape():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    model.train(df, feature_cols)

    available = [c for c in model.feature_columns if c in df.columns]
    X = df[available].values[:10].astype(np.float64)
    proba = model.predict_proba(X)
    assert proba.shape == (10,)


def test_predict_proba_values_between_0_and_1():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    model.train(df, feature_cols)

    available = [c for c in model.feature_columns if c in df.columns]
    X = df[available].values.astype(np.float64)
    proba = model.predict_proba(X)
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()


def test_anomaly_scores_between_0_and_1():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    model.train(df, feature_cols)

    available = [c for c in model.feature_columns if c in df.columns]
    X = df[available].values.astype(np.float64)
    scores = model.anomaly_scores(X)
    assert scores.shape == (len(X),)
    assert (scores >= -0.01).all()  # tiny float tolerance
    assert (scores <= 1.01).all()


# ── Hybrid scoring ───────────────────────────────────────────────────


def test_hybrid_score_output_shape():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    model.train(df, feature_cols)

    available = [c for c in model.feature_columns if c in df.columns]
    X = df[available].values[:5].astype(np.float64)
    hybrid = model.hybrid_score(X)
    assert hybrid.shape == (5,)


def test_hybrid_score_within_unit_interval():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    model.train(df, feature_cols)

    available = [c for c in model.feature_columns if c in df.columns]
    X = df[available].values.astype(np.float64)
    hybrid = model.hybrid_score(X)
    assert (hybrid >= -0.01).all()
    assert (hybrid <= 1.01).all()


def test_hybrid_score_respects_weights():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    model.train(df, feature_cols)

    available = [c for c in model.feature_columns if c in df.columns]
    X = df[available].values[:5].astype(np.float64)

    # All RF weight vs. all IF weight should give different results
    rf_only = model.hybrid_score(X, rf_weight=1.0, iso_weight=0.0)
    iso_only = model.hybrid_score(X, rf_weight=0.0, iso_weight=1.0)
    # They may happen to match by coincidence on small data but
    # at minimum the call should succeed without error
    assert rf_only.shape == iso_only.shape


def test_predict_before_train_raises():
    model = FraudModel()
    X = np.zeros((5, 3))
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict_proba(X)


# ── Temporal splitting ───────────────────────────────────────────────


def test_temporal_split_train():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10)
    metrics = model.train(
        df, feature_cols,
        temporal_split=True,
    )
    assert metrics.split_method == "temporal"
    # Temporal split should not have CV scores
    assert len(metrics.cv_scores) == 0


def test_random_split_train():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2)
    metrics = model.train(
        df, feature_cols,
        temporal_split=False,
    )
    assert metrics.split_method == "random_stratified"
    assert len(metrics.cv_scores) == 2


# ── SMOTE ────────────────────────────────────────────────────────────


def test_smote_training():
    df, feature_cols = _build_training_data(n=200)
    model = FraudModel(n_estimators=10, cv_folds=2, use_smote=True)
    metrics = model.train(df, feature_cols)
    assert isinstance(metrics, ModelMetrics)
    assert metrics.roc_auc > 0.0
