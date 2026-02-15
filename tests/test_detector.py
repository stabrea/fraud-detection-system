"""Tests for the FraudDetector real-time scoring engine."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.preprocessor import TransactionPreprocessor
from fraud_detector.feature_engineer import FeatureEngineer
from fraud_detector.model import FraudModel
from fraud_detector.detector import FraudDetector, RiskLevel, ScoringResult


def _build_detector() -> tuple[FraudDetector, TransactionPreprocessor, FeatureEngineer]:
    """Train a small model and wire up a FraudDetector."""
    rng = np.random.default_rng(42)
    n = 300

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
    n_fraud = max(int(n * 0.05), 5)
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)
    raw["is_fraud"] = labels

    pp = TransactionPreprocessor()
    processed = pp.fit_transform(raw)

    fe = FeatureEngineer()
    featured = fe.fit_transform(processed)

    feature_cols = pp.get_feature_columns() + fe.get_feature_columns()
    model = FraudModel(n_estimators=10, cv_folds=2)
    model.train(featured, feature_cols)

    detector = FraudDetector(model, pp, fe)
    return detector, pp, fe


# ── Risk level classification thresholds ─────────────────────────────


def test_classify_risk_low():
    detector, _, _ = _build_detector()
    # Access the private method for direct threshold testing
    assert detector._classify_risk(0.1) == RiskLevel.LOW
    assert detector._classify_risk(0.0) == RiskLevel.LOW
    assert detector._classify_risk(0.29) == RiskLevel.LOW


def test_classify_risk_medium():
    detector, _, _ = _build_detector()
    assert detector._classify_risk(0.3) == RiskLevel.MEDIUM
    assert detector._classify_risk(0.49) == RiskLevel.MEDIUM


def test_classify_risk_high():
    detector, _, _ = _build_detector()
    assert detector._classify_risk(0.5) == RiskLevel.HIGH
    assert detector._classify_risk(0.69) == RiskLevel.HIGH


def test_classify_risk_critical():
    detector, _, _ = _build_detector()
    assert detector._classify_risk(0.7) == RiskLevel.CRITICAL
    assert detector._classify_risk(0.99) == RiskLevel.CRITICAL
    assert detector._classify_risk(1.0) == RiskLevel.CRITICAL


def test_custom_thresholds():
    detector, pp, fe = _build_detector()
    custom = {"low": 0.2, "medium": 0.4, "high": 0.6}
    detector._thresholds = custom
    assert detector._classify_risk(0.1) == RiskLevel.LOW
    assert detector._classify_risk(0.3) == RiskLevel.MEDIUM
    assert detector._classify_risk(0.5) == RiskLevel.HIGH
    assert detector._classify_risk(0.6) == RiskLevel.CRITICAL


# ── Scoring returns expected structure ───────────────────────────────


def test_score_transaction_returns_scoring_result():
    detector, _, _ = _build_detector()
    txn = {
        "amount": 250.0,
        "timestamp": "2024-06-15T14:30:00",
        "merchant_category": "retail",
        "transaction_type": "purchase",
        "channel": "online",
        "location": "NYC",
        "customer_id": "CUST-001",
        "transaction_id": "TEST-001",
    }
    result = detector.score_transaction(txn)
    assert isinstance(result, ScoringResult)
    assert result.transaction_id == "TEST-001"
    assert isinstance(result.risk_level, RiskLevel)
    assert isinstance(result.contributing_factors, list)
    assert len(result.contributing_factors) > 0


def test_scoring_result_to_dict():
    detector, _, _ = _build_detector()
    txn = {
        "amount": 250.0,
        "timestamp": "2024-06-15T14:30:00",
        "merchant_category": "retail",
        "transaction_type": "purchase",
        "channel": "online",
        "location": "NYC",
        "customer_id": "CUST-001",
        "transaction_id": "TEST-002",
    }
    result = detector.score_transaction(txn)
    d = result.to_dict()
    assert "transaction_id" in d
    assert "fraud_probability" in d
    assert "anomaly_score" in d
    assert "hybrid_score" in d
    assert "risk_level" in d
    assert "contributing_factors" in d
    assert isinstance(d["risk_level"], str)


def test_score_probabilities_in_valid_range():
    detector, _, _ = _build_detector()
    txn = {
        "amount": 1500.0,
        "timestamp": "2024-06-15T03:00:00",
        "merchant_category": "travel",
        "transaction_type": "purchase",
        "channel": "online",
        "location": "Houston",
        "customer_id": "CUST-005",
    }
    result = detector.score_transaction(txn)
    assert 0.0 <= result.fraud_probability <= 1.0
    assert 0.0 <= result.anomaly_score <= 1.01
    assert 0.0 <= result.hybrid_score <= 1.01


def test_score_batch_returns_list():
    detector, _, _ = _build_detector()
    df = pd.DataFrame([
        {
            "amount": 100.0,
            "timestamp": "2024-06-15T10:00:00",
            "merchant_category": "grocery",
            "transaction_type": "purchase",
            "channel": "pos",
            "location": "LA",
            "customer_id": "CUST-001",
            "transaction_id": "BATCH-001",
        },
        {
            "amount": 3000.0,
            "timestamp": "2024-06-15T02:00:00",
            "merchant_category": "online",
            "transaction_type": "transfer",
            "channel": "online",
            "location": "Chicago",
            "customer_id": "CUST-002",
            "transaction_id": "BATCH-002",
        },
    ])
    results = detector.score_batch(df)
    assert len(results) == 2
    assert all(isinstance(r, ScoringResult) for r in results)


def test_statistics_after_scoring():
    detector, _, _ = _build_detector()
    txn = {
        "amount": 250.0,
        "timestamp": "2024-06-15T14:30:00",
        "merchant_category": "retail",
        "transaction_type": "purchase",
        "channel": "online",
        "location": "NYC",
        "customer_id": "CUST-001",
    }
    detector.score_transaction(txn)
    stats = detector.get_statistics()
    assert stats["total"] == 1
    assert "mean_score" in stats
    assert "max_score" in stats
    assert "risk_distribution" in stats
