"""Tests for the FeatureEngineer pipeline."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.preprocessor import TransactionPreprocessor
from fraud_detector.feature_engineer import FeatureEngineer


def _make_preprocessed_df(n: int = 50) -> pd.DataFrame:
    """Return a preprocessed DataFrame ready for feature engineering."""
    rng = np.random.default_rng(42)
    raw = pd.DataFrame({
        "transaction_id": [f"TXN-{i:04d}" for i in range(n)],
        "customer_id": rng.choice(["CUST-001", "CUST-002", "CUST-003"], size=n),
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
    pp = TransactionPreprocessor()
    return pp.fit_transform(raw)


# ── Velocity feature calculation ─────────────────────────────────────


def test_velocity_features_created_with_default_windows():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(50)
    result = fe.fit_transform(df)
    for w in [3, 5, 10]:
        assert f"velocity_{w}_mean" in result.columns
        assert f"velocity_{w}_std" in result.columns
        assert f"velocity_{w}_max" in result.columns


def test_velocity_features_created_with_custom_windows():
    fe = FeatureEngineer(velocity_windows=[2, 7])
    df = _make_preprocessed_df(30)
    result = fe.fit_transform(df)
    assert "velocity_2_mean" in result.columns
    assert "velocity_7_mean" in result.columns
    # Default windows should NOT be present
    assert "velocity_3_mean" not in result.columns


def test_velocity_mean_is_positive():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(50)
    result = fe.fit_transform(df)
    # All amounts are positive, so rolling mean should be > 0
    assert (result["velocity_3_mean"] > 0).all()


# ── Amount pattern features ──────────────────────────────────────────


def test_amount_is_round_detection():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(20)
    # Set specific amounts on known transaction_ids so we can find them
    # after the sort that happens inside fit_transform.
    df.loc[df.index[0], "amount"] = 500.0
    df.loc[df.index[1], "amount"] = 200.0
    df.loc[df.index[2], "amount"] = 37.50

    tid_500 = df.loc[df.index[0], "transaction_id"]
    tid_200 = df.loc[df.index[1], "transaction_id"]
    tid_37 = df.loc[df.index[2], "transaction_id"]

    result = fe.fit_transform(df)

    row_500 = result[result["transaction_id"] == tid_500].iloc[0]
    row_200 = result[result["transaction_id"] == tid_200].iloc[0]
    row_37 = result[result["transaction_id"] == tid_37].iloc[0]

    assert row_500["amount_is_round"] == 1
    assert row_200["amount_is_round"] == 1
    # 37.50 % 100 != 0
    assert row_37["amount_is_round"] == 0


def test_amount_above_p95_flag():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(100)
    result = fe.fit_transform(df)
    # At most ~5% should be flagged above p95
    flagged_ratio = result["amount_above_p95"].mean()
    assert flagged_ratio <= 0.15  # generous tolerance for small samples


def test_amount_ratio_to_category_mean_positive():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(50)
    result = fe.fit_transform(df)
    # Ratio should be non-negative for positive amounts
    ratios = result["amount_ratio_to_category_mean"]
    assert (ratios >= 0).all()


# ── Risk score computation ───────────────────────────────────────────


def test_risk_score_heuristic_range():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(50)
    result = fe.fit_transform(df)
    assert (result["risk_score_heuristic"] >= 0).all()
    assert (result["risk_score_heuristic"] <= 1).all()


def test_risk_indicators_are_binary():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(50)
    result = fe.fit_transform(df)
    for col in ["rapid_succession", "high_amount_night"]:
        assert set(result[col].unique()).issubset({0, 1})


def test_all_feature_columns_present():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(50)
    result = fe.fit_transform(df)
    for col in fe.get_feature_columns():
        assert col in result.columns, f"Missing feature column: {col}"


def test_transform_requires_fit_first():
    fe = FeatureEngineer()
    df = _make_preprocessed_df(10)
    with pytest.raises(RuntimeError, match="not fitted"):
        fe.transform(df)
