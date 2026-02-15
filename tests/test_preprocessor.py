"""Tests for the TransactionPreprocessor pipeline."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.preprocessor import TransactionPreprocessor


def _make_raw_df(n: int = 50) -> pd.DataFrame:
    """Create a small but realistic raw transaction DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
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


# ── Missing value handling ───────────────────────────────────────────


def test_missing_amounts_filled_with_median():
    pp = TransactionPreprocessor()
    df = _make_raw_df(20)
    df.loc[0, "amount"] = np.nan
    df.loc[3, "amount"] = np.nan
    result = pp.fit_transform(df)
    assert result["amount"].isna().sum() == 0


def test_missing_categoricals_filled_with_unknown():
    pp = TransactionPreprocessor()
    df = _make_raw_df(10)
    df.loc[0, "merchant_category"] = None
    df.loc[1, "transaction_type"] = None
    df.loc[2, "channel"] = None
    df.loc[3, "location"] = None
    result = pp.fit_transform(df)
    assert result.loc[0, "merchant_category"] == "unknown"
    assert result.loc[1, "transaction_type"] == "unknown"
    assert result.loc[2, "channel"] == "unknown"
    assert result.loc[3, "location"] == "unknown"


def test_no_missing_values_unchanged():
    pp = TransactionPreprocessor()
    df = _make_raw_df(10)
    result = pp.fit_transform(df)
    assert result["amount"].isna().sum() == 0
    assert result["merchant_category"].isna().sum() == 0


# ── Feature column creation ─────────────────────────────────────────


def test_get_feature_columns_returns_expected_list():
    pp = TransactionPreprocessor()
    cols = pp.get_feature_columns()
    assert isinstance(cols, list)
    assert "amount_normalized" in cols
    assert "hour_of_day" in cols
    assert "day_of_week" in cols
    assert "is_weekend" in cols
    assert "is_night_transaction" in cols
    assert "transaction_frequency" in cols
    assert "amount_deviation" in cols
    assert "amount_log" in cols
    assert "amount_zscore" in cols
    assert "merchant_category_encoded" in cols
    assert "transaction_type_encoded" in cols
    assert "channel_encoded" in cols


def test_all_feature_columns_present_after_fit_transform():
    pp = TransactionPreprocessor()
    df = _make_raw_df(20)
    result = pp.fit_transform(df)
    for col in pp.get_feature_columns():
        assert col in result.columns, f"Missing feature column: {col}"


def test_time_features_extracted_correctly():
    pp = TransactionPreprocessor()
    df = pd.DataFrame({
        "amount": [100.0],
        "timestamp": pd.to_datetime(["2024-06-15T03:30:00"]),
        "merchant_category": ["retail"],
        "transaction_type": ["purchase"],
        "channel": ["online"],
        "customer_id": ["CUST-001"],
    })
    result = pp.fit_transform(df)
    assert result.iloc[0]["hour_of_day"] == 3
    assert result.iloc[0]["is_night_transaction"] == 1  # 3 AM is night
    # 2024-06-15 is a Saturday (dayofweek == 5)
    assert result.iloc[0]["is_weekend"] == 1


# ── Normalization ────────────────────────────────────────────────────


def test_amount_normalized_has_zero_mean():
    pp = TransactionPreprocessor()
    df = _make_raw_df(100)
    result = pp.fit_transform(df)
    mean = result["amount_normalized"].mean()
    assert abs(mean) < 0.1, f"Normalized mean should be near 0, got {mean}"


def test_amount_log_nonnegative():
    pp = TransactionPreprocessor()
    df = _make_raw_df(50)
    result = pp.fit_transform(df)
    assert (result["amount_log"] >= 0).all()


def test_transform_requires_fit_first():
    pp = TransactionPreprocessor()
    df = _make_raw_df(10)
    with pytest.raises(RuntimeError, match="not been fitted"):
        pp.transform(df)


def test_transform_after_fit_produces_same_columns():
    pp = TransactionPreprocessor()
    train = _make_raw_df(40)
    test = _make_raw_df(10)
    pp.fit_transform(train)
    result = pp.transform(test)
    for col in pp.get_feature_columns():
        assert col in result.columns
