"""Tests for the data loader module."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.data_loader import (
    ULB_COLUMNS,
    detect_dataset_type,
    generate_ulb_format_synthetic,
    load_dataset,
    _validate_ulb_schema,
)


# ── ULB-format synthetic generation ────────────────────────────────


def test_generate_ulb_format_has_correct_columns():
    df = generate_ulb_format_synthetic(n_rows=1000)
    assert list(df.columns) == ULB_COLUMNS


def test_generate_ulb_format_correct_row_count():
    df = generate_ulb_format_synthetic(n_rows=500)
    assert len(df) == 500


def test_generate_ulb_format_fraud_rate():
    df = generate_ulb_format_synthetic(n_rows=10000, fraud_rate=0.01)
    actual_rate = df["Class"].mean()
    assert 0.005 < actual_rate < 0.02  # allow some tolerance


def test_generate_ulb_format_realistic_class_imbalance():
    df = generate_ulb_format_synthetic(n_rows=10000, fraud_rate=0.00172)
    n_fraud = df["Class"].sum()
    assert 10 <= n_fraud <= 30  # ~17 expected at 0.172%


def test_generate_ulb_format_sorted_by_time():
    df = generate_ulb_format_synthetic(n_rows=1000)
    assert (df["Time"].diff().dropna() >= 0).all()


def test_generate_ulb_format_amount_positive():
    df = generate_ulb_format_synthetic(n_rows=1000)
    assert (df["Amount"] >= 0).all()


def test_generate_ulb_format_class_binary():
    df = generate_ulb_format_synthetic(n_rows=1000)
    assert set(df["Class"].unique()).issubset({0, 1})


def test_generate_ulb_format_reproducible():
    df1 = generate_ulb_format_synthetic(n_rows=100, seed=42)
    df2 = generate_ulb_format_synthetic(n_rows=100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


# ── Schema validation ──────────────────────────────────────────────


def test_validate_ulb_schema_correct():
    df = generate_ulb_format_synthetic(n_rows=10)
    assert _validate_ulb_schema(df) is True


def test_validate_ulb_schema_missing_class():
    df = generate_ulb_format_synthetic(n_rows=10)
    df = df.drop(columns=["Class"])
    assert _validate_ulb_schema(df) is False


def test_validate_ulb_schema_wrong_format():
    df = pd.DataFrame({"customer_id": ["A"], "amount": [100], "is_fraud": [0]})
    assert _validate_ulb_schema(df) is False


# ── Dataset type detection ─────────────────────────────────────────


def test_detect_dataset_type_nonexistent():
    assert detect_dataset_type("/nonexistent/path.csv") == "unknown"


# ── Load dataset with synthetic fallback ───────────────────────────


def test_load_dataset_synthetic_fallback():
    df = load_dataset(
        "/nonexistent/path.csv",
        fallback_synthetic=True,
        synthetic_n=500,
    )
    assert len(df) == 500
    assert list(df.columns) == ULB_COLUMNS


def test_load_dataset_no_fallback_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path.csv", fallback_synthetic=False)
