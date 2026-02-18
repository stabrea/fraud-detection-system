"""Tests for the ULB-specific preprocessor."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.data_loader import generate_ulb_format_synthetic
from fraud_detector.ulb_preprocessor import ULBPreprocessor


def _make_ulb_df(n: int = 100) -> pd.DataFrame:
    """Create a small ULB-format dataset for testing."""
    return generate_ulb_format_synthetic(n_rows=n, seed=42)


# ── Feature creation ───────────────────────────────────────────────


def test_fit_transform_creates_derived_features():
    pp = ULBPreprocessor()
    df = _make_ulb_df(50)
    result = pp.fit_transform(df)
    assert "Amount_scaled" in result.columns
    assert "Time_scaled" in result.columns
    assert "Amount_log" in result.columns
    assert "Time_hour" in result.columns


def test_feature_columns_includes_pca_and_derived():
    pp = ULBPreprocessor()
    cols = pp.get_feature_columns()
    # Should include V1-V28
    for i in range(1, 29):
        assert f"V{i}" in cols
    # Should include derived features
    assert "Amount_scaled" in cols
    assert "Time_scaled" in cols
    assert "Amount_log" in cols
    assert "Time_hour" in cols


def test_feature_columns_count():
    pp = ULBPreprocessor()
    cols = pp.get_feature_columns()
    assert len(cols) == 32  # 28 PCA + 4 derived


# ── Normalization ──────────────────────────────────────────────────


def test_amount_scaled_near_zero_mean():
    pp = ULBPreprocessor()
    df = _make_ulb_df(200)
    result = pp.fit_transform(df)
    mean = result["Amount_scaled"].mean()
    assert abs(mean) < 0.1


def test_amount_log_nonnegative():
    pp = ULBPreprocessor()
    df = _make_ulb_df(100)
    result = pp.fit_transform(df)
    assert (result["Amount_log"] >= 0).all()


def test_time_hour_in_range():
    pp = ULBPreprocessor()
    df = _make_ulb_df(100)
    result = pp.fit_transform(df)
    assert (result["Time_hour"] >= 0).all()
    assert (result["Time_hour"] < 24).all()


# ── Transform consistency ─────────────────────────────────────────


def test_transform_requires_fit_first():
    pp = ULBPreprocessor()
    df = _make_ulb_df(10)
    with pytest.raises(RuntimeError, match="not fitted"):
        pp.transform(df)


def test_transform_after_fit_same_columns():
    pp = ULBPreprocessor()
    train = _make_ulb_df(100)
    test = _make_ulb_df(20)
    pp.fit_transform(train)
    result = pp.transform(test)
    for col in pp.get_feature_columns():
        assert col in result.columns


# ── Integration with model ─────────────────────────────────────────


def test_preprocessor_output_compatible_with_model():
    from fraud_detector.model import FraudModel

    pp = ULBPreprocessor()
    df = _make_ulb_df(500)
    processed = pp.fit_transform(df)
    feature_cols = pp.get_feature_columns()

    model = FraudModel(n_estimators=10, cv_folds=2, contamination=0.01)
    metrics = model.train(processed, feature_cols, target_column="Class")
    assert metrics.roc_auc > 0.5  # better than random


def test_temporal_split_with_ulb_data():
    from fraud_detector.model import FraudModel

    pp = ULBPreprocessor()
    df = _make_ulb_df(500)
    processed = pp.fit_transform(df)
    feature_cols = pp.get_feature_columns()

    model = FraudModel(n_estimators=10, contamination=0.01)
    metrics = model.train(
        processed, feature_cols,
        target_column="Class",
        temporal_split=True,
        time_column="Time",
    )
    assert metrics.split_method == "temporal"
    assert metrics.roc_auc > 0.5
