"""
Transaction data preprocessing pipeline.

Handles missing values, normalization, categorical encoding,
and initial feature engineering for the fraud detection model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TransactionPreprocessor:
    """Preprocesses raw transaction data for fraud detection modeling.

    Applies a deterministic pipeline: missing value imputation,
    amount normalization, categorical encoding, and derived feature
    generation (frequency, deviation, time-based).
    """

    def __init__(self) -> None:
        self._amount_scaler: StandardScaler = StandardScaler()
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._fitted: bool = False
        self._amount_stats: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor on training data and return transformed copy.

        Args:
            df: Raw transaction DataFrame. Expected columns include
                ``amount``, ``timestamp``, ``merchant_category``,
                ``transaction_type``, and ``customer_id``.

        Returns:
            Preprocessed DataFrame with engineered features.
        """
        df = df.copy()
        df = self._handle_missing_values(df)
        df = self._parse_timestamps(df)
        df = self._engineer_time_features(df)
        df = self._engineer_frequency_features(df)
        df = self._engineer_amount_features(df)
        df = self._normalize_amounts(df, fit=True)
        df = self._encode_categoricals(df, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using already-fitted preprocessor.

        Args:
            df: Raw transaction DataFrame with the same schema used
                during ``fit_transform``.

        Returns:
            Preprocessed DataFrame.

        Raises:
            RuntimeError: If called before ``fit_transform``.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. Call fit_transform first."
            )
        df = df.copy()
        df = self._handle_missing_values(df)
        df = self._parse_timestamps(df)
        df = self._engineer_time_features(df)
        df = self._engineer_frequency_features(df)
        df = self._engineer_amount_features(df)
        df = self._normalize_amounts(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        return df

    def get_feature_columns(self) -> list[str]:
        """Return the list of feature column names after preprocessing."""
        return [
            "amount_normalized",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_night_transaction",
            "transaction_frequency",
            "amount_deviation",
            "amount_log",
            "amount_zscore",
            "merchant_category_encoded",
            "transaction_type_encoded",
            "channel_encoded",
        ]

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with sensible defaults."""
        if "amount" in df.columns:
            df["amount"] = df["amount"].fillna(df["amount"].median())

        if "merchant_category" in df.columns:
            df["merchant_category"] = df["merchant_category"].fillna("unknown")

        if "transaction_type" in df.columns:
            df["transaction_type"] = df["transaction_type"].fillna("unknown")

        if "channel" in df.columns:
            df["channel"] = df["channel"].fillna("unknown")

        if "location" in df.columns:
            df["location"] = df["location"].fillna("unknown")

        return df

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamp column is datetime."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamp."""
        if "timestamp" not in df.columns:
            return df

        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night_transaction"] = (
            (df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)
        ).astype(int)

        return df

    def _engineer_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-customer transaction frequency."""
        if "customer_id" not in df.columns:
            df["transaction_frequency"] = 0
            return df

        freq = df.groupby("customer_id")["customer_id"].transform("count")
        df["transaction_frequency"] = freq
        return df

    def _engineer_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive amount-based statistical features."""
        if "amount" not in df.columns:
            df["amount_deviation"] = 0.0
            df["amount_log"] = 0.0
            df["amount_zscore"] = 0.0
            return df

        # Log-transform (shift by 1 to handle zero amounts)
        df["amount_log"] = np.log1p(df["amount"].clip(lower=0))

        # Per-customer amount deviation
        if "customer_id" in df.columns:
            customer_mean = df.groupby("customer_id")["amount"].transform("mean")
            customer_std = df.groupby("customer_id")["amount"].transform("std").fillna(1.0)
            df["amount_deviation"] = (df["amount"] - customer_mean) / customer_std
        else:
            global_mean = df["amount"].mean()
            global_std = df["amount"].std() or 1.0
            df["amount_deviation"] = (df["amount"] - global_mean) / global_std

        # Z-score against global distribution
        mean = df["amount"].mean()
        std = df["amount"].std() or 1.0
        df["amount_zscore"] = (df["amount"] - mean) / std

        return df

    def _normalize_amounts(
        self, df: pd.DataFrame, *, fit: bool
    ) -> pd.DataFrame:
        """Standard-scale the raw amount column."""
        if "amount" not in df.columns:
            df["amount_normalized"] = 0.0
            return df

        values = df[["amount"]].values
        if fit:
            df["amount_normalized"] = self._amount_scaler.fit_transform(values).ravel()
        else:
            df["amount_normalized"] = self._amount_scaler.transform(values).ravel()
        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, *, fit: bool
    ) -> pd.DataFrame:
        """Label-encode categorical columns."""
        categorical_cols = {
            "merchant_category": "merchant_category_encoded",
            "transaction_type": "transaction_type_encoded",
            "channel": "channel_encoded",
        }
        for src_col, dst_col in categorical_cols.items():
            if src_col not in df.columns:
                df[dst_col] = 0
                continue

            if fit:
                le = LabelEncoder()
                df[dst_col] = le.fit_transform(df[src_col].astype(str))
                self._label_encoders[src_col] = le
            else:
                le = self._label_encoders.get(src_col)
                if le is None:
                    df[dst_col] = 0
                    continue
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[dst_col] = df[src_col].astype(str).map(
                    lambda x, _known=known, _le=le: (
                        _le.transform([x])[0] if x in _known else -1
                    )
                )
        return df
