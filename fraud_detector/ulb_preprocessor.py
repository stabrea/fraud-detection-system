"""
Preprocessing pipeline for the ULB Credit Card Fraud dataset.

The ULB dataset has PCA-transformed features (V1-V28) that are
already numerical.  Only Time and Amount need standardization.
This is a simpler pipeline than the synthetic data preprocessor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ULBPreprocessor:
    """Preprocesses the ULB Credit Card Fraud dataset.

    V1-V28 are already PCA-transformed and don't need further
    feature engineering.  This preprocessor standardizes Amount
    and Time, and optionally creates derived features.
    """

    def __init__(self) -> None:
        self._amount_scaler: StandardScaler = StandardScaler()
        self._time_scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the ULB dataset.

        Args:
            df: Raw ULB DataFrame with columns Time, V1-V28, Amount, Class.

        Returns:
            Preprocessed DataFrame ready for model training.
        """
        df = df.copy()

        # Standardize Amount
        df["Amount_scaled"] = self._amount_scaler.fit_transform(
            df[["Amount"]].values
        ).ravel()

        # Standardize Time
        df["Time_scaled"] = self._time_scaler.fit_transform(
            df[["Time"]].values
        ).ravel()

        # Log-transform Amount (useful feature)
        df["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))

        # Hour-of-day proxy from Time (seconds from first transaction)
        # ULB data spans ~2 days, Time is seconds elapsed
        df["Time_hour"] = (df["Time"] / 3600) % 24

        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scalers.

        Args:
            df: Raw ULB DataFrame.

        Returns:
            Preprocessed DataFrame.

        Raises:
            RuntimeError: If called before fit_transform.
        """
        if not self._fitted:
            raise RuntimeError(
                "ULBPreprocessor not fitted. Call fit_transform first."
            )
        df = df.copy()

        df["Amount_scaled"] = self._amount_scaler.transform(
            df[["Amount"]].values
        ).ravel()
        df["Time_scaled"] = self._time_scaler.transform(
            df[["Time"]].values
        ).ravel()
        df["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
        df["Time_hour"] = (df["Time"] / 3600) % 24

        return df

    def get_feature_columns(self) -> list[str]:
        """Return feature columns for model training.

        Uses V1-V28 (PCA features) plus the standardized/derived
        Amount and Time features.
        """
        pca_features = [f"V{i}" for i in range(1, 29)]
        derived_features = [
            "Amount_scaled",
            "Time_scaled",
            "Amount_log",
            "Time_hour",
        ]
        return pca_features + derived_features
