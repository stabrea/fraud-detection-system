"""
ML model training and evaluation for fraud detection.

Combines a supervised Random Forest classifier with an unsupervised
Isolation Forest for anomaly detection, producing a hybrid fraud score.

Supports both random stratified splitting (for synthetic data) and
temporal splitting (for real time-series data like the ULB dataset).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    cv_scores: list[float] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    precision_curve: Optional[np.ndarray] = None
    recall_curve: Optional[np.ndarray] = None
    split_method: str = "random"

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Model Performance Metrics",
            "=" * 40,
            f"Split:     {self.split_method}",
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1 Score:  {self.f1:.4f}",
            f"ROC AUC:   {self.roc_auc:.4f}",
            f"PR AUC:    {self.pr_auc:.4f}",
        ]
        if self.cv_scores:
            lines.append(
                f"CV Mean:   {np.mean(self.cv_scores):.4f} "
                f"(+/- {np.std(self.cv_scores):.4f})"
            )
        lines.append("")
        lines.append(self.classification_report)
        return "\n".join(lines)


class FraudModel:
    """Hybrid fraud detection model.

    Trains a Random Forest for supervised classification and an
    Isolation Forest for unsupervised anomaly detection.  The final
    fraud score blends both signals.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.02,
        random_state: int = 42,
        rf_class_weight: Optional[str] = "balanced",
        test_size: float = 0.2,
        cv_folds: int = 5,
        use_smote: bool = False,
    ) -> None:
        """
        Args:
            n_estimators: Number of trees for both forests.
            contamination: Expected fraud proportion for Isolation Forest.
            random_state: Seed for reproducibility.
            rf_class_weight: Class-weight strategy for Random Forest.
                ``"balanced"`` compensates for the low fraud rate.
            test_size: Fraction held out for evaluation.
            cv_folds: Number of stratified cross-validation folds.
            use_smote: Whether to apply SMOTE to the training set.
        """
        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=rf_class_weight,
            random_state=random_state,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        self._iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._random_state = random_state
        self._test_size = test_size
        self._cv_folds = cv_folds
        self._use_smote = use_smote
        self._feature_columns: list[str] = []
        self._trained: bool = False
        self._metrics: Optional[ModelMetrics] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "is_fraud",
        temporal_split: bool = False,
        time_column: Optional[str] = None,
    ) -> ModelMetrics:
        """Train both models and evaluate on a held-out split.

        Args:
            df: Feature-enriched DataFrame.
            feature_columns: Column names to use as features.
            target_column: Binary label column (1 = fraud).
            temporal_split: If True, use temporal ordering for
                train/test split instead of random stratified split.
                This is the correct approach for time-series fraud data.
            time_column: Column to sort by for temporal split.
                If None and temporal_split is True, uses the DataFrame
                index order (assumes data is already sorted by time).

        Returns:
            ``ModelMetrics`` with evaluation results.
        """
        self._feature_columns = [
            c for c in feature_columns if c in df.columns
        ]
        X = df[self._feature_columns].values.astype(np.float64)
        y = df[target_column].values.astype(int)

        # Replace any remaining NaN / inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if temporal_split:
            # Sort by time column if provided
            if time_column and time_column in df.columns:
                sort_idx = df[time_column].values.argsort()
                X = X[sort_idx]
                y = y[sort_idx]

            # Temporal split: first 80% train, last 20% test
            split_idx = int(len(X) * (1 - self._test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            split_method = "temporal"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self._test_size,
                random_state=self._random_state,
                stratify=y,
            )
            split_method = "random_stratified"

        # --- Optional SMOTE on training set only ---
        if self._use_smote:
            X_train, y_train = self._apply_smote(X_train, y_train)

        # --- Supervised: Random Forest ---
        self._rf.fit(X_train, y_train)

        # --- Unsupervised: Isolation Forest ---
        self._iso.fit(X_train)

        # --- Evaluate ---
        y_pred = self._rf.predict(X_test)
        y_proba = self._rf.predict_proba(X_test)[:, 1]

        # Precision-recall curve
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)

        metrics = ModelMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, y_proba)),
            pr_auc=float(average_precision_score(y_test, y_proba)),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            classification_report=classification_report(
                y_test, y_pred, target_names=["Legitimate", "Fraud"]
            ),
            precision_curve=pr_precision,
            recall_curve=pr_recall,
            split_method=split_method,
        )

        # Cross-validation (only meaningful for non-temporal splits)
        if not temporal_split:
            cv = StratifiedKFold(
                n_splits=self._cv_folds,
                shuffle=True,
                random_state=self._random_state,
            )
            cv_scores = cross_val_score(
                self._rf, X, y, cv=cv, scoring="f1", n_jobs=-1
            )
            metrics.cv_scores = cv_scores.tolist()

        # Feature importances
        importances = self._rf.feature_importances_
        metrics.feature_importances = dict(
            sorted(
                zip(self._feature_columns, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        self._metrics = metrics
        self._trained = True
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probabilities from the Random Forest.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            1-D array of fraud probabilities.
        """
        self._assert_trained()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self._rf.predict_proba(X)[:, 1]

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return Isolation Forest anomaly scores (lower = more anomalous).

        Args:
            X: Feature matrix.

        Returns:
            1-D array of anomaly scores in ``[-1, 0]`` range,
            rescaled to ``[0, 1]`` where 1 = most anomalous.
        """
        self._assert_trained()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        raw = self._iso.decision_function(X)
        # decision_function: large positive = normal, negative = anomaly
        # Rescale so 1 = most anomalous
        normalized = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
        return normalized

    def hybrid_score(
        self,
        X: np.ndarray,
        rf_weight: float = 0.7,
        iso_weight: float = 0.3,
    ) -> np.ndarray:
        """Blended fraud score combining both models.

        Args:
            X: Feature matrix.
            rf_weight: Weight for Random Forest probability.
            iso_weight: Weight for Isolation Forest anomaly score.

        Returns:
            1-D array of hybrid scores in ``[0, 1]``.
        """
        rf_proba = self.predict_proba(X)
        iso_score = self.anomaly_scores(X)
        return rf_weight * rf_proba + iso_weight * iso_score

    @property
    def metrics(self) -> Optional[ModelMetrics]:
        """Most recent evaluation metrics, or ``None`` if not yet trained."""
        return self._metrics

    @property
    def feature_columns(self) -> list[str]:
        """Feature columns used during training."""
        return list(self._feature_columns)

    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk.

        Args:
            path: Directory in which to save model artifacts.
        """
        self._assert_trained()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "random_forest.pkl", "wb") as f:
            pickle.dump(self._rf, f)
        with open(path / "isolation_forest.pkl", "wb") as f:
            pickle.dump(self._iso, f)
        with open(path / "feature_columns.json", "w") as f:
            json.dump(self._feature_columns, f)

    def load(self, path: str | Path) -> None:
        """Load a previously saved model.

        Args:
            path: Directory containing model artifacts.
        """
        path = Path(path)
        with open(path / "random_forest.pkl", "rb") as f:
            self._rf = pickle.load(f)
        with open(path / "isolation_forest.pkl", "rb") as f:
            self._iso = pickle.load(f)
        with open(path / "feature_columns.json") as f:
            self._feature_columns = json.load(f)
        self._trained = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to the training set only.

        Falls back silently if imbalanced-learn is not installed.
        """
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=self._random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            n_original = len(y)
            n_resampled = len(y_resampled)
            print(
                f"  SMOTE: {n_original:,} -> {n_resampled:,} samples "
                f"(+{n_resampled - n_original:,} synthetic fraud)"
            )
            return X_resampled, y_resampled
        except ImportError:
            print("  Warning: imbalanced-learn not installed, skipping SMOTE.")
            print("  Install with: pip install imbalanced-learn")
            return X, y

    def _assert_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
