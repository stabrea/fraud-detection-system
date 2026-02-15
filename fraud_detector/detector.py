"""
Real-time fraud detection scoring engine.

Loads a trained model and scores incoming transactions, assigning
risk levels based on configurable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from fraud_detector.feature_engineer import FeatureEngineer
from fraud_detector.model import FraudModel
from fraud_detector.preprocessor import TransactionPreprocessor


class RiskLevel(str, Enum):
    """Transaction risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScoringResult:
    """Result of scoring a single transaction."""

    transaction_id: str
    fraud_probability: float
    anomaly_score: float
    hybrid_score: float
    risk_level: RiskLevel
    contributing_factors: list[str]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "fraud_probability": round(self.fraud_probability, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "hybrid_score": round(self.hybrid_score, 4),
            "risk_level": self.risk_level.value,
            "contributing_factors": self.contributing_factors,
        }


class FraudDetector:
    """Real-time fraud scoring service.

    Wraps the preprocessing, feature engineering, and model inference
    steps into a single ``score`` call suitable for production use.
    """

    # Default risk thresholds
    THRESHOLDS: dict[str, float] = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
    }

    def __init__(
        self,
        model: FraudModel,
        preprocessor: TransactionPreprocessor,
        feature_engineer: FeatureEngineer,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Args:
            model: Trained ``FraudModel``.
            preprocessor: Fitted ``TransactionPreprocessor``.
            feature_engineer: Fitted ``FeatureEngineer``.
            thresholds: Custom risk-level thresholds mapping
                ``{level: min_score}``.  Scores above the highest
                threshold are classified CRITICAL.
        """
        self._model = model
        self._preprocessor = preprocessor
        self._feature_engineer = feature_engineer
        self._thresholds = thresholds or self.THRESHOLDS
        self._scoring_history: list[ScoringResult] = []

    def score_transaction(self, transaction: dict) -> ScoringResult:
        """Score a single transaction.

        Args:
            transaction: Dictionary with transaction fields (``amount``,
                ``timestamp``, ``merchant_category``, etc.).

        Returns:
            ``ScoringResult`` with risk assessment.
        """
        df = pd.DataFrame([transaction])
        results = self.score_batch(df)
        return results[0]

    def score_batch(self, df: pd.DataFrame) -> list[ScoringResult]:
        """Score a batch of transactions.

        Args:
            df: DataFrame of raw transactions.

        Returns:
            List of ``ScoringResult`` objects, one per row.
        """
        processed = self._preprocessor.transform(df)
        featured = self._feature_engineer.transform(processed)

        # Build feature matrix
        all_features = (
            self._preprocessor.get_feature_columns()
            + self._feature_engineer.get_feature_columns()
        )
        available = [c for c in all_features if c in featured.columns]
        model_features = [
            c for c in self._model.feature_columns if c in available
        ]

        # Pad missing columns with zeros
        for col in self._model.feature_columns:
            if col not in featured.columns:
                featured[col] = 0.0

        X = featured[self._model.feature_columns].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        fraud_proba = self._model.predict_proba(X)
        anomaly_scores = self._model.anomaly_scores(X)
        hybrid_scores = self._model.hybrid_score(X)

        results: list[ScoringResult] = []
        for i in range(len(df)):
            tid = str(
                df.iloc[i].get("transaction_id", f"txn_{i}")
            )
            factors = self._identify_contributing_factors(featured.iloc[i])
            risk = self._classify_risk(float(hybrid_scores[i]))

            result = ScoringResult(
                transaction_id=tid,
                fraud_probability=float(fraud_proba[i]),
                anomaly_score=float(anomaly_scores[i]),
                hybrid_score=float(hybrid_scores[i]),
                risk_level=risk,
                contributing_factors=factors,
            )
            results.append(result)

        self._scoring_history.extend(results)
        return results

    @property
    def history(self) -> list[ScoringResult]:
        """All scoring results from this session."""
        return list(self._scoring_history)

    def get_statistics(self) -> dict:
        """Return summary statistics of scored transactions."""
        if not self._scoring_history:
            return {"total": 0}

        scores = [r.hybrid_score for r in self._scoring_history]
        risk_counts = {}
        for level in RiskLevel:
            risk_counts[level.value] = sum(
                1 for r in self._scoring_history if r.risk_level == level
            )

        return {
            "total": len(self._scoring_history),
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "risk_distribution": risk_counts,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify_risk(self, score: float) -> RiskLevel:
        """Map a hybrid score to a risk level."""
        if score >= self._thresholds.get("high", 0.7):
            return RiskLevel.CRITICAL
        if score >= self._thresholds.get("medium", 0.5):
            return RiskLevel.HIGH
        if score >= self._thresholds.get("low", 0.3):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @staticmethod
    def _identify_contributing_factors(row: pd.Series) -> list[str]:
        """Identify which features contributed to a high-risk score."""
        factors: list[str] = []

        if row.get("is_night_transaction", 0) == 1:
            factors.append("Night-time transaction")
        if row.get("amount_above_p95", 0) == 1:
            factors.append("Unusually high amount")
        if row.get("is_new_location", 0) == 1:
            factors.append("New geographic location")
        if row.get("amount_is_round", 0) == 1:
            factors.append("Round transaction amount")
        if row.get("rapid_succession", 0) == 1:
            factors.append("Rapid transaction succession")
        if row.get("high_amount_night", 0) == 1:
            factors.append("High amount during night hours")

        amount_dev = row.get("amount_deviation", 0)
        if isinstance(amount_dev, (int, float)) and abs(amount_dev) > 2:
            factors.append(
                f"Amount deviation from customer mean ({amount_dev:+.1f} sigma)"
            )

        ratio = row.get("amount_ratio_to_category_mean", 0)
        if isinstance(ratio, (int, float)) and ratio > 3.0:
            factors.append(
                f"Amount {ratio:.1f}x category average"
            )

        if not factors:
            factors.append("No significant risk factors identified")

        return factors
