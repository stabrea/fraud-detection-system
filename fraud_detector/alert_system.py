"""
Alert generation and management for flagged transactions.

Generates structured alerts, tracks false positive rates, and
produces summary reports for security analysts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from fraud_detector.detector import RiskLevel, ScoringResult


class AlertStatus(str, Enum):
    """Lifecycle status of an alert."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"
    DISMISSED = "dismissed"


@dataclass
class Alert:
    """A fraud alert generated from a scoring result."""

    alert_id: str
    transaction_id: str
    risk_level: RiskLevel
    hybrid_score: float
    contributing_factors: list[str]
    created_at: str
    status: AlertStatus = AlertStatus.OPEN
    analyst_notes: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "risk_level": self.risk_level.value,
            "hybrid_score": round(self.hybrid_score, 4),
            "contributing_factors": self.contributing_factors,
            "created_at": self.created_at,
            "status": self.status.value,
            "analyst_notes": self.analyst_notes,
        }


class AlertSystem:
    """Manages fraud alerts: generation, tracking, and reporting.

    Alerts are generated for transactions exceeding a configurable
    score threshold.  The system tracks analyst feedback to compute
    false positive rates.
    """

    def __init__(self, alert_threshold: float = 0.5) -> None:
        """
        Args:
            alert_threshold: Minimum hybrid score to trigger an alert.
        """
        self._threshold = alert_threshold
        self._alerts: list[Alert] = []
        self._alert_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_results(self, results: list[ScoringResult]) -> list[Alert]:
        """Generate alerts for scoring results above threshold.

        Args:
            results: List of ``ScoringResult`` from the detector.

        Returns:
            Newly created ``Alert`` objects.
        """
        new_alerts: list[Alert] = []
        for result in results:
            if result.hybrid_score >= self._threshold:
                alert = self._create_alert(result)
                self._alerts.append(alert)
                new_alerts.append(alert)
        return new_alerts

    def update_alert(
        self,
        alert_id: str,
        status: AlertStatus,
        notes: str = "",
    ) -> Optional[Alert]:
        """Update the status of an existing alert.

        Args:
            alert_id: The alert to update.
            status: New status.
            notes: Optional analyst notes.

        Returns:
            Updated alert, or ``None`` if not found.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.status = status
                if notes:
                    alert.analyst_notes = notes
                return alert
        return None

    def get_open_alerts(self) -> list[Alert]:
        """Return all alerts that are still open or under investigation."""
        return [
            a
            for a in self._alerts
            if a.status in (AlertStatus.OPEN, AlertStatus.INVESTIGATING)
        ]

    def get_alerts_by_risk(self, risk_level: RiskLevel) -> list[Alert]:
        """Return alerts filtered by risk level."""
        return [a for a in self._alerts if a.risk_level == risk_level]

    def get_false_positive_rate(self) -> float:
        """Compute the false positive rate from analyst feedback.

        Only considers alerts that have been resolved (confirmed
        fraud or false positive).

        Returns:
            False positive rate as a float in ``[0, 1]``, or ``0.0``
            if no alerts have been resolved.
        """
        resolved = [
            a
            for a in self._alerts
            if a.status
            in (AlertStatus.CONFIRMED_FRAUD, AlertStatus.FALSE_POSITIVE)
        ]
        if not resolved:
            return 0.0

        fp_count = sum(
            1 for a in resolved if a.status == AlertStatus.FALSE_POSITIVE
        )
        return fp_count / len(resolved)

    def generate_report(self) -> str:
        """Generate a text summary report of all alerts.

        Returns:
            Multi-line report string.
        """
        if not self._alerts:
            return "No alerts generated."

        lines: list[str] = [
            "Fraud Alert Report",
            "=" * 60,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Total Alerts: {len(self._alerts)}",
            "",
        ]

        # Status breakdown
        lines.append("Status Breakdown:")
        status_counts: dict[str, int] = {}
        for alert in self._alerts:
            key = alert.status.value
            status_counts[key] = status_counts.get(key, 0) + 1
        for status, count in sorted(status_counts.items()):
            lines.append(f"  {status:20s} {count:>5d}")

        # Risk level breakdown
        lines.append("")
        lines.append("Risk Level Breakdown:")
        risk_counts: dict[str, int] = {}
        for alert in self._alerts:
            key = alert.risk_level.value
            risk_counts[key] = risk_counts.get(key, 0) + 1
        for level, count in sorted(risk_counts.items()):
            lines.append(f"  {level:20s} {count:>5d}")

        # Score statistics
        scores = [a.hybrid_score for a in self._alerts]
        lines.append("")
        lines.append("Score Statistics:")
        lines.append(f"  Mean:   {np.mean(scores):.4f}")
        lines.append(f"  Median: {np.median(scores):.4f}")
        lines.append(f"  Max:    {np.max(scores):.4f}")
        lines.append(f"  Min:    {np.min(scores):.4f}")

        # False positive rate
        fpr = self.get_false_positive_rate()
        lines.append("")
        lines.append(f"False Positive Rate: {fpr:.2%}")

        # Top alerts
        lines.append("")
        lines.append("Top 10 Highest-Risk Alerts:")
        lines.append("-" * 60)
        top = sorted(self._alerts, key=lambda a: a.hybrid_score, reverse=True)[
            :10
        ]
        for alert in top:
            lines.append(
                f"  [{alert.alert_id}] {alert.transaction_id} "
                f"| score={alert.hybrid_score:.4f} "
                f"| risk={alert.risk_level.value} "
                f"| status={alert.status.value}"
            )
            if alert.contributing_factors:
                factors_str = "; ".join(alert.contributing_factors[:3])
                lines.append(f"    Factors: {factors_str}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all alerts as a DataFrame."""
        if not self._alerts:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self._alerts])

    @property
    def total_alerts(self) -> int:
        """Total number of alerts generated."""
        return len(self._alerts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_alert(self, result: ScoringResult) -> Alert:
        """Create an Alert from a ScoringResult."""
        self._alert_counter += 1
        return Alert(
            alert_id=f"ALT-{self._alert_counter:06d}",
            transaction_id=result.transaction_id,
            risk_level=result.risk_level,
            hybrid_score=result.hybrid_score,
            contributing_factors=result.contributing_factors,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
