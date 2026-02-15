"""
Visualization utilities for fraud detection analysis.

Generates publication-quality plots: confusion matrix, ROC curve,
feature importance, fraud patterns, and score distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from fraud_detector.model import ModelMetrics


class FraudVisualizer:
    """Generates visualizations for fraud detection model analysis.

    All methods optionally save to disk and/or display interactively.
    """

    # Consistent styling
    COLORS = {
        "primary": "#1a73e8",
        "secondary": "#ea4335",
        "success": "#34a853",
        "warning": "#fbbc04",
        "bg": "#fafafa",
    }

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        """
        Args:
            output_dir: Directory for saving plots. Created if missing.
                If ``None``, plots are shown but not saved.
        """
        self._output_dir: Optional[Path] = None
        if output_dir:
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        metrics: ModelMetrics,
        *,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot the confusion matrix as a heatmap.

        Args:
            metrics: Trained model metrics containing the confusion matrix.
            show: Whether to display the plot interactively.

        Returns:
            Path to the saved image, or ``None`` if not saving.
        """
        if metrics.confusion_matrix is None:
            return None

        cm = metrics.confusion_matrix
        fig, ax = plt.subplots(figsize=(7, 6))

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)

        labels = ["Legitimate", "Fraud"]
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                    fontweight="bold",
                )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        return self._save_or_show(fig, "confusion_matrix.png", show=show)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        *,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot the Receiver Operating Characteristic curve.

        Args:
            y_true: True binary labels.
            y_scores: Predicted probabilities for the positive class.
            show: Whether to display interactively.

        Returns:
            Path to the saved image, or ``None``.
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = np.trapz(tpr, fpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            fpr,
            tpr,
            color=self.COLORS["primary"],
            linewidth=2,
            label=f"ROC Curve (AUC = {auc_val:.4f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Random Classifier",
        )
        ax.fill_between(fpr, tpr, alpha=0.1, color=self.COLORS["primary"])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Fraud Detection Model")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        fig.tight_layout()
        return self._save_or_show(fig, "roc_curve.png", show=show)

    def plot_feature_importance(
        self,
        metrics: ModelMetrics,
        *,
        top_n: int = 15,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot feature importances as a horizontal bar chart.

        Args:
            metrics: Model metrics with ``feature_importances``.
            top_n: Number of top features to display.
            show: Whether to display interactively.

        Returns:
            Path to saved image, or ``None``.
        """
        if not metrics.feature_importances:
            return None

        items = sorted(
            metrics.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        names = [x[0] for x in reversed(items)]
        values = [x[1] for x in reversed(items)]

        fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.35)))
        bars = ax.barh(
            names,
            values,
            color=self.COLORS["primary"],
            edgecolor="white",
            height=0.7,
        )

        # Annotate values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()
        return self._save_or_show(fig, "feature_importance.png", show=show)

    def plot_fraud_patterns(
        self,
        df: pd.DataFrame,
        *,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot fraud distribution across time and amount dimensions.

        Args:
            df: DataFrame with ``is_fraud``, ``hour_of_day``, and
                ``amount`` columns.
            show: Whether to display interactively.

        Returns:
            Path to saved image, or ``None``.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- (0, 0) Fraud by hour of day ---
        if "hour_of_day" in df.columns and "is_fraud" in df.columns:
            fraud = df[df["is_fraud"] == 1]
            legit = df[df["is_fraud"] == 0]
            hours = range(24)
            fraud_by_hour = fraud.groupby("hour_of_day").size().reindex(hours, fill_value=0)
            legit_by_hour = legit.groupby("hour_of_day").size().reindex(hours, fill_value=0)

            ax = axes[0, 0]
            width = 0.35
            x = np.arange(24)
            ax.bar(x - width / 2, legit_by_hour, width, label="Legitimate", color=self.COLORS["primary"], alpha=0.7)
            ax.bar(x + width / 2, fraud_by_hour, width, label="Fraud", color=self.COLORS["secondary"], alpha=0.7)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Transaction Count")
            ax.set_title("Transaction Distribution by Hour")
            ax.legend()
            ax.set_xticks(range(0, 24, 2))

        # --- (0, 1) Amount distribution ---
        if "amount" in df.columns and "is_fraud" in df.columns:
            ax = axes[0, 1]
            fraud_amounts = df[df["is_fraud"] == 1]["amount"]
            legit_amounts = df[df["is_fraud"] == 0]["amount"]
            ax.hist(legit_amounts, bins=50, alpha=0.6, label="Legitimate", color=self.COLORS["primary"], density=True)
            ax.hist(fraud_amounts, bins=50, alpha=0.6, label="Fraud", color=self.COLORS["secondary"], density=True)
            ax.set_xlabel("Transaction Amount ($)")
            ax.set_ylabel("Density")
            ax.set_title("Amount Distribution: Fraud vs Legitimate")
            ax.legend()

        # --- (1, 0) Fraud rate by day of week ---
        if "day_of_week" in df.columns and "is_fraud" in df.columns:
            ax = axes[1, 0]
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fraud_rate = df.groupby("day_of_week")["is_fraud"].mean()
            fraud_rate = fraud_rate.reindex(range(7), fill_value=0)
            ax.bar(range(7), fraud_rate, color=self.COLORS["warning"], edgecolor="white")
            ax.set_xticks(range(7))
            ax.set_xticklabels(day_names)
            ax.set_xlabel("Day of Week")
            ax.set_ylabel("Fraud Rate")
            ax.set_title("Fraud Rate by Day of Week")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # --- (1, 1) Score distribution ---
        if "hybrid_score" in df.columns:
            ax = axes[1, 1]
            if "is_fraud" in df.columns:
                fraud_scores = df[df["is_fraud"] == 1]["hybrid_score"]
                legit_scores = df[df["is_fraud"] == 0]["hybrid_score"]
                ax.hist(legit_scores, bins=50, alpha=0.6, label="Legitimate", color=self.COLORS["primary"], density=True)
                ax.hist(fraud_scores, bins=50, alpha=0.6, label="Fraud", color=self.COLORS["secondary"], density=True)
                ax.legend()
            else:
                ax.hist(df["hybrid_score"], bins=50, color=self.COLORS["primary"], alpha=0.7, density=True)
            ax.set_xlabel("Hybrid Fraud Score")
            ax.set_ylabel("Density")
            ax.set_title("Fraud Score Distribution")
        else:
            axes[1, 1].text(0.5, 0.5, "Score data not available", ha="center", va="center", transform=axes[1, 1].transAxes)

        fig.suptitle("Fraud Pattern Analysis", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        return self._save_or_show(fig, "fraud_patterns.png", show=show)

    def plot_score_distribution(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        *,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot the distribution of fraud scores.

        Args:
            scores: Hybrid fraud scores.
            labels: Optional true labels for coloring.
            show: Whether to display interactively.

        Returns:
            Path to saved image, or ``None``.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        if labels is not None:
            legit_scores = scores[labels == 0]
            fraud_scores = scores[labels == 1]
            ax.hist(
                legit_scores, bins=50, alpha=0.6,
                label="Legitimate", color=self.COLORS["primary"], density=True,
            )
            ax.hist(
                fraud_scores, bins=50, alpha=0.6,
                label="Fraud", color=self.COLORS["secondary"], density=True,
            )
            ax.legend()
        else:
            ax.hist(
                scores, bins=50, alpha=0.7,
                color=self.COLORS["primary"], density=True,
            )

        # Risk threshold lines
        for name, threshold in [("Medium", 0.3), ("High", 0.5), ("Critical", 0.7)]:
            ax.axvline(
                x=threshold, color="gray", linestyle="--", linewidth=1, alpha=0.7
            )
            ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.9, name, fontsize=8, rotation=90, va="top")

        ax.set_xlabel("Hybrid Fraud Score")
        ax.set_ylabel("Density")
        ax.set_title("Fraud Score Distribution with Risk Thresholds")
        fig.tight_layout()
        return self._save_or_show(fig, "score_distribution.png", show=show)

    def generate_all(
        self,
        metrics: ModelMetrics,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        hybrid_scores: np.ndarray,
        *,
        show: bool = False,
    ) -> list[Optional[Path]]:
        """Generate all available visualizations.

        Args:
            metrics: Trained model metrics.
            df: Feature-enriched DataFrame.
            y_true: True labels.
            y_scores: Predicted probabilities.
            hybrid_scores: Hybrid fraud scores.
            show: Whether to display interactively.

        Returns:
            List of saved file paths.
        """
        paths = [
            self.plot_confusion_matrix(metrics, show=show),
            self.plot_roc_curve(y_true, y_scores, show=show),
            self.plot_feature_importance(metrics, show=show),
            self.plot_fraud_patterns(df, show=show),
            self.plot_score_distribution(hybrid_scores, y_true, show=show),
        ]
        return paths

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_or_show(
        self,
        fig: plt.Figure,
        filename: str,
        *,
        show: bool,
    ) -> Optional[Path]:
        """Save figure to disk and/or display it."""
        saved_path: Optional[Path] = None
        if self._output_dir:
            saved_path = self._output_dir / filename
            fig.savefig(saved_path, bbox_inches="tight", dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path
