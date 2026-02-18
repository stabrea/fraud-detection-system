"""
Fraud Detection System — End-to-End Demo
=========================================

Runs the complete pipeline: data loading, preprocessing, feature
engineering, model training, scoring, alerting, and visualization.

Supports both synthetic data and real-world datasets (ULB Credit Card).

Usage:
    python main.py
    python main.py --data data/sample_transactions.csv
    python main.py --ulb data/raw/creditcard.csv
    python main.py --rows 10000 --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detector.alert_system import AlertSystem
from fraud_detector.detector import FraudDetector
from fraud_detector.feature_engineer import FeatureEngineer
from fraud_detector.model import FraudModel
from fraud_detector.preprocessor import TransactionPreprocessor
from fraud_detector.visualizer import FraudVisualizer


def _divider(title: str) -> None:
    """Print a section divider."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def run_ulb_pipeline(
    data_path: str,
    save_model: bool = True,
    visualize: bool = False,
    output_dir: str = "output",
    use_smote: bool = True,
) -> None:
    """Execute the fraud detection pipeline on ULB Credit Card data.

    Uses temporal train/test split and ULB-specific preprocessing.

    Args:
        data_path: Path to ULB creditcard.csv.
        save_model: Whether to persist the trained model.
        visualize: Whether to generate visualization plots.
        output_dir: Directory for all output artifacts.
        use_smote: Whether to apply SMOTE on training set.
    """
    from fraud_detector.data_loader import load_dataset
    from fraud_detector.ulb_preprocessor import ULBPreprocessor

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data Loading
    # ------------------------------------------------------------------
    _divider("1. DATA LOADING (ULB Credit Card)")

    df = load_dataset(data_path, fallback_synthetic=True)
    fraud_count = int(df["Class"].sum())
    print(f"  Total transactions: {len(df):,}")
    print(f"  Fraudulent:         {fraud_count:,} ({fraud_count / len(df):.3%})")
    print(f"  Legitimate:         {len(df) - fraud_count:,}")

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    _divider("2. PREPROCESSING")

    preprocessor = ULBPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    feature_cols = preprocessor.get_feature_columns()

    print(f"  Features: {len(feature_cols)}")
    print(f"  PCA features: V1-V28")
    print(f"  Derived: Amount_scaled, Time_scaled, Amount_log, Time_hour")

    # ------------------------------------------------------------------
    # 3. Model Training (Temporal Split)
    # ------------------------------------------------------------------
    _divider("3. MODEL TRAINING (Temporal Split)")

    fraud_rate = df["Class"].mean()
    model = FraudModel(
        n_estimators=200,
        contamination=fraud_rate,
        use_smote=use_smote,
    )
    metrics = model.train(
        df_processed,
        feature_cols,
        target_column="Class",
        temporal_split=True,
        time_column="Time",
    )

    print(f"\n{metrics.summary()}")

    if save_model:
        model_dir = output / "model"
        model.save(model_dir)
        print(f"  Model saved to {model_dir}/")

    # ------------------------------------------------------------------
    # 4. Visualization
    # ------------------------------------------------------------------
    if visualize:
        _divider("4. VISUALIZATION")

        viz_dir = output / "plots"
        visualizer = FraudVisualizer(output_dir=viz_dir)

        X = df_processed[model.feature_columns].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_true = df_processed["Class"].values
        y_scores = model.predict_proba(X)

        paths = [
            visualizer.plot_confusion_matrix(metrics),
            visualizer.plot_roc_curve(y_true, y_scores),
            visualizer.plot_feature_importance(metrics),
            visualizer.plot_score_distribution(y_scores, y_true),
        ]
        saved = [p for p in paths if p is not None]
        print(f"  Generated {len(saved)} plots:")
        for p in saved:
            print(f"    - {p}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _divider("PIPELINE COMPLETE")
    print(f"  All artifacts saved to: {output.resolve()}/")
    print(f"  Split method:    {metrics.split_method}")
    print(f"  Model ROC AUC:   {metrics.roc_auc:.4f}")
    print(f"  Model PR AUC:    {metrics.pr_auc:.4f}")
    print(f"  Model F1 score:  {metrics.f1:.4f}")


def run_pipeline(
    data_path: str | None = None,
    n_rows: int = 10000,
    fraud_rate: float = 0.02,
    save_model: bool = True,
    visualize: bool = False,
    output_dir: str = "output",
) -> None:
    """Execute the full fraud detection pipeline on synthetic data.

    Args:
        data_path: Path to existing CSV. If ``None``, synthetic data
            is generated on the fly.
        n_rows: Number of rows to generate (ignored if ``data_path``
            is provided).
        fraud_rate: Target fraud rate for synthetic data.
        save_model: Whether to persist the trained model.
        visualize: Whether to generate visualization plots.
        output_dir: Directory for all output artifacts.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data Loading / Generation
    # ------------------------------------------------------------------
    _divider("1. DATA LOADING")

    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} transactions from {data_path}")
    else:
        from data.generate_dataset import generate_transactions

        print(f"Generating {n_rows:,} synthetic transactions...")
        df = generate_transactions(n_rows=n_rows, fraud_rate=fraud_rate)
        gen_path = output / "generated_transactions.csv"
        df.to_csv(gen_path, index=False)
        print(f"Saved generated data to {gen_path}")

    fraud_count = df["is_fraud"].sum()
    print(f"  Total transactions: {len(df):,}")
    print(f"  Fraudulent:         {fraud_count:,} ({fraud_count / len(df):.1%})")
    print(f"  Legitimate:         {len(df) - fraud_count:,}")

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    _divider("2. PREPROCESSING")

    preprocessor = TransactionPreprocessor()
    df_processed = preprocessor.fit_transform(df)

    print(f"  Features after preprocessing: {len(preprocessor.get_feature_columns())}")
    for col in preprocessor.get_feature_columns():
        print(f"    - {col}")

    # ------------------------------------------------------------------
    # 3. Feature Engineering
    # ------------------------------------------------------------------
    _divider("3. FEATURE ENGINEERING")

    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df_processed)

    print(f"  Features after engineering:   {len(engineer.get_feature_columns())}")
    for col in engineer.get_feature_columns():
        print(f"    - {col}")

    # ------------------------------------------------------------------
    # 4. Model Training
    # ------------------------------------------------------------------
    _divider("4. MODEL TRAINING")

    feature_cols = (
        preprocessor.get_feature_columns()
        + engineer.get_feature_columns()
    )
    feature_cols = [c for c in feature_cols if c in df_featured.columns]
    print(f"  Total features: {len(feature_cols)}")

    model = FraudModel(n_estimators=200, contamination=fraud_rate)
    metrics = model.train(df_featured, feature_cols, target_column="is_fraud")

    print(f"\n{metrics.summary()}")

    if save_model:
        model_dir = output / "model"
        model.save(model_dir)
        print(f"  Model saved to {model_dir}/")

    # ------------------------------------------------------------------
    # 5. Real-Time Scoring Demo
    # ------------------------------------------------------------------
    _divider("5. REAL-TIME SCORING DEMO")

    detector = FraudDetector(model, preprocessor, engineer)

    # Score a sample batch
    sample = df.sample(n=min(20, len(df)), random_state=42)
    results = detector.score_batch(sample)

    print(f"  Scored {len(results)} transactions:\n")
    for r in sorted(results, key=lambda x: x.hybrid_score, reverse=True)[:10]:
        risk_color = {
            "low": "  ",
            "medium": "* ",
            "high": "**",
            "critical": "!!",
        }.get(r.risk_level.value, "  ")
        print(
            f"  {risk_color} {r.transaction_id} "
            f"| score={r.hybrid_score:.4f} "
            f"| risk={r.risk_level.value:8s} "
            f"| factors: {', '.join(r.contributing_factors[:2])}"
        )

    stats = detector.get_statistics()
    print(f"\n  Mean score: {stats['mean_score']:.4f}")
    print(f"  Risk distribution: {stats['risk_distribution']}")

    # ------------------------------------------------------------------
    # 6. Alert System
    # ------------------------------------------------------------------
    _divider("6. ALERT SYSTEM")

    # Score the full dataset for alerting
    all_results = detector.score_batch(df)
    alert_system = AlertSystem(alert_threshold=0.5)
    alerts = alert_system.process_results(all_results)

    print(f"  Total alerts generated: {alert_system.total_alerts}")
    print(f"\n{alert_system.generate_report()}")

    # ------------------------------------------------------------------
    # 7. Visualization
    # ------------------------------------------------------------------
    if visualize:
        _divider("7. VISUALIZATION")

        viz_dir = output / "plots"
        visualizer = FraudVisualizer(output_dir=viz_dir)

        X = df_featured[model.feature_columns].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_true = df_featured["is_fraud"].values
        y_scores = model.predict_proba(X)
        hybrid_scores = model.hybrid_score(X)

        df_featured["hybrid_score"] = hybrid_scores

        paths = visualizer.generate_all(
            metrics, df_featured, y_true, y_scores, hybrid_scores
        )
        saved = [p for p in paths if p is not None]
        print(f"  Generated {len(saved)} plots:")
        for p in saved:
            print(f"    - {p}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _divider("PIPELINE COMPLETE")
    print(f"  All artifacts saved to: {output.resolve()}/")
    print(f"  Model accuracy:  {metrics.accuracy:.4f}")
    print(f"  Model F1 score:  {metrics.f1:.4f}")
    print(f"  Model ROC AUC:   {metrics.roc_auc:.4f}")
    print(f"  Alerts raised:   {alert_system.total_alerts}")


def main() -> int:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Fraud Detection System — End-to-End Demo",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to synthetic transaction CSV (generates data if omitted)",
    )
    parser.add_argument(
        "--ulb",
        type=str,
        default=None,
        help="Path to ULB Credit Card dataset CSV (enables real-data pipeline)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Number of synthetic transactions (default: 10000)",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.02,
        help="Synthetic fraud rate (default: 0.02)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling on training set (ULB pipeline)",
    )

    args = parser.parse_args()

    try:
        if args.ulb:
            run_ulb_pipeline(
                data_path=args.ulb,
                visualize=args.visualize,
                output_dir=args.output,
                use_smote=args.smote,
            )
        else:
            run_pipeline(
                data_path=args.data,
                n_rows=args.rows,
                fraud_rate=args.fraud_rate,
                visualize=args.visualize,
                output_dir=args.output,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"\nFatal error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
