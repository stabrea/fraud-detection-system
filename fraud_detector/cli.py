"""
Command-line interface for the fraud detection system.

Provides subcommands for training, scoring, generating data,
and producing reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fraud_detector.alert_system import AlertSystem
from fraud_detector.detector import FraudDetector
from fraud_detector.feature_engineer import FeatureEngineer
from fraud_detector.model import FraudModel
from fraud_detector.preprocessor import TransactionPreprocessor
from fraud_detector.visualizer import FraudVisualizer


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="fraud-detector",
        description="ML-based Financial Fraud Detection System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train the fraud detection model")
    train_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training CSV file",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="model_output",
        help="Directory to save trained model (default: model_output)",
    )

    # --- score ---
    score_parser = subparsers.add_parser("score", help="Score transactions for fraud")
    score_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to transactions CSV to score",
    )
    score_parser.add_argument(
        "--model",
        type=str,
        default="model_output",
        help="Path to trained model directory",
    )
    score_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Alert threshold (default: 0.5)",
    )

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic transaction data")
    gen_parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Number of transactions to generate (default: 10000)",
    )
    gen_parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.02,
        help="Fraud rate (default: 0.02)",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default="data/generated_transactions.csv",
        help="Output CSV path",
    )

    # --- visualize ---
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to scored/labeled CSV",
    )
    viz_parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Directory for plot output (default: plots)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "train":
            return _cmd_train(args)
        elif args.command == "score":
            return _cmd_score(args)
        elif args.command == "generate":
            return _cmd_generate(args)
        elif args.command == "visualize":
            return _cmd_visualize(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train the model on labeled transaction data."""
    print(f"Loading training data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df):,} transactions loaded ({df['is_fraud'].sum():,} fraud)")

    # Preprocess
    print("Preprocessing...")
    preprocessor = TransactionPreprocessor()
    df_processed = preprocessor.fit_transform(df)

    # Feature engineering
    print("Engineering features...")
    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df_processed)

    # Determine feature columns
    feature_cols = (
        preprocessor.get_feature_columns()
        + engineer.get_feature_columns()
    )
    feature_cols = [c for c in feature_cols if c in df_featured.columns]

    # Train
    print(f"Training model with {len(feature_cols)} features...")
    model = FraudModel()
    metrics = model.train(df_featured, feature_cols, target_column="is_fraud")

    print("\n" + metrics.summary())

    # Save
    model.save(args.output)
    print(f"\nModel saved to {args.output}/")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Score transactions using a trained model."""
    print(f"Loading model from {args.model}...")
    model = FraudModel()
    model.load(args.model)

    print(f"Loading transactions from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df):,} transactions to score")

    preprocessor = TransactionPreprocessor()
    engineer = FeatureEngineer()

    # For scoring we need to fit on the data (in production you'd
    # persist and load the fitted preprocessor/engineer too)
    df_processed = preprocessor.fit_transform(df)
    df_featured = engineer.fit_transform(df_processed)

    detector = FraudDetector(model, preprocessor, engineer)
    results = detector.score_batch(df)

    # Alerts
    alert_system = AlertSystem(alert_threshold=args.threshold)
    alerts = alert_system.process_results(results)

    print(f"\nScored {len(results):,} transactions")
    print(f"Alerts generated: {len(alerts)}")
    stats = detector.get_statistics()
    print(f"Risk distribution: {stats.get('risk_distribution', {})}")

    if alerts:
        print(f"\n{alert_system.generate_report()}")

    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate synthetic transaction data."""
    # Import here to avoid circular dependency
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.generate_dataset import generate_transactions

    print(f"Generating {args.rows:,} transactions (fraud rate: {args.fraud_rate:.1%})...")
    df = generate_transactions(n_rows=args.rows, fraud_rate=args.fraud_rate)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"  Total: {len(df):,} | Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.1%})")
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    """Generate visualizations from scored data."""
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    visualizer = FraudVisualizer(output_dir=args.output)

    if "is_fraud" not in df.columns:
        print("Warning: no 'is_fraud' column found. Limited visualizations available.")

    # Run full pipeline to get metrics for visualization
    preprocessor = TransactionPreprocessor()
    df_processed = preprocessor.fit_transform(df)

    engineer = FeatureEngineer()
    df_featured = engineer.fit_transform(df_processed)

    feature_cols = preprocessor.get_feature_columns() + engineer.get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df_featured.columns]

    if "is_fraud" in df_featured.columns:
        import numpy as np

        model = FraudModel()
        metrics = model.train(df_featured, feature_cols)

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
        print(f"Generated {len(saved)} visualizations in {args.output}/")
        for p in saved:
            print(f"  {p}")
    else:
        print("Cannot generate full visualizations without labels.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
