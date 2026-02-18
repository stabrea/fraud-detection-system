"""
Data loading module for real-world fraud detection datasets.

Supports the ULB Credit Card Fraud dataset (Kaggle) and generates
a ULB-format synthetic dataset for development/testing when the
real data is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Expected ULB Credit Card Fraud dataset schema
ULB_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
ULB_N_FEATURES = 28  # V1-V28 are PCA-transformed features


def load_ulb_credit_card(
    path: str | Path = "data/raw/creditcard.csv",
) -> Optional[pd.DataFrame]:
    """Load the ULB Credit Card Fraud dataset.

    The dataset contains 284,807 transactions made by European
    cardholders in September 2013.  Features V1-V28 are PCA
    components; Time and Amount are the only non-transformed features.
    Class is the label (1 = fraud, 0 = legitimate).

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with the ULB dataset, or ``None`` if file not found.
    """
    path = Path(path)
    if not path.exists():
        print(f"[data_loader] ULB dataset not found at: {path}")
        print_download_instructions()
        return None

    df = pd.read_csv(path)
    if not _validate_ulb_schema(df):
        print(f"[data_loader] File at {path} does not match ULB schema.")
        return None

    print(f"[data_loader] Loaded ULB Credit Card dataset: {len(df):,} transactions")
    print(f"  Fraud: {df['Class'].sum():,} ({df['Class'].mean():.3%})")
    print(f"  Legitimate: {(df['Class'] == 0).sum():,}")
    return df


def load_dataset(
    path: str | Path = "data/raw/creditcard.csv",
    fallback_synthetic: bool = True,
    synthetic_n: int = 284807,
) -> pd.DataFrame:
    """Load a fraud dataset, falling back to synthetic if real data unavailable.

    Tries to load the ULB Credit Card Fraud dataset.  If not found and
    ``fallback_synthetic`` is True, generates a synthetic dataset that
    matches the ULB format.

    Args:
        path: Path to the real dataset CSV.
        fallback_synthetic: Whether to generate synthetic data if real
            data is not found.
        synthetic_n: Number of rows for synthetic fallback.

    Returns:
        DataFrame with columns matching the ULB schema.
    """
    df = load_ulb_credit_card(path)
    if df is not None:
        return df

    if fallback_synthetic:
        print("[data_loader] Generating ULB-format synthetic dataset...")
        print("  NOTE: This is synthetic data for pipeline testing.")
        print("  Download the real dataset for meaningful benchmarks.")
        return generate_ulb_format_synthetic(n_rows=synthetic_n)

    raise FileNotFoundError(
        f"Dataset not found at {path}. Run with fallback_synthetic=True "
        "or download the ULB Credit Card Fraud dataset."
    )


def detect_dataset_type(path: str | Path) -> str:
    """Detect the type of fraud dataset from file structure.

    Args:
        path: Path to a CSV file.

    Returns:
        One of: ``"ulb_credit_card"``, ``"synthetic_original"``, ``"unknown"``.
    """
    path = Path(path)
    if not path.exists():
        return "unknown"

    df = pd.read_csv(path, nrows=5)

    if _validate_ulb_schema(df):
        return "ulb_credit_card"

    if "is_fraud" in df.columns and "customer_id" in df.columns:
        return "synthetic_original"

    return "unknown"


def generate_ulb_format_synthetic(
    n_rows: int = 284807,
    fraud_rate: float = 0.00172,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset matching the ULB Credit Card format.

    Produces data with the same column structure, similar distributions,
    and realistic class imbalance.  Clearly labeled as synthetic.

    Args:
        n_rows: Total number of transactions.
        fraud_rate: Fraction of fraudulent transactions (default matches
            the real ULB rate of 0.172%).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ULB-format columns: Time, V1-V28, Amount, Class.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 10)
    n_legit = n_rows - n_fraud

    # Build all data as numpy arrays first (memory efficient)
    # Time: seconds elapsed from first transaction (2 days span)
    time_all = rng.uniform(0, 172792, size=n_rows)

    # V1-V28: PCA components
    v_all = rng.normal(0, 1, size=(n_rows, ULB_N_FEATURES))

    # Amount: log-normal distribution
    amount_all = np.abs(rng.lognormal(mean=3.5, sigma=1.5, size=n_rows))

    # Class labels: first n_legit are 0, rest are 1
    class_all = np.zeros(n_rows, dtype=np.int32)
    class_all[n_legit:] = 1

    # Apply fraud-specific feature shifts to the fraud rows
    fraud_slice = slice(n_legit, None)
    v_all[fraud_slice, 0] -= 2.0   # V1 shifts negative for fraud
    v_all[fraud_slice, 1] += 1.5   # V2 shifts positive
    v_all[fraud_slice, 2] -= 2.5   # V3 shifts negative
    v_all[fraud_slice, 3] += 1.0   # V4 shifts positive
    v_all[fraud_slice, 4] -= 1.5   # V5 shifts negative
    v_all[fraud_slice, 13] -= 3.0  # V14
    v_all[fraud_slice, 16] -= 2.0  # V17
    # Fraud amounts slightly higher
    amount_all[fraud_slice] = np.abs(
        rng.lognormal(mean=4.0, sigma=2.0, size=n_fraud)
    )

    # Sort by time (using numpy argsort, then reorder all arrays)
    sort_idx = np.argsort(time_all)
    time_all = time_all[sort_idx]
    v_all = v_all[sort_idx]
    amount_all = amount_all[sort_idx]
    class_all = class_all[sort_idx]

    # Build DataFrame from pre-sorted arrays
    data = {"Time": time_all}
    for i in range(ULB_N_FEATURES):
        data[f"V{i + 1}"] = v_all[:, i]
    data["Amount"] = np.round(amount_all, 2)
    data["Class"] = class_all

    df = pd.DataFrame(data, columns=ULB_COLUMNS)

    print(f"[data_loader] Generated ULB-format synthetic: {len(df):,} rows")
    print(f"  Fraud: {df['Class'].sum():,} ({df['Class'].mean():.3%})")
    return df


def print_download_instructions() -> None:
    """Print instructions for downloading the ULB Credit Card Fraud dataset."""
    print(
        "\n"
        "=" * 60 + "\n"
        "  Download Instructions: ULB Credit Card Fraud Dataset\n"
        "=" * 60 + "\n"
        "\n"
        "Option 1 — Kaggle CLI:\n"
        "  pip install kaggle\n"
        "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/\n"
        "  unzip data/raw/creditcardfraud.zip -d data/raw/\n"
        "\n"
        "Option 2 — Manual download:\n"
        "  1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "  2. Download creditcard.csv\n"
        "  3. Place it at: data/raw/creditcard.csv\n"
        "\n"
        "Dataset details:\n"
        "  - 284,807 transactions (492 fraudulent, 0.172%)\n"
        "  - Features: Time, V1-V28 (PCA), Amount, Class\n"
        "  - Source: ULB Machine Learning Group\n"
        "  - License: Open Database License (ODbL) v1.0\n"
        "=" * 60 + "\n"
    )


def _validate_ulb_schema(df: pd.DataFrame) -> bool:
    """Check whether a DataFrame matches the ULB dataset schema."""
    required = {"Time", "Amount", "Class"}
    if not required.issubset(set(df.columns)):
        return False
    # Check for at least V1-V10
    for i in range(1, 11):
        if f"V{i}" not in df.columns:
            return False
    return True
