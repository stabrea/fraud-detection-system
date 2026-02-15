"""
Synthetic transaction data generator.

Produces realistic financial transaction datasets with controlled
fraud rates and known fraud patterns for model training and testing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "gas_station", "online_retail",
    "travel", "entertainment", "healthcare", "utilities", "clothing",
    "jewelry", "cash_advance", "wire_transfer", "cryptocurrency",
]

TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "refund", "payment"]

CHANNELS = ["pos", "online", "atm", "mobile_app", "phone"]

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
    "Miami", "Atlanta", "Denver", "Seattle", "Boston",
    "Foreign-A", "Foreign-B", "Foreign-C",
]

# Amount distributions by category (mean, std)
AMOUNT_PROFILES: dict[str, tuple[float, float]] = {
    "grocery": (65.0, 40.0),
    "electronics": (350.0, 250.0),
    "restaurant": (45.0, 30.0),
    "gas_station": (40.0, 20.0),
    "online_retail": (120.0, 100.0),
    "travel": (500.0, 400.0),
    "entertainment": (80.0, 60.0),
    "healthcare": (200.0, 150.0),
    "utilities": (150.0, 50.0),
    "clothing": (90.0, 70.0),
    "jewelry": (800.0, 600.0),
    "cash_advance": (300.0, 200.0),
    "wire_transfer": (1000.0, 800.0),
    "cryptocurrency": (500.0, 500.0),
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_transactions(
    n_rows: int = 10000,
    fraud_rate: float = 0.02,
    n_customers: int = 500,
    seed: int = 42,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset.

    Args:
        n_rows: Total number of transactions.
        fraud_rate: Proportion of fraudulent transactions.
        n_customers: Number of unique customers.
        seed: Random seed for reproducibility.
        start_date: Start of the date range.
        end_date: End of the date range.

    Returns:
        DataFrame with transaction records and ``is_fraud`` labels.
    """
    rng = np.random.default_rng(seed)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_seconds = int((end - start).total_seconds())

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # Assign each customer a "home" location and typical categories
    customer_ids = [f"CUST-{i:05d}" for i in range(n_customers)]
    customer_home = {
        cid: rng.choice(LOCATIONS[:15])  # domestic only for home
        for cid in customer_ids
    }
    customer_categories = {
        cid: rng.choice(MERCHANT_CATEGORIES[:10], size=rng.integers(2, 6), replace=False).tolist()
        for cid in customer_ids
    }

    records: list[dict] = []

    # --- Legitimate transactions ---
    for _ in range(n_legit):
        cid = rng.choice(customer_ids)
        category = rng.choice(customer_categories[cid])
        mean_amt, std_amt = AMOUNT_PROFILES[category]
        amount = max(0.50, rng.normal(mean_amt, std_amt))

        # Mostly during business/evening hours
        hour = int(rng.choice(
            range(24),
            p=_legitimate_hour_distribution(),
        ))
        ts = start + timedelta(
            seconds=int(rng.integers(0, total_seconds)),
        )
        ts = ts.replace(hour=hour, minute=int(rng.integers(0, 60)))

        # Mostly from home location
        location = (
            customer_home[cid]
            if rng.random() < 0.85
            else rng.choice(LOCATIONS[:15])
        )

        channel = rng.choice(CHANNELS, p=[0.30, 0.30, 0.10, 0.25, 0.05])

        records.append({
            "customer_id": cid,
            "amount": round(amount, 2),
            "timestamp": ts.isoformat(),
            "merchant_category": category,
            "transaction_type": rng.choice(["purchase", "payment", "purchase", "purchase"]),
            "channel": channel,
            "location": location,
            "is_fraud": 0,
        })

    # --- Fraudulent transactions ---
    for _ in range(n_fraud):
        cid = rng.choice(customer_ids)
        fraud_pattern = rng.choice([
            "high_amount", "foreign_location", "unusual_time",
            "rapid_fire", "category_anomaly",
        ])

        category, amount, hour, location, channel, tx_type = _generate_fraud_pattern(
            rng, fraud_pattern, cid, customer_home, customer_categories
        )

        ts = start + timedelta(
            seconds=int(rng.integers(0, total_seconds)),
        )
        ts = ts.replace(hour=hour, minute=int(rng.integers(0, 60)))

        records.append({
            "customer_id": cid,
            "amount": round(amount, 2),
            "timestamp": ts.isoformat(),
            "merchant_category": category,
            "transaction_type": tx_type,
            "channel": channel,
            "location": location,
            "is_fraud": 1,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN-{i:07d}" for i in range(len(df))])
    return df


def _legitimate_hour_distribution() -> list[float]:
    """Probability distribution for legitimate transaction hours."""
    # Low activity 0-6, peak 9-21, taper off after
    weights = [
        1, 1, 1, 1, 1, 1,      # 0-5: very low
        2, 4, 7, 10, 10, 10,   # 6-11: morning ramp
        12, 12, 11, 10, 10, 11, # 12-17: afternoon
        12, 10, 8, 5, 3, 2,    # 18-23: evening decline
    ]
    total = sum(weights)
    return [w / total for w in weights]


def _generate_fraud_pattern(
    rng: np.random.Generator,
    pattern: str,
    customer_id: str,
    customer_home: dict[str, str],
    customer_categories: dict[str, list[str]],
) -> tuple[str, float, int, str, str, str]:
    """Generate transaction attributes for a specific fraud pattern.

    Returns:
        Tuple of (category, amount, hour, location, channel, tx_type).
    """
    if pattern == "high_amount":
        # Abnormally high amounts, often in suspicious categories
        category = rng.choice(["jewelry", "electronics", "wire_transfer", "cryptocurrency"])
        mean_amt, _ = AMOUNT_PROFILES[category]
        amount = max(100, rng.normal(mean_amt * 4, mean_amt))
        hour = int(rng.integers(0, 24))
        location = customer_home.get(customer_id, "New York")
        channel = rng.choice(["online", "mobile_app"])
        tx_type = "purchase"

    elif pattern == "foreign_location":
        # Transaction from unusual foreign location
        category = rng.choice(customer_categories.get(customer_id, ["online_retail"]))
        mean_amt, std_amt = AMOUNT_PROFILES.get(category, (200, 100))
        amount = max(10, rng.normal(mean_amt * 1.5, std_amt))
        hour = int(rng.integers(0, 24))
        location = rng.choice(["Foreign-A", "Foreign-B", "Foreign-C"])
        channel = rng.choice(["pos", "atm"])
        tx_type = rng.choice(["purchase", "withdrawal"])

    elif pattern == "unusual_time":
        # Transactions at odd hours (2-5 AM)
        category = rng.choice(["cash_advance", "wire_transfer", "online_retail"])
        mean_amt, std_amt = AMOUNT_PROFILES[category]
        amount = max(50, rng.normal(mean_amt * 2, std_amt))
        hour = int(rng.integers(1, 5))
        location = customer_home.get(customer_id, "New York")
        channel = rng.choice(["online", "atm", "mobile_app"])
        tx_type = rng.choice(["withdrawal", "transfer"])

    elif pattern == "rapid_fire":
        # Multiple small-to-medium transactions (card testing)
        category = rng.choice(["gas_station", "online_retail", "entertainment"])
        amount = round(rng.uniform(5, 100), 2)
        hour = int(rng.integers(0, 24))
        location = rng.choice(LOCATIONS)
        channel = "online"
        tx_type = "purchase"

    elif pattern == "category_anomaly":
        # Purchases in categories the customer never uses
        unusual_cats = [
            c for c in MERCHANT_CATEGORIES
            if c not in customer_categories.get(customer_id, [])
        ]
        category = rng.choice(unusual_cats) if unusual_cats else "cryptocurrency"
        mean_amt, std_amt = AMOUNT_PROFILES.get(category, (300, 200))
        amount = max(20, rng.normal(mean_amt * 2, std_amt))
        hour = int(rng.integers(0, 24))
        location = rng.choice(LOCATIONS)
        channel = rng.choice(["online", "mobile_app"])
        tx_type = "purchase"

    else:
        # Fallback
        category = "online_retail"
        amount = 500.0
        hour = 3
        location = "Foreign-A"
        channel = "online"
        tx_type = "purchase"

    return category, amount, hour, location, channel, tx_type


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate dataset from command line."""
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows")
    parser.add_argument("--fraud-rate", type=float, default=0.02, help="Fraud rate")
    parser.add_argument("--output", type=str, default="data/transactions.csv", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_transactions(
        n_rows=args.rows,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} transactions -> {output_path}")
    print(f"  Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.1%})")


if __name__ == "__main__":
    main()
