# Fraud Detection System

An end-to-end ML pipeline for detecting fraudulent financial transactions in real time. Built at the intersection of **cybersecurity** and **finance**, this system combines supervised classification (Random Forest) with unsupervised anomaly detection (Isolation Forest) to identify fraud patterns that neither approach catches alone.

## How It Works

```
Raw Transactions
       |
       v
+------------------+     +--------------------+     +---------------+
|  Preprocessing   | --> | Feature Engineering| --> | Model Training|
|  - Missing vals  |     |  - Velocity checks |     |  - Random     |
|  - Normalization |     |  - Geo anomalies   |     |    Forest     |
|  - Encoding      |     |  - Amount patterns |     |  - Isolation  |
|  - Time features |     |  - Time-of-day     |     |    Forest     |
+------------------+     +--------------------+     +---------------+
                                                           |
                    +--------------------------------------+
                    |
                    v
          +------------------+     +----------------+
          | Fraud Scoring    | --> | Alert System   |
          |  - RF probability|     |  - Risk levels |
          |  - IF anomaly    |     |  - Reports     |
          |  - Hybrid score  |     |  - FP tracking |
          +------------------+     +----------------+
```

## Features

- **Hybrid Detection Model** — Blends Random Forest (supervised) with Isolation Forest (unsupervised) for a combined fraud score that catches both known and novel fraud patterns
- **Advanced Feature Engineering** — 24+ engineered features including velocity checks, geographic anomaly detection, cyclical time encoding, and per-customer behavioral baselines
- **Real-Time Scoring** — Score individual transactions or batches with sub-second latency; returns risk level (low/medium/high/critical) with explanations
- **Alert Management** — Automatic alert generation with threshold tuning, analyst feedback tracking, and false positive rate monitoring
- **Synthetic Data Generator** — Produces realistic transaction datasets with configurable fraud rates and five distinct fraud patterns (high amount, foreign location, unusual time, rapid-fire, category anomaly)
- **Visualization Suite** — Publication-quality plots: confusion matrix, ROC curve, feature importance, fraud patterns, and score distributions
- **CLI Interface** — Train, score, generate data, and visualize from the command line

## Model Performance

Typical results on synthetic data (10,000 transactions, 2% fraud rate):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | >0.98  |
| Precision | >0.85  |
| Recall    | >0.80  |
| F1 Score  | >0.82  |
| ROC AUC   | >0.97  |

*Actual metrics depend on data characteristics and hyperparameters. Run the pipeline to see exact numbers.*

## Installation

```bash
# Clone the repository
git clone https://github.com/taofikbishi/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

## Usage

### Quick Demo (Recommended)

Run the full pipeline with synthetic data:

```bash
python main.py
```

With visualization output:

```bash
python main.py --visualize --output output
```

Using the included sample dataset:

```bash
python main.py --data data/sample_transactions.csv
```

### CLI Commands

```bash
# Generate synthetic data
fraud-detector generate --rows 10000 --fraud-rate 0.02 --output data/transactions.csv

# Train model
fraud-detector train --data data/transactions.csv --output model_output

# Score new transactions
fraud-detector score --data data/new_transactions.csv --model model_output --threshold 0.5

# Generate visualizations
fraud-detector visualize --data data/transactions.csv --output plots
```

### Python API

```python
from fraud_detector import (
    TransactionPreprocessor,
    FeatureEngineer,
    FraudModel,
    FraudDetector,
    AlertSystem,
)

# Preprocess
preprocessor = TransactionPreprocessor()
df_processed = preprocessor.fit_transform(raw_df)

# Engineer features
engineer = FeatureEngineer()
df_featured = engineer.fit_transform(df_processed)

# Train
feature_cols = preprocessor.get_feature_columns() + engineer.get_feature_columns()
model = FraudModel(n_estimators=200)
metrics = model.train(df_featured, feature_cols)
print(metrics.summary())

# Score in real time
detector = FraudDetector(model, preprocessor, engineer)
result = detector.score_transaction({
    "amount": 5000.00,
    "timestamp": "2024-06-15T03:22:00",
    "merchant_category": "wire_transfer",
    "transaction_type": "transfer",
    "channel": "online",
    "location": "Foreign-A",
    "customer_id": "CUST-00042",
})
print(f"Risk: {result.risk_level.value} (score: {result.hybrid_score:.4f})")
print(f"Factors: {result.contributing_factors}")
```

## Architecture

```
fraud-detection-system/
|
|-- fraud_detector/              # Core package
|   |-- __init__.py              # Public API exports
|   |-- preprocessor.py          # Data cleaning & normalization
|   |-- feature_engineer.py      # Behavioral feature generation
|   |-- model.py                 # RF + Isolation Forest training
|   |-- detector.py              # Real-time scoring engine
|   |-- alert_system.py          # Alert generation & tracking
|   |-- visualizer.py            # Matplotlib visualizations
|   |-- cli.py                   # Command-line interface
|
|-- data/
|   |-- generate_dataset.py      # Synthetic data generator
|   |-- sample_transactions.csv  # 1,000-row demo dataset
|
|-- main.py                      # End-to-end demo pipeline
|-- requirements.txt
|-- setup.py
|-- LICENSE
|-- README.md
```

### Key Design Decisions

- **Hybrid scoring** — Random Forest handles known patterns; Isolation Forest catches novel anomalies. The weighted blend (70/30 default) outperforms either model alone.
- **Balanced class weights** — `class_weight="balanced"` in Random Forest compensates for the extreme class imbalance (~2% fraud rate) without needing SMOTE or other resampling.
- **Cyclical time encoding** — Hours are encoded as sin/cos pairs so the model understands that 23:00 and 01:00 are close together.
- **Per-customer baselines** — Amount deviation and transaction frequency are computed relative to each customer's history, not global averages.

## Future Improvements

- [ ] **Deep learning** — LSTM/Transformer model for sequential transaction pattern detection
- [ ] **Graph features** — Network analysis of merchant-customer transaction graphs
- [ ] **Online learning** — Incremental model updates as new labeled data arrives
- [ ] **REST API** — FastAPI service for production deployment with async scoring
- [ ] **Dashboard** — Real-time monitoring dashboard with Streamlit or Dash
- [ ] **Explainability** — SHAP values for per-transaction feature attribution
- [ ] **Drift detection** — Monitor for data/concept drift in production scoring

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built by [Taofik Bishi](https://github.com/taofikbishi) — Finance & Cybersecurity*
