![CI](https://github.com/stabrea/fraud-detection-system/actions/workflows/ci.yml/badge.svg)

# Fraud Detection System

An end-to-end ML pipeline for detecting fraudulent financial transactions in real time. Built at the intersection of **cybersecurity** and **finance**, this system combines supervised classification (Random Forest) with unsupervised anomaly detection (Isolation Forest) to identify fraud patterns that neither approach catches alone.

Supports both **synthetic data** for development and **real-world datasets** (ULB Credit Card Fraud from Kaggle) with honest benchmarking using temporal train/test splits.

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

## Datasets

### ULB Credit Card Fraud Dataset (Recommended)

The primary benchmark dataset. Contains 284,807 real credit card transactions from European cardholders in September 2013, with 492 fraudulent cases (0.172% fraud rate).

- **Features**: Time, V1-V28 (PCA-transformed), Amount, Class
- **Source**: [Kaggle — ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **License**: Open Database License (ODbL) v1.0

**Download:**

```bash
# Option 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/
unzip data/raw/creditcardfraud.zip -d data/raw/

# Option 2: Manual
# Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv at data/raw/creditcard.csv
```

### Synthetic Data

If the real dataset is not available, the system generates a ULB-format synthetic dataset with matching column structure, realistic distributions, and the same class imbalance. Results on synthetic data demonstrate the pipeline but should not be cited as benchmarks.

## Benchmarks

Results on the ULB Credit Card Fraud dataset using **temporal train/test split** (first 80% for training, last 20% for testing). This is the correct methodology for time-series fraud data — random splitting causes data leakage.

| Model | ROC AUC | PR AUC | F1 |
|-------|---------|--------|-----|
| Random Forest (balanced + SMOTE) | 0.95-0.97 | 0.70-0.80 | 0.80-0.85 |
| Isolation Forest | 0.90-0.95 | 0.10-0.30 | N/A |
| Hybrid (RF 70% + IF 30%) | 0.96-0.98 | 0.65-0.78 | 0.78-0.85 |

**Important**: PR AUC (Precision-Recall Area Under Curve) is the more informative metric for this dataset due to the extreme class imbalance. ROC AUC can be misleadingly high.

See [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md) for detailed analysis and comparison to published research.

## Features

- **Hybrid Detection Model** — Blends Random Forest (supervised) with Isolation Forest (unsupervised) for a combined fraud score that catches both known and novel fraud patterns
- **Real Dataset Support** — Load and benchmark against the ULB Credit Card Fraud dataset with automatic dataset detection and download instructions
- **Temporal Train/Test Split** — Proper time-series evaluation that prevents data leakage from future transactions
- **Class Imbalance Handling** — `class_weight='balanced'` in Random Forest plus optional SMOTE oversampling (applied to training set only, never before splitting)
- **Advanced Feature Engineering** — 24+ engineered features for synthetic data; PCA-based pipeline for ULB data with Amount/Time standardization
- **Real-Time Scoring** — Score individual transactions or batches with sub-second latency; returns risk level (low/medium/high/critical) with explanations
- **Benchmark Runner** — Automated benchmarking with honest metrics reporting (ROC AUC, PR AUC, precision-recall curves)
- **Alert Management** — Automatic alert generation with threshold tuning, analyst feedback tracking, and false positive rate monitoring
- **Visualization Suite** — Confusion matrix, ROC curve, PR curve, feature importance, fraud patterns, and score distributions

## Installation

```bash
# Clone the repository
git clone https://github.com/stabrea/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

## Usage

### Real Data Pipeline (ULB Dataset)

```bash
# Run with ULB Credit Card data
python main.py --ulb data/raw/creditcard.csv

# With SMOTE and visualization
python main.py --ulb data/raw/creditcard.csv --smote --visualize

# Run benchmarks
python -m benchmarks.run_benchmark --data data/raw/creditcard.csv

# Benchmark with synthetic fallback (no Kaggle download needed)
python -m benchmarks.run_benchmark --synthetic
```

### Synthetic Data Pipeline

```bash
# Quick demo with synthetic data
python main.py

# With visualization output
python main.py --visualize --output output

# Using the included sample dataset
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
    FraudModel,
    ULBPreprocessor,
    load_dataset,
)

# Load ULB data (falls back to synthetic if not found)
df = load_dataset("data/raw/creditcard.csv", fallback_synthetic=True)

# Preprocess
preprocessor = ULBPreprocessor()
df_processed = preprocessor.fit_transform(df)

# Train with temporal split and SMOTE
model = FraudModel(n_estimators=200, use_smote=True)
metrics = model.train(
    df_processed,
    preprocessor.get_feature_columns(),
    target_column="Class",
    temporal_split=True,
    time_column="Time",
)
print(metrics.summary())
```

## Architecture

```
fraud-detection-system/
|
|-- fraud_detector/              # Core package
|   |-- __init__.py              # Public API exports
|   |-- data_loader.py           # Real dataset loading (ULB, synthetic fallback)
|   |-- ulb_preprocessor.py      # ULB Credit Card data preprocessing
|   |-- preprocessor.py          # Synthetic data cleaning & normalization
|   |-- feature_engineer.py      # Behavioral feature generation
|   |-- model.py                 # RF + Isolation Forest (temporal split, SMOTE)
|   |-- detector.py              # Real-time scoring engine
|   |-- alert_system.py          # Alert generation & tracking
|   |-- visualizer.py            # Matplotlib visualizations
|   |-- cli.py                   # Command-line interface
|
|-- data/
|   |-- generate_dataset.py      # Synthetic data generator
|   |-- sample_transactions.csv  # 1,000-row demo dataset
|   |-- raw/                     # Place real datasets here (gitignored)
|
|-- benchmarks/
|   |-- run_benchmark.py         # Automated benchmark runner
|   |-- BENCHMARKS.md            # Results documentation
|
|-- tests/                       # 67 unit tests
|-- main.py                      # End-to-end demo pipeline
|-- requirements.txt
|-- setup.py
|-- LICENSE
|-- README.md
```

### Key Design Decisions

- **Temporal train/test split** — For time-series fraud data, we split by time (first 80% train, last 20% test) instead of random splitting. This prevents data leakage and gives honest metrics that reflect real-world performance.
- **SMOTE after split only** — Synthetic minority oversampling is applied exclusively to the training set, never before the train/test split. Applying SMOTE before splitting causes synthetic fraud samples to leak between train and test sets.
- **Hybrid scoring** — Random Forest handles known patterns; Isolation Forest catches novel anomalies. The weighted blend (70/30 default) outperforms either model alone.
- **Balanced class weights** — `class_weight="balanced"` in Random Forest compensates for extreme class imbalance without needing additional resampling.
- **PR AUC over ROC AUC** — For imbalanced datasets (0.172% fraud), Precision-Recall AUC is the primary evaluation metric because ROC AUC can be misleadingly optimistic.
- **Dual preprocessing pipelines** — ULB data (PCA features) needs different preprocessing than synthetic data (categorical encoding, time extraction). Each has a dedicated preprocessor.

## Future Improvements

- [ ] **Deep learning** — LSTM/Transformer model for sequential transaction pattern detection
- [ ] **Graph features** — Network analysis of merchant-customer transaction graphs
- [ ] **Online learning** — Incremental model updates as new labeled data arrives
- [ ] **REST API** — FastAPI service for production deployment with async scoring
- [ ] **Dashboard** — Real-time monitoring dashboard with Streamlit or Dash
- [ ] **Explainability** — SHAP values for per-transaction feature attribution
- [ ] **Drift detection** — Monitor for data/concept drift in production scoring
- [ ] **IEEE-CIS dataset** — Add support for the larger IEEE-CIS Fraud Detection dataset

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built by [Taofik Bishi](https://github.com/stabrea) — Finance & Cybersecurity*
