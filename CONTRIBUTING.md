# Contributing to Fraud Detection System

Thank you for your interest in contributing. This guide covers setting up the development environment, running tests, and submitting changes.

## Development Setup

### Prerequisites

- Python 3.10 or later
- pip package manager
- Git

### Clone and Install

```bash
git clone https://github.com/stabrea/fraud-detection-system.git
cd fraud-detection-system

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Verify Setup

```bash
# Run the demo pipeline with synthetic data
python main.py --rows 1000
```

## Running Tests

Tests live in the `tests/` directory and use `pytest`.

```bash
# Install test dependencies
pip install pytest

# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_detector.py -v

# Run with verbose output and full tracebacks
pytest tests/ -v --tb=long
```

All tests must pass before submitting a pull request.

## Code Style

- **Type hints**: All functions must have complete type annotations. Use `from __future__ import annotations` for modern syntax.
- **Docstrings**: Every public class and method needs a docstring. Follow the existing NumPy/Google style used throughout the codebase.
- **Dataclasses**: Use `@dataclass` for structured data. Include a `to_dict()` method for serialization when the data will be output.
- **Imports**: Group imports in the standard order -- stdlib, third-party, local -- separated by blank lines.
- **Decimal precision**: Financial calculations should use `Decimal` or `numpy` with explicit rounding, not bare `float` arithmetic.
- **No `Any` types**: Avoid `typing.Any`. Use specific types or generics.

## Submitting Changes

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** in small, focused commits. Each commit should do one thing.

3. **Run the test suite** and confirm all tests pass:
   ```bash
   pytest tests/ -v
   ```

4. **Test the full pipeline** to make sure nothing is broken end-to-end:
   ```bash
   python main.py --rows 1000
   ```

5. **Push** your branch and open a pull request against `main`.

6. In your PR description, explain:
   - What the change does
   - Why it is needed
   - How you tested it

## Project Structure

```
fraud_detector/
    __init__.py              # Public API exports
    preprocessor.py          # Data cleaning and normalization
    feature_engineer.py      # Behavioral feature generation
    model.py                 # Random Forest + Isolation Forest training
    detector.py              # Real-time scoring engine
    alert_system.py          # Alert generation and tracking
    visualizer.py            # Matplotlib visualizations
    cli.py                   # Command-line interface
tests/
    test_preprocessor.py
    test_feature_engineer.py
    test_model.py
    test_detector.py
data/
    generate_dataset.py      # Synthetic data generator
    sample_transactions.csv  # Demo dataset
```

## Areas for Contribution

- Deep learning models (LSTM/Transformer) for sequential pattern detection
- SHAP-based explainability for per-transaction feature attribution
- REST API service for production deployment
- Data/concept drift detection in production scoring
- Additional test coverage for edge cases

## Questions

Open an issue if you have questions or want to discuss a feature before starting work.
