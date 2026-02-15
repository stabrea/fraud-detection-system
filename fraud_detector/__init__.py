"""
Fraud Detection System
=====================

An ML-based financial fraud detection pipeline combining Random Forest
classification with Isolation Forest anomaly detection.

Designed at the intersection of cybersecurity and finance.

Author: Taofik Bishi
"""

__version__ = "1.0.0"
__author__ = "Taofik Bishi"

from fraud_detector.preprocessor import TransactionPreprocessor
from fraud_detector.feature_engineer import FeatureEngineer
from fraud_detector.model import FraudModel
from fraud_detector.detector import FraudDetector
from fraud_detector.alert_system import AlertSystem
from fraud_detector.visualizer import FraudVisualizer

__all__ = [
    "TransactionPreprocessor",
    "FeatureEngineer",
    "FraudModel",
    "FraudDetector",
    "AlertSystem",
    "FraudVisualizer",
]
