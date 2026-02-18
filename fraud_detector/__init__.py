"""
Fraud Detection System
=====================

An ML-based financial fraud detection pipeline combining Random Forest
classification with Isolation Forest anomaly detection.

Designed at the intersection of cybersecurity and finance.

Supports both synthetic data and real-world datasets including the
ULB Credit Card Fraud dataset from Kaggle.

Author: Taofik Bishi
"""

__version__ = "2.0.0"
__author__ = "Taofik Bishi"

from fraud_detector.preprocessor import TransactionPreprocessor
from fraud_detector.feature_engineer import FeatureEngineer
from fraud_detector.model import FraudModel
from fraud_detector.detector import FraudDetector
from fraud_detector.alert_system import AlertSystem
from fraud_detector.visualizer import FraudVisualizer
from fraud_detector.data_loader import load_dataset, load_ulb_credit_card
from fraud_detector.ulb_preprocessor import ULBPreprocessor

__all__ = [
    "TransactionPreprocessor",
    "FeatureEngineer",
    "FraudModel",
    "FraudDetector",
    "AlertSystem",
    "FraudVisualizer",
    "ULBPreprocessor",
    "load_dataset",
    "load_ulb_credit_card",
]
