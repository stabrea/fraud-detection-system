"""Package setup for fraud-detection-system."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="fraud-detection-system",
    version="2.0.0",
    author="Taofik Bishi",
    author_email="taofik.bishi@example.com",
    description="ML-based financial fraud detection system combining supervised and unsupervised learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taofikbishi/fraud-detection-system",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fraud-detector=fraud_detector.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    keywords="fraud detection machine-learning finance cybersecurity anomaly-detection",
)
