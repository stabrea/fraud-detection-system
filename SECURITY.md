# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

Only the latest release on the `master` branch receives security updates.

## Reporting a Vulnerability

If you discover a security vulnerability in the Fraud Detection System, please report it responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

**Email:** [bishitaofik@gmail.com](mailto:bishitaofik@gmail.com)

Include in your report:

- A description of the vulnerability and its potential impact
- The affected component (data pipeline, model training, scoring engine, alert system, CLI, etc.)
- Steps to reproduce the issue
- Any proof-of-concept code or output, if available

## Response Timeline

| Action                     | Timeframe       |
|----------------------------|-----------------|
| Acknowledgment of report   | 48 hours        |
| Initial assessment         | 5 business days |
| Patch or mitigation issued | 30 days         |
| Public disclosure           | After patch     |

We will coordinate disclosure timing with the reporter. Credit will be given unless anonymity is requested.

## Scope

The following are **in scope** for security reports:

- Data poisoning vectors that could degrade model accuracy or bypass fraud detection
- Injection attacks through transaction input fields (CSV parsing, CLI arguments)
- Sensitive data exposure in logs, reports, or generated visualizations
- Path traversal or arbitrary file access via CLI file parameters
- Model serialization/deserialization vulnerabilities (pickle, joblib)
- Dependencies with known CVEs that affect this system
- Adversarial evasion techniques that reliably bypass the detection model

The following are **out of scope**:

- Issues requiring physical access to the host machine
- General ML model accuracy concerns (use GitHub Issues for those)
- Vulnerabilities in the synthetic data generator output (it produces fake data by design)
- The project not being deployed as a production service (it is a portfolio/research project)

## Data Privacy

This system is designed to work with synthetic transaction data. If you are adapting it for real financial data, you are responsible for compliance with applicable regulations (PCI-DSS, GDPR, SOX, etc.). The maintainers provide no warranty for production use with real financial data.
