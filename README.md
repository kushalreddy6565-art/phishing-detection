# AI-Based Detection and Classification of Phishing URLs with Explainable AI

This project detects phishing URLs using a machine learning model and explains the prediction with SHAP-based feature importance. It includes a simple Streamlit interface for real-time URL analysis.

## Features
- URL-based phishing detection
- Explainable AI (XAI) using SHAP values
- Real-time prediction through a web UI
- Feature extraction for:
  - URL length
  - suspicious keywords
  - domain structure
  - query/path characteristics
  - special characters
  - IP-based domains
  - suspicious TLDs

## Project setup

1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate