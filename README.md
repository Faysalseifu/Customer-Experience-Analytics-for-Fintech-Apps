# Customer Experience Analytics for Fintech Apps

![CI Status](https://github.com/yourusername/your-repo/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pytest](https://img.shields.io/badge/Testing-pytest-green)

---

**10 Academy Week 2 Challenge** – End-to-End Data Pipeline with Sentiment Analysis, Thematic Clustering, PostgreSQL, and CI/CD.

## Features
- Robust Google Play Store scraper
- DistilBERT Sentiment Analysis
- Rule-based Thematic Analysis
- PostgreSQL Data Storage
- Unit Tests + GitHub Actions CI/CD
- Professional Visualizations & Report

## Setup
Install project dependencies into your Python environment (recommended to use a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

If you run the notebooks interactively, `notebooks/insert_into_tables.ipynb` contains a helper cell that will attempt to install `requirements.txt` automatically.

Set your Neon DB URL before running the notebook (recommended via `.env` or environment variable):

```bash
# temporary for current bash session
export NEON_DATABASE_URL="postgresql://neondb_owner:YOUR_PASSWORD@ep-crimson-bush-.../bank_reviews?sslmode=require&channel_binding=require"
```


