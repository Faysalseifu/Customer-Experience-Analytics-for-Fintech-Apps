# Customer Experience Analytics for Fintech Apps

[![CircleCI](https://circleci.com/gh/yourusername/your-repo.svg?style=svg)](https://circleci.com/gh/yourusername/your-repo)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pytest](https://img.shields.io/badge/Testing-pytest-green)

---

**10 Academy Week 2 Challenge** – End-to-End Data Pipeline with Sentiment Analysis, Thematic Clustering, PostgreSQL, and CI/CD.

## Project Structure
- `app.py`: Main application file.
- `config.py`: Configuration settings.
- `main.py`: Main script to run the pipeline.
- `requirements.txt`: Project dependencies.
- `src/`: Source code for the project.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `tests/`: Unit tests for the project.
- `data/`: Data files.
- `reports/`: Generated reports and visualizations.

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

## Running Tests
To run the tests, use the following command:
```bash
pytest
```



