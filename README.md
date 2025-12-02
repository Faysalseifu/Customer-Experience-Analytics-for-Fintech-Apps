# Customer-Experience-Analytics-for-Fintech-Apps
A Real-World Data Engineering Challenge: Scraping, Analyzing, and Visualizing Google Play Store Reviews

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

Copy `.env.example` to `.env` and fill in your URL to persist it for notebook runs (don't commit `.env`).

