"""Constants to eliminate magic numbers."""

# Product Categories
PRODUCT_CATEGORIES = [
    "All",
    "Credit card or prepaid card",
    "Payday loan, title loan, or personal loan",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service"
]

# UI & UX
DEFAULT_TOP_K = 5
STREAMING_ENABLED = True

# Paths
DEFAULT_VECTOR_STORE_DIR = "vector_store/full"
DEFAULT_PARQUET_PATH = "data/raw/complaint_embeddings.parquet"

# Evaluation
QUALITY_SCORE_SCALE = (1, 5)
