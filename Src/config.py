from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class AppConfig:
    """Central configuration for the RAG application."""
    hf_token: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    vector_store_path: Path = Path("vector_store/full")
    parquet_path: Path = Path("data/raw/complaint_embeddings.parquet")
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    temperature: float = 0.35
    max_new_tokens: int = 512
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load config from environment variables."""
        import os
        return cls(
            hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", "vector_store/full")),
            parquet_path=Path(os.getenv("PARQUET_PATH", "data/raw/complaint_embeddings.parquet")),
        )
