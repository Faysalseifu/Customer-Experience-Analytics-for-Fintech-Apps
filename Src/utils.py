from typing import List, Dict, Any
from langchain.schema import Document

def format_sources(docs: List[Document]) -> str:
    """Format retrieved documents into readable markdown."""
    if not docs:
        return "**No relevant sources found.**"
    
    md = "**Retrieved Sources:**\n\n"
    for i, doc in enumerate(docs, 1):
        md += f"**Source {i}** — "
        md += f"**Product:** {doc.metadata.get('product_category', 'N/A')}, "
        md += f"**Issue:** {doc.metadata.get('issue', 'N/A')}\n"
        md += f"**Complaint ID:** {doc.metadata.get('complaint_id', 'N/A')}\n"
        excerpt = doc.page_content[:280].replace('\n', ' ')
        md += f"**Excerpt:** {excerpt}...\n\n"
    return md


def validate_config(config: "AppConfig") -> None:
    """Validate critical configuration."""
    if not config.hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is required. Set it in .env or environment.")
    if not config.parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {config.parquet_path}")
