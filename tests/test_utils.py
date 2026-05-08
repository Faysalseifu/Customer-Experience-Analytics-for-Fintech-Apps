import pytest
from unittest.mock import patch
from src.utils import format_sources, validate_config
from langchain.schema import Document
from src.config import AppConfig

def test_format_sources():
    """Test the format_sources utility function."""
    docs = [
        Document(page_content="This is the first document.", metadata={"product_category": "Card", "issue": "Fraud", "complaint_id": "123"}),
        Document(page_content="This is the second document.", metadata={"product_category": "Loan", "issue": "Billing", "complaint_id": "456"})
    ]
    
    formatted_string = format_sources(docs)
    
    assert "**Retrieved Sources:**" in formatted_string
    assert "**Source 1**" in formatted_string
    assert "**Source 2**" in formatted_string
    assert "Product:** Card" in formatted_string
    assert "Issue:** Fraud" in formatted_string
    assert "Complaint ID:** 123" in formatted_string
    assert "Excerpt:** This is the first document." in formatted_string

def test_validate_config_success():
    """Test validate_config with a valid configuration."""
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = True
        config = AppConfig(hf_token="test_token", parquet_path="dummy.parquet")
        try:
            validate_config(config)
        except (ValueError, FileNotFoundError):
            pytest.fail("validate_config raised an exception unexpectedly.")

def test_validate_config_no_token():
    """Test validate_config with a missing Hugging Face token."""
    with pytest.raises(ValueError, match="HUGGINGFACEHUB_API_TOKEN is required"):
        config = AppConfig(parquet_path="dummy.parquet")
        validate_config(config)

def test_validate_config_file_not_found():
    """Test validate_config with a non-existent parquet file."""
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="Parquet file not found"):
            config = AppConfig(hf_token="test_token", parquet_path="non_existent.parquet")
            validate_config(config)
