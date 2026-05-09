import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from unittest.mock import patch
from src.rag_pipeline import RAGPipeline
from src.config import AppConfig

@pytest.fixture
def mock_config():
    """Fixture for a mock AppConfig."""
    return AppConfig(
        hf_token="test_token",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        vector_store_path="tests/test_vector_store",
        parquet_path="tests/test_data/test_embeddings.parquet"
    )

@patch('src.rag_pipeline.Chroma')
@patch('src.rag_pipeline.HuggingFaceEmbeddings')
@patch('src.rag_pipeline.HuggingFaceHub')
@patch('src.rag_pipeline.PromptTemplate')
@patch('src.rag_pipeline.RetrievalQA')
def test_rag_pipeline_initialization(mock_retrieval_qa, mock_prompt_template, mock_hf_hub, mock_hf_embeddings, mock_chroma, mock_config):
    """Test the initialization of the RAGPipeline."""
    rag = RAGPipeline(mock_config)
    
    mock_hf_embeddings.assert_called_once_with(model_name=mock_config.embedding_model)
    mock_chroma.assert_called_once_with(
        persist_directory=str(mock_config.vector_store_path),
        embedding_function=rag.embeddings
    )
    mock_hf_hub.assert_called_once()
    mock_prompt_template.from_template.assert_called_once()
    mock_retrieval_qa.from_chain_type.assert_called_once()
    
    assert rag.qa_chain is not None

@patch('src.rag_pipeline.RAGPipeline.query')
def test_rag_pipeline_query(mock_query, mock_config):
    """Test the query method of the RAGPipeline."""
    mock_query.return_value = ("This is a test answer.", [])
    
    rag = RAGPipeline(mock_config)
    answer, sources = rag.query("What is a test?")
    
    assert answer == "This is a test answer."
    assert sources == []
