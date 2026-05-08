from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from src.config import AppConfig
from src.utils import validate_config

class VectorStoreBuilder:
    """Handles building and persisting the full vector store."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        validate_config(config)
    
    def build(self, force_rebuild: bool = False) -> Chroma:
        """Build vector store from pre-computed embeddings."""
        if not force_rebuild and self.config.vector_store_path.exists():
            print("✅ Using existing vector store.")
            return Chroma(
                persist_directory=str(self.config.vector_store_path),
                embedding_function=None  # Will be set at runtime
            )
        
        print("Loading pre-built embeddings...")
        df = pd.read_parquet(self.config.parquet_path)
        
        # Custom embedder for pre-computed vectors
        class PrecomputedEmbedder:
            def embed_documents(self, texts):
                return df['embedding'].tolist()
            def embed_query(self, text):
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=self.config.embedding_model).embed_query(text)
        
        embedder = PrecomputedEmbedder()
        
        print(f"Building Chroma index with {len(df)} chunks...")
        vectordb = Chroma.from_texts(
            texts=df['chunk_text'].tolist(),
            embedding=embedder,
            metadatas=df.drop(columns=['chunk_text', 'embedding']).to_dict('records'),
            persist_directory=str(self.config.vector_store_path)
        )
        vectordb.persist()
        print("✅ Full vector store built successfully!")
        return vectordb
