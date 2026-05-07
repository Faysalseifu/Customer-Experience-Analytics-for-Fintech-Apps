from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document

from src.config import AppConfig
from src.constants import DEFAULT_TOP_K

class RAGPipeline:
    """Main RAG pipeline with type hints and clean design."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.vectordb: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize vector store and chain."""
        self.vectordb = Chroma(
            persist_directory=str(self.config.vector_store_path),
            embedding_function=self.embeddings
        )
        
        prompt = PromptTemplate.from_template(
            """You are a financial analyst for CrediTrust. 
            Answer ONLY using the provided context. Be concise and insightful.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        llm = HuggingFaceHub(
            repo_id=self.config.llm_repo_id,
            model_kwargs={
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens
            }
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_kwargs={"k": DEFAULT_TOP_K}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str, product_filter: Optional[str] = None) -> Tuple[str, List[Document]]:
        """Main query method."""
        filter_dict = {"product_category": product_filter} if product_filter and product_filter != "All" else None
        
        result = self.qa_chain({
            "query": question,
            "filter": filter_dict
        })
        
        return result["result"], result["source_documents"]
