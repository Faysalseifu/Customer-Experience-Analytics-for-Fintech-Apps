import gradio as gr
from src.config import AppConfig
from src.rag_pipeline import RAGPipeline
from src.utils import format_sources
from src.constants import PRODUCT_CATEGORIES

config = AppConfig.from_env()
rag = RAGPipeline(config)

def chat_response(message: str, history, product_filter: str):
    answer, sources = rag.query(message, product_filter)
    return answer, format_sources(sources)

# Gradio UI (clean Blocks implementation)
with gr.Blocks(title="CrediTrust RAG Chatbot") as demo:
    # ... (same UI as before but cleaner and using constants)
