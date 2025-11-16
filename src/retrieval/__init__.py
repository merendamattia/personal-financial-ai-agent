"""
Pacchetto Retrieval.
Espone il servizio RAG basato su LangChain come unico punto di accesso
per la ricerca documentale.
"""
from .rag_service import RAGService

__all__ = ["RAGService"]
