"""RAG Asset Retriever Module.

This module provides the RAGAssetRetriever class for semantic search
over ETF/asset PDFs in the dataset directory.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# RAG Constants from environment variables
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "dataset/ETFs"))
CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", "dataset/ETFs/.cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE = Path(
    os.getenv("RAG_EMBEDDINGS_CACHE", "dataset/ETFs/.cache/embeddings.pkl")
)

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMB_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-roberta-large-v1")

# Global embedding model cache
_embedding_model = None


class RAGAssetRetriever:
    """Simple RAG retriever for asset PDFs using semantic search."""

    def __init__(self, data_dir: Path = DATA_DIR):
        """Initialize RAG retriever.

        Args:
            data_dir: Path to the ETF dataset directory
        """
        self.data_dir = data_dir
        self.cache_dir = CACHE_DIR
        self.emb_cache = EMB_CACHE
        self._documents = None
        self._embeddings = None

    @staticmethod
    def _load_embedder():
        """Load and cache the embedding model globally.

        Returns:
            SentenceTransformer model instance
        """
        global _embedding_model
        if _embedding_model is None:
            logger.info("Loading embedding model: %s", EMB_MODEL_NAME)
            _embedding_model = SentenceTransformer(EMB_MODEL_NAME)
        return _embedding_model

    def _read_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text from all pages
        """
        try:
            reader = PdfReader(str(pdf_path))
            texts = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                texts.append(txt)
            return "\n".join(texts)
        except Exception as e:
            logger.error("Error reading PDF %s: %s", pdf_path, e)
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks with overlap
        """
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + DEFAULT_CHUNK_SIZE, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - DEFAULT_CHUNK_OVERLAP
            if start < 0:
                start = 0
        return chunks

    def ingest_pdfs(self) -> List[Dict]:
        """Ingest all PDFs from data directory.

        Recursively scans data_dir for PDF files, extracts text,
        and chunks them for embedding.

        Returns:
            List of document dictionaries with id, source, and text

        Raises:
            RuntimeError: If no PDFs found in data directory
        """
        logger.info("Ingesting PDFs from: %s", self.data_dir)
        docs = []
        for pdf in sorted(self.data_dir.rglob("*.pdf")):
            try:
                txt = self._read_pdf_text(pdf)
                if not txt.strip():
                    logger.debug("Skipping empty PDF: %s", pdf.name)
                    continue
                for i, ch in enumerate(self._chunk_text(txt)):
                    docs.append(
                        {"id": f"{pdf.name}::chunk_{i}", "source": str(pdf), "text": ch}
                    )
                logger.debug(
                    "Processed PDF: %s (%d chunks)",
                    pdf.name,
                    len(self._chunk_text(txt)),
                )
            except Exception as e:
                logger.error("Error parsing %s: %s", pdf, e)

        if not docs:
            raise RuntimeError(f"No PDFs found in {self.data_dir}")

        logger.info(
            "Ingested %d document chunks from %d PDFs",
            len(docs),
            len(list(self.data_dir.rglob("*.pdf"))),
        )
        return docs

    def build_or_load_index(self) -> Tuple[List[Dict], np.ndarray]:
        """Build or load cached embedding index.

        If embeddings cache exists, loads from disk. Otherwise, ingests
        all PDFs from data_dir and generates embeddings using the
        SentenceTransformer model.

        Returns:
            Tuple of (documents list, embeddings array)

        Raises:
            RuntimeError: If no PDFs found when building index
        """
        if self.emb_cache.exists():
            logger.info("Loading cached embeddings from: %s", self.emb_cache)
            with open(self.emb_cache, "rb") as f:
                payload = pickle.load(f)
            self._documents = payload["docs"]
            self._embeddings = payload["embeddings"]
            logger.info("Loaded %d documents from cache", len(self._documents))
            return self._documents, self._embeddings

        logger.info("Building embedding index from PDFs")
        docs = self.ingest_pdfs()
        if not docs:
            raise RuntimeError(f"No PDFs found in {self.data_dir}")

        embedder = self._load_embedder()
        logger.info("Generating embeddings for %d document chunks", len(docs))
        texts = [d["text"] for d in docs]
        embs = embedder.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )

        logger.info("Caching embeddings to: %s", self.emb_cache)
        with open(self.emb_cache, "wb") as f:
            pickle.dump({"docs": docs, "embeddings": embs}, f)

        self._documents = docs
        self._embeddings = embs
        logger.info("Index built and cached successfully")
        return docs, embs

    def retrieve(self, query: str, k: int = 15) -> List[Dict]:
        """Retrieve k most similar documents via semantic search.

        Encodes the query and finds the k documents with highest
        cosine similarity to the query embedding.

        Args:
            query: Search query text
            k: Number of results to return (default: 5)

        Returns:
            List of k most similar documents with scores
        """
        if self._documents is None or self._embeddings is None:
            logger.debug("Index not loaded, building or loading now")
            self.build_or_load_index()

        embedder = self._load_embedder()
        logger.debug("Encoding query: %s", query[:100])
        q = embedder.encode([query], convert_to_numpy=True)

        logger.debug(
            "Computing similarity scores against %d documents", len(self._documents)
        )
        sims = cosine_similarity(q, self._embeddings)[0]
        idxs = np.argsort(-sims)[:k]

        results = []
        for idx in idxs:
            d = self._documents[idx].copy()
            d["score"] = float(sims[idx])
            results.append(d)

        logger.info(
            "Retrieved %d documents, top score: %.4f",
            len(results),
            results[0]["score"] if results else 0,
        )
        return results
