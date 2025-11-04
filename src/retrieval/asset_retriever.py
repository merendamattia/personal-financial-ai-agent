"""RAG Asset Retriever Module.

This module provides the RAGAssetRetriever class for semantic search
over ETF/asset PDFs in the dataset directory using datapizza-ai's
Qdrant vector store for efficient semantic search.
"""

import logging
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

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

# Qdrant storage directory (part of datapizza-ai ecosystem)
QDRANT_STORAGE_DIR = CACHE_DIR / "qdrant_storage"
QDRANT_COLLECTION = "financial_assets"

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMB_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-roberta-large-v1")

# Global embedding model cache
_embedding_model = None


class RAGAssetRetriever:
    """RAG retriever for asset PDFs using datapizza-ai's Qdrant vector store."""

    def __init__(self, data_dir: Path = DATA_DIR):
        """Initialize RAG retriever with Qdrant vector store.

        Args:
            data_dir: Path to the ETF dataset directory
        """
        self.data_dir = data_dir
        self.cache_dir = CACHE_DIR
        self.emb_cache = EMB_CACHE
        self.qdrant_path = QDRANT_STORAGE_DIR
        self._documents = None
        self._embeddings = None
        self._qdrant_client: Optional[QdrantClient] = None
        self._is_indexed = False

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
                chunks = self._chunk_text(txt)
                for i, ch in enumerate(chunks):
                    docs.append(
                        {"id": f"{pdf.name}::chunk_{i}", "source": str(pdf), "text": ch}
                    )
                logger.debug(
                    "Processed PDF: %s (%d chunks)",
                    pdf.name,
                    len(chunks),
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

    def build_or_load_index(self) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Build or load cached embedding index using Qdrant vector store.

        If Qdrant collection exists, loads from it. Otherwise, ingests
        all PDFs from data_dir and generates embeddings using the
        SentenceTransformer model, storing them in Qdrant.

        Returns:
            Tuple of (documents list, embeddings array or None)

        Raises:
            RuntimeError: If no PDFs found when building index
        """
        # Initialize Qdrant client
        if self._qdrant_client is None:
            logger.info("Initializing Qdrant client at: %s", self.qdrant_path)
            self._qdrant_client = QdrantClient(path=str(self.qdrant_path))

        # Check if collection exists with documents
        try:
            collections = self._qdrant_client.get_collections().collections
            collection_exists = any(c.name == QDRANT_COLLECTION for c in collections)
            if collection_exists:
                # Check if collection has points
                count = self._qdrant_client.count(
                    collection_name=QDRANT_COLLECTION
                )
                if count.count > 0:
                    logger.info(
                        "Loaded existing Qdrant index with %d documents",
                        count.count
                    )
                    self._is_indexed = True
                    # Load documents metadata if pickle cache exists
                    if self.emb_cache.exists():
                        logger.debug("Loading document metadata from pickle cache")
                        with open(self.emb_cache, "rb") as f:
                            payload = pickle.load(f)
                        self._documents = payload.get("docs", [])
                    return self._documents or [], None
        except Exception as e:
            logger.debug("No existing Qdrant index found: %s", e)

        # Build new index
        logger.info("Building new embedding index from PDFs")
        docs = self.ingest_pdfs()
        if not docs:
            raise RuntimeError(f"No PDFs found in {self.data_dir}")

        self._documents = docs

        # Generate embeddings
        embedder = self._load_embedder()
        logger.info("Generating embeddings for %d document chunks", len(docs))
        texts = [d["text"] for d in docs]
        embs = embedder.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        self._embeddings = embs

        # Get embedding dimension
        embedding_dim = embs.shape[1]
        logger.info("Embedding dimension: %d", embedding_dim)

        # Create Qdrant collection
        logger.info("Creating Qdrant collection: %s", QDRANT_COLLECTION)
        self._qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )

        # Upload points to Qdrant
        logger.info("Uploading %d points to Qdrant", len(docs))
        points = []
        for idx, (doc, emb) in enumerate(zip(docs, embs)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=emb.tolist(),
                    payload={
                        "doc_id": doc["id"],
                        "source": doc["source"],
                        "text": doc["text"]
                    }
                )
            )

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=batch
            )
            logger.debug("Uploaded batch %d/%d", i // batch_size + 1, math.ceil(len(points) / batch_size))

        # Cache embeddings and documents for backward compatibility
        logger.info("Caching embeddings to: %s", self.emb_cache)
        with open(self.emb_cache, "wb") as f:
            pickle.dump({"docs": docs, "embeddings": embs}, f)

        logger.info("Index built and cached successfully in Qdrant")
        self._is_indexed = True
        return docs, embs

    def retrieve(self, query: str, k: int = 15) -> List[Dict]:
        """Retrieve k most similar documents via semantic search using Qdrant.

        Encodes the query and finds the k documents with highest
        cosine similarity using Qdrant vector search.

        Args:
            query: Search query text
            k: Number of results to return (default: 15)

        Returns:
            List of k most similar documents with scores
        """
        if self._qdrant_client is None or not self._is_indexed:
            logger.debug("Index not loaded, building or loading now")
            self.build_or_load_index()

        embedder = self._load_embedder()
        logger.debug("Encoding query: %s", query[:100])
        query_vector = embedder.encode([query], convert_to_numpy=True)[0]

        logger.debug("Performing Qdrant search for top %d documents", k)
        # Search in Qdrant
        search_results = self._qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector.tolist(),
            limit=k
        )

        logger.info("Retrieved %d documents from Qdrant", len(search_results))

        # Format results to match expected output format
        results = []
        for result in search_results:
            results.append({
                "id": result.payload["doc_id"],
                "source": result.payload["source"],
                "text": result.payload["text"],
                "score": float(result.score)
            })

        if results:
            logger.info("Top score: %.4f", results[0]["score"])

        return results
