"""RAG Asset Retriever Module using Datapizza-AI.

This module provides the RAGAssetRetriever class for semantic search
over ETF/asset PDFs in the dataset directory using datapizza-ai's
native RAG implementation with QdrantVectorstore and FastEmbedder.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datapizza.embedders.fastembedder import FastEmbedder
from datapizza.type.type import Chunk
from datapizza.vectorstores.qdrant import QdrantVectorstore
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# RAG Constants from environment variables
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "dataset/ETFs"))
CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", "dataset/ETFs/.cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Qdrant storage directory (datapizza-ai vector store)
QDRANT_STORAGE_DIR = CACHE_DIR / "qdrant_storage"
QDRANT_COLLECTION = "financial_assets"

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
# Use sparse embedding model compatible with datapizza-ai FastEmbedder
EMB_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "prithivida/Splade_PP_en_v1")

# Global embedder and vectorstore cache
_embedder = None
_vectorstore = None


class RAGAssetRetriever:
    """RAG retriever for asset PDFs using datapizza-ai framework."""

    def __init__(self, data_dir: Path = DATA_DIR):
        """Initialize RAG retriever with datapizza-ai components.

        Args:
            data_dir: Path to the ETF dataset directory
        """
        self.data_dir = data_dir
        self.cache_dir = CACHE_DIR
        self.qdrant_path = QDRANT_STORAGE_DIR
        self._documents = None
        self._embedder: Optional[FastEmbedder] = None
        self._vectorstore: Optional[QdrantVectorstore] = None
        self._retriever = None
        self._is_indexed = False

    def _get_embedder(self) -> FastEmbedder:
        """Get or create the datapizza-ai FastEmbedder.

        Returns:
            FastEmbedder instance
        """
        global _embedder
        if _embedder is None:
            logger.info("Loading FastEmbedder model: %s", EMB_MODEL_NAME)
            _embedder = FastEmbedder(model_name=EMB_MODEL_NAME)
        return _embedder

    def _get_vectorstore(self) -> QdrantVectorstore:
        """Get or create the datapizza-ai QdrantVectorstore.

        Returns:
            QdrantVectorstore instance
        """
        global _vectorstore
        if _vectorstore is None:
            logger.info("Initializing QdrantVectorstore at: %s", self.qdrant_path)
            _vectorstore = QdrantVectorstore(location=str(self.qdrant_path))
        return _vectorstore

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

    def build_or_load_index(self) -> Tuple[List[Dict], None]:
        """Build or load index using datapizza-ai vectorstore.

        Creates QdrantVectorstore collection and adds document chunks
        with embeddings using datapizza-ai's FastEmbedder.

        Returns:
            Tuple of (documents list, None) - embeddings managed by vectorstore

        Raises:
            RuntimeError: If no PDFs found when building index
        """
        # Initialize vectorstore and embedder
        if self._vectorstore is None:
            self._vectorstore = self._get_vectorstore()
        if self._embedder is None:
            self._embedder = self._get_embedder()

        # Check if collection already exists with data
        try:
            collections = self._vectorstore.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if QDRANT_COLLECTION in collection_names:
                count = self._vectorstore.get_client().count(
                    collection_name=QDRANT_COLLECTION
                )
                if count.count > 0:
                    logger.info(
                        "Loaded existing collection '%s' with %d documents",
                        QDRANT_COLLECTION,
                        count.count
                    )
                    self._is_indexed = True
                    # Load documents list if needed
                    self._documents = []
                    return self._documents, None
        except Exception as e:
            logger.debug("No existing collection found: %s", e)

        # Build new index
        logger.info("Building new index with datapizza-ai")
        docs = self.ingest_pdfs()
        if not docs:
            raise RuntimeError(f"No PDFs found in {self.data_dir}")

        self._documents = docs

        # Create Chunk objects for datapizza-ai
        logger.info("Creating %d chunks for vectorstore", len(docs))
        chunks = []
        for doc in docs:
            # Embed the text using datapizza-ai FastEmbedder
            embedding = self._embedder.embed(doc["text"])
            
            # Create Chunk with embedding
            chunk = Chunk(
                id=doc["id"],
                text=doc["text"],
                embeddings=embedding,
                metadata={"source": doc["source"]}
            )
            chunks.append(chunk)

        # Add chunks to vectorstore (this will create the collection)
        logger.info("Adding chunks to QdrantVectorstore collection '%s'", QDRANT_COLLECTION)
        self._vectorstore.add(chunks, collection_name=QDRANT_COLLECTION)

        logger.info("Index built successfully using datapizza-ai")
        self._is_indexed = True
        return docs, None

    def retrieve(self, query: str, k: int = 15) -> List[Dict]:
        """Retrieve k most similar documents using datapizza-ai vectorstore.

        Uses datapizza-ai's FastEmbedder to encode the query and
        QdrantVectorstore to find similar documents.

        Args:
            query: Search query text
            k: Number of results to return (default: 15)

        Returns:
            List of k most similar documents with scores
        """
        if not self._is_indexed:
            logger.debug("Index not loaded, building or loading now")
            self.build_or_load_index()

        if self._vectorstore is None:
            self._vectorstore = self._get_vectorstore()
        if self._embedder is None:
            self._embedder = self._get_embedder()

        logger.debug("Encoding query with datapizza-ai: %s", query[:100])
        
        # Embed query using datapizza-ai FastEmbedder
        query_embedding = self._embedder.embed(query)

        logger.debug("Searching in QdrantVectorstore for top %d documents", k)
        
        # Search using datapizza-ai vectorstore
        # Extract the embedding vector (FastEmbedder returns list of embeddings)
        if isinstance(query_embedding, list) and len(query_embedding) > 0:
            query_vector = query_embedding[0]
        else:
            query_vector = query_embedding

        # Perform search
        search_results = self._vectorstore.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            k=k
        )

        logger.info("Retrieved %d documents from datapizza-ai vectorstore", len(search_results))

        # Convert Chunk objects to dict format for compatibility
        results = []
        for chunk in search_results:
            results.append({
                "id": chunk.id,
                "source": chunk.metadata.get("source", "unknown"),
                "text": chunk.text,
                "score": 1.0  # Qdrant returns chunks without explicit scores in this mode
            })

        if results:
            logger.info("Retrieved %d results", len(results))

        return results
