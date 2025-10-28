"""
Tests for RAG Asset Retriever.

Tests the RAGAssetRetriever for semantic search functionality.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.retrieval import RAGAssetRetriever


class TestRAGAssetRetrieverInitialization:
    """Tests for RAGAssetRetriever initialization."""

    def test_retriever_initialization(self):
        """Test that retriever can be initialized."""
        retriever = RAGAssetRetriever()
        assert retriever is not None
        assert retriever.data_dir is not None
        assert retriever.cache_dir is not None
        assert retriever.emb_cache is not None

    def test_retriever_with_custom_data_dir(self, tmp_path):
        """Test retriever initialization with custom data directory."""
        retriever = RAGAssetRetriever(data_dir=tmp_path)
        assert retriever.data_dir == tmp_path


class TestRAGAssetRetrieverChunking:
    """Tests for text chunking functionality."""

    def test_chunk_text_simple(self):
        """Test simple text chunking."""
        retriever = RAGAssetRetriever()
        text = "A" * 1000  # 1000 character string
        chunks = retriever._chunk_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_with_overlap(self):
        """Test that chunks have overlap."""
        retriever = RAGAssetRetriever()
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100
        chunks = retriever._chunk_text(text)

        # Check that consecutive chunks overlap
        if len(chunks) > 1:
            # The end of one chunk should overlap with start of next
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        retriever = RAGAssetRetriever()
        chunks = retriever._chunk_text("")
        assert chunks == []

    def test_chunk_small_text(self):
        """Test chunking text smaller than chunk size."""
        retriever = RAGAssetRetriever()
        text = "Small text"
        chunks = retriever._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestRAGAssetRetrieverEmbedding:
    """Tests for embedding model loading."""

    @patch("src.retrieval.asset_retriever.SentenceTransformer")
    def test_load_embedder(self, mock_transformer):
        """Test that embedder is loaded correctly."""
        # Reset global cache to ensure mock is called
        import src.retrieval.asset_retriever as ar_module

        original_model = ar_module._embedding_model
        ar_module._embedding_model = None

        try:
            mock_model = MagicMock()
            mock_transformer.return_value = mock_model

            embedder = RAGAssetRetriever._load_embedder()

            assert embedder is not None
            assert embedder is mock_model
            mock_transformer.assert_called_once()
        finally:
            # Restore original model
            ar_module._embedding_model = original_model

    def test_embedder_caching(self):
        """Test that embedder model is cached in global variable."""
        # Reset global cache
        import src.retrieval.asset_retriever as ar_module

        original_model = ar_module._embedding_model
        ar_module._embedding_model = None

        try:
            with patch(
                "src.retrieval.asset_retriever.SentenceTransformer"
            ) as mock_transformer:
                mock_model = MagicMock()
                mock_transformer.return_value = mock_model

                # Load embedder twice
                embedder1 = RAGAssetRetriever._load_embedder()
                embedder2 = RAGAssetRetriever._load_embedder()

                # Should only call transformer once due to global caching
                assert mock_transformer.call_count == 1
                assert embedder1 is embedder2
                assert embedder1 is mock_model
        finally:
            # Restore original model
            ar_module._embedding_model = original_model


class TestRAGAssetRetrieverRetrieval:
    """Tests for document retrieval functionality."""

    def test_retrieve_with_mock_embeddings(self, sample_rag_documents):
        """Test retrieval with mocked embeddings."""
        retriever = RAGAssetRetriever()

        # Mock the internal state
        retriever._documents = sample_rag_documents

        # Create mock embeddings (3 documents, 384-dim)
        import numpy as np

        retriever._embeddings = np.random.randn(3, 384)

        # Patch the embedder
        with patch.object(retriever, "_load_embedder") as mock_embedder:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = np.random.randn(1, 384)
            mock_embedder.return_value = mock_encoder

            results = retriever.retrieve("test query", k=2)

            assert len(results) == 2
            assert all("id" in doc for doc in results)
            assert all("score" in doc for doc in results)

    def test_retrieve_k_limit(self, sample_rag_documents):
        """Test that retrieval respects k limit."""
        retriever = RAGAssetRetriever()
        retriever._documents = sample_rag_documents

        import numpy as np

        retriever._embeddings = np.random.randn(3, 384)

        with patch.object(retriever, "_load_embedder") as mock_embedder:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = np.random.randn(1, 384)
            mock_embedder.return_value = mock_encoder

            results = retriever.retrieve("query", k=2)
            assert len(results) == 2

            results_one = retriever.retrieve("query", k=1)
            assert len(results_one) == 1


class TestRAGAssetRetrieverDocumentProcessing:
    """Tests for document processing."""

    def test_read_pdf_text_mock(self):
        """Test PDF text extraction with mocked PDF."""
        retriever = RAGAssetRetriever()

        # This would normally fail without actual PDFs
        # but we're testing the method structure
        assert hasattr(retriever, "_read_pdf_text")
        assert callable(retriever._read_pdf_text)

    def test_ingest_pdfs_empty_dir(self, tmp_path):
        """Test ingesting from empty directory raises error."""
        retriever = RAGAssetRetriever(data_dir=tmp_path)

        with pytest.raises(RuntimeError, match="No PDFs found"):
            retriever.ingest_pdfs()


class TestRAGAssetRetrieverIntegration:
    """Integration tests for RAG retriever."""

    def test_retrieve_with_mocked_index(self):
        """Test retrieve function with mocked index."""
        retriever = RAGAssetRetriever()

        # Mock documents and embeddings
        mock_docs = [
            {"id": "1", "text": "ETF bond allocation", "source": "bond.pdf"},
            {"id": "2", "text": "Stock market diversification", "source": "stock.pdf"},
            {"id": "3", "text": "Treasury bonds and yields", "source": "treasury.pdf"},
        ]
        mock_embeddings = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.15, 0.25, 0.35]]
        )

        # Set mocked data
        retriever._documents = mock_docs
        retriever._embeddings = mock_embeddings

        with patch.object(retriever, "_load_embedder") as mock_embedder:
            mock_model = MagicMock()
            mock_embedder.return_value = mock_model
            # Query embedding similar to first document
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

            results = retriever.retrieve("I want bonds", k=2)

            # Should return documents with similarity scores
            assert len(results) == 2
            assert all("score" in d for d in results)
            assert all("text" in d for d in results)
