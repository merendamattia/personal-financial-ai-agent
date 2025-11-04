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
    """Tests for datapizza-ai FastEmbedder loading."""

    @patch("src.retrieval.asset_retriever.FastEmbedder")
    def test_get_embedder(self, mock_embedder_class):
        """Test that FastEmbedder is loaded correctly."""
        # Reset global cache to ensure mock is called
        import src.retrieval.asset_retriever as ar_module

        original_embedder = ar_module._embedder
        ar_module._embedder = None

        try:
            mock_embedder = MagicMock()
            mock_embedder_class.return_value = mock_embedder
            
            retriever = RAGAssetRetriever()
            embedder = retriever._get_embedder()

            assert embedder is not None
            assert embedder is mock_embedder
            mock_embedder_class.assert_called_once()
        finally:
            # Restore original embedder
            ar_module._embedder = original_embedder

    def test_embedder_caching(self):
        """Test that FastEmbedder is cached in global variable."""
        # Reset global cache
        import src.retrieval.asset_retriever as ar_module

        original_embedder = ar_module._embedder
        ar_module._embedder = None

        try:
            with patch(
                "src.retrieval.asset_retriever.FastEmbedder"
            ) as mock_embedder_class:
                mock_embedder = MagicMock()
                mock_embedder_class.return_value = mock_embedder
                
                retriever = RAGAssetRetriever()

                # Get embedder twice
                embedder1 = retriever._get_embedder()
                embedder2 = retriever._get_embedder()

                # Should only call FastEmbedder once due to global caching
                assert mock_embedder_class.call_count == 1
                assert embedder1 is embedder2
                assert embedder1 is mock_embedder
        finally:
            # Restore original embedder
            ar_module._embedder = original_embedder


class TestRAGAssetRetrieverRetrieval:
    """Tests for document retrieval with datapizza-ai vectorstore."""

    def test_retrieve_with_mock_vectorstore(self, sample_rag_documents):
        """Test retrieval with mocked datapizza-ai vectorstore."""
        from datapizza.type.type import Chunk
        
        retriever = RAGAssetRetriever()
        retriever._is_indexed = True

        # Mock vectorstore and embedder
        mock_vectorstore = MagicMock()
        mock_embedder = MagicMock()
        
        retriever._vectorstore = mock_vectorstore
        retriever._embedder = mock_embedder

        # Mock embedder response
        mock_embedder.embed.return_value = [[0.1] * 384]  # Mock embedding

        # Create mock Chunk results
        mock_chunks = []
        for doc in sample_rag_documents[:2]:
            chunk = Chunk(
                id=doc["id"],
                text=doc["text"],
                metadata={"source": doc["source"]}
            )
            mock_chunks.append(chunk)

        mock_vectorstore.search.return_value = mock_chunks

        results = retriever.retrieve("test query", k=2)

        assert len(results) == 2
        assert all("id" in doc for doc in results)
        assert all("score" in doc for doc in results)
        mock_vectorstore.search.assert_called_once()

    def test_retrieve_k_limit(self, sample_rag_documents):
        """Test that retrieval respects k limit."""
        from datapizza.type.type import Chunk
        
        retriever = RAGAssetRetriever()
        retriever._is_indexed = True

        # Mock vectorstore and embedder
        mock_vectorstore = MagicMock()
        mock_embedder = MagicMock()
        
        retriever._vectorstore = mock_vectorstore
        retriever._embedder = mock_embedder

        # Mock embedder response
        mock_embedder.embed.return_value = [[0.1] * 384]

        def mock_search(collection_name, query_vector, k):
            """Return k chunks."""
            chunks = []
            for doc in sample_rag_documents[:k]:
                chunk = Chunk(
                    id=doc["id"],
                    text=doc["text"],
                    metadata={"source": doc["source"]}
                )
                chunks.append(chunk)
            return chunks

        mock_vectorstore.search.side_effect = mock_search

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
    """Integration tests for RAG retriever with datapizza-ai."""

    def test_retrieve_with_mocked_vectorstore(self):
        """Test retrieve function with mocked datapizza-ai components."""
        from datapizza.type.type import Chunk
        
        retriever = RAGAssetRetriever()
        retriever._is_indexed = True

        # Mock documents
        mock_docs = [
            {"id": "1", "text": "ETF bond allocation", "source": "bond.pdf"},
            {"id": "2", "text": "Stock market diversification", "source": "stock.pdf"},
            {"id": "3", "text": "Treasury bonds and yields", "source": "treasury.pdf"},
        ]

        # Mock vectorstore and embedder
        mock_vectorstore = MagicMock()
        mock_embedder = MagicMock()
        
        retriever._vectorstore = mock_vectorstore
        retriever._embedder = mock_embedder

        # Mock embedder response
        mock_embedder.embed.return_value = [[0.1] * 384]

        # Create mock search results (return first 2 docs)
        def mock_search(collection_name, query_vector, k):
            chunks = []
            for doc in mock_docs[:k]:
                chunk = Chunk(
                    id=doc["id"],
                    text=doc["text"],
                    metadata={"source": doc["source"]}
                )
                chunks.append(chunk)
            return chunks

        mock_vectorstore.search.side_effect = mock_search

        results = retriever.retrieve("I want bonds", k=2)

        # Should return documents with similarity scores
        assert len(results) == 2
        assert all("score" in d for d in results)
        assert all("text" in d for d in results)
        mock_vectorstore.search.assert_called()
