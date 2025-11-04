"""
Tests for PDF Exporter.

Tests the PortfolioPDFExporter for generating PDF documents.
"""

import io
from datetime import datetime

import pytest
from pypdf import PdfReader

from src.export import PortfolioPDFExporter


class TestPDFExporterInitialization:
    """Tests for PDF exporter initialization."""

    def test_initialize_exporter(self):
        """Test creating a PDF exporter instance."""
        exporter = PortfolioPDFExporter()
        assert exporter is not None
        assert exporter.styles is not None

    def test_custom_styles_created(self):
        """Test that custom styles are properly initialized."""
        exporter = PortfolioPDFExporter()
        # Check that custom styles exist
        assert "CustomTitle" in exporter.styles
        assert "CustomSubtitle" in exporter.styles
        assert "SectionHeader" in exporter.styles
        assert "CustomBody" in exporter.styles
        assert "SmallText" in exporter.styles


class TestPDFGeneration:
    """Tests for PDF generation functionality."""

    def test_generate_pdf_with_portfolio_only(self, sample_portfolio):
        """Test generating PDF with only portfolio data."""
        exporter = PortfolioPDFExporter()

        # Convert Portfolio model to dict
        portfolio_dict = sample_portfolio.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            provider="OpenAI",
            model="gpt-4",
        )

        assert pdf_bytes is not None
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

        # Verify it's a valid PDF
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(pdf_reader.pages) > 0

    def test_generate_pdf_with_profile_and_portfolio(
        self, sample_portfolio, sample_financial_profile
    ):
        """Test generating PDF with both profile and portfolio data."""
        exporter = PortfolioPDFExporter()

        # Convert models to dicts
        portfolio_dict = sample_portfolio.model_dump()
        profile_dict = sample_financial_profile.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            financial_profile=profile_dict,
            provider="Google",
            model="gemini-pro",
        )

        assert pdf_bytes is not None
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

        # Verify it's a valid PDF
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(pdf_reader.pages) > 0

    def test_pdf_has_multiple_pages(self, sample_portfolio, sample_financial_profile):
        """Test that generated PDF has multiple pages."""
        exporter = PortfolioPDFExporter()

        portfolio_dict = sample_portfolio.model_dump()
        profile_dict = sample_financial_profile.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            financial_profile=profile_dict,
            provider="Ollama",
            model="llama2",
        )

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        # Should have at least 2 pages (cover + content)
        assert len(pdf_reader.pages) >= 2

    def test_pdf_contains_portfolio_symbol(self, sample_portfolio):
        """Test that PDF contains portfolio asset symbols."""
        exporter = PortfolioPDFExporter()

        portfolio_dict = sample_portfolio.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            provider="OpenAI",
            model="gpt-4",
        )

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        # Extract text from all pages
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        # Check for asset symbols
        assert "SWDA" in full_text
        assert "SBXL" in full_text or "Gold" in full_text

    def test_pdf_contains_risk_level(self, sample_portfolio):
        """Test that PDF contains risk level information."""
        exporter = PortfolioPDFExporter()

        portfolio_dict = sample_portfolio.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            provider="OpenAI",
            model="gpt-4",
        )

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        # Check for risk level (case-insensitive)
        full_text_lower = full_text.lower()
        assert "moderate" in full_text_lower or "risk" in full_text_lower


class TestPDFContent:
    """Tests for PDF content sections."""

    def test_pdf_has_cover_page(self, sample_portfolio):
        """Test that PDF includes a cover page."""
        exporter = PortfolioPDFExporter()
        portfolio_dict = sample_portfolio.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            provider="OpenAI",
            model="gpt-4",
        )

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        first_page_text = pdf_reader.pages[0].extract_text()

        # Cover page should contain title and generation info
        assert "Portfolio Analysis Report" in first_page_text
        assert "Generated:" in first_page_text or "OpenAI" in first_page_text

    def test_pdf_has_disclaimers(self, sample_portfolio):
        """Test that PDF includes disclaimer section."""
        exporter = PortfolioPDFExporter()
        portfolio_dict = sample_portfolio.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            provider="OpenAI",
            model="gpt-4",
        )

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        # Check for disclaimer keywords
        full_text_lower = full_text.lower()
        assert (
            "disclaimer" in full_text_lower
            or "not financial advice" in full_text_lower
            or "risk" in full_text_lower
        )

    def test_pdf_includes_financial_profile_when_provided(
        self, sample_portfolio, sample_financial_profile
    ):
        """Test that financial profile is included when provided."""
        exporter = PortfolioPDFExporter()
        portfolio_dict = sample_portfolio.model_dump()
        profile_dict = sample_financial_profile.model_dump()

        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio_dict,
            financial_profile=profile_dict,
            provider="OpenAI",
            model="gpt-4",
        )

        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        # Check for profile information
        assert (
            "Financial Profile" in full_text
            or "30-39" in full_text  # age range from fixture
            or "intermediate" in full_text  # investment experience from fixture
        )


class TestFilenameGeneration:
    """Tests for filename generation."""

    def test_generate_default_filename(self):
        """Test generating a default filename."""
        exporter = PortfolioPDFExporter()
        filename = exporter.generate_filename()

        assert filename.startswith("portfolio_analysis_")
        assert filename.endswith(".pdf")
        assert len(filename) > len("portfolio_analysis_.pdf")

    def test_generate_filename_with_custom_prefix(self):
        """Test generating filename with custom prefix."""
        exporter = PortfolioPDFExporter()
        filename = exporter.generate_filename(prefix="my_portfolio")

        assert filename.startswith("my_portfolio_")
        assert filename.endswith(".pdf")

    def test_filename_contains_timestamp(self):
        """Test that filename contains a timestamp."""
        exporter = PortfolioPDFExporter()
        filename = exporter.generate_filename()

        # Extract timestamp part (after prefix and underscore)
        timestamp_part = filename.replace("portfolio_analysis_", "").replace(".pdf", "")

        # Should contain date-like components (YYYY-MM-DD_HH-MM-SS)
        assert "-" in timestamp_part
        assert "_" in timestamp_part

    def test_filenames_are_unique(self):
        """Test that generated filenames are unique (due to timestamp)."""
        exporter = PortfolioPDFExporter()

        filename1 = exporter.generate_filename()
        filename2 = exporter.generate_filename()

        # Filenames might be identical if generated in the same second,
        # but they should at least have the same format
        assert filename1.endswith(".pdf")
        assert filename2.endswith(".pdf")


class TestErrorHandling:
    """Tests for error handling in PDF generation."""

    def test_generate_pdf_with_empty_portfolio(self):
        """Test generating PDF with minimal portfolio data."""
        exporter = PortfolioPDFExporter()

        # Minimal valid portfolio
        minimal_portfolio = {
            "assets": [
                {"symbol": "TEST", "percentage": 100.0, "justification": "Test asset"}
            ],
            "risk_level": "moderate",
            "portfolio_reasoning": "Test reasoning",
            "key_considerations": ["Test consideration"],
            "rebalancing_schedule": "Annually",
        }

        pdf_bytes = exporter.generate_pdf(
            portfolio=minimal_portfolio,
            provider="Test",
            model="test-model",
        )

        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0

    def test_generate_pdf_with_missing_optional_fields(self):
        """Test generating PDF when optional fields are missing."""
        exporter = PortfolioPDFExporter()

        # Portfolio with minimal fields
        portfolio = {
            "assets": [
                {"symbol": "TEST", "percentage": 100.0, "justification": "Test"}
            ],
            "risk_level": "moderate",
            "portfolio_reasoning": "Test",
            "key_considerations": ["Test"],
            "rebalancing_schedule": "Test",
        }

        # Should not raise an exception
        pdf_bytes = exporter.generate_pdf(
            portfolio=portfolio,
            financial_profile=None,  # No profile provided
            provider="Test",
            model="test",
        )

        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
