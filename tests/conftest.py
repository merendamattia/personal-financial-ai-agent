"""
Pytest configuration and shared fixtures.

This module provides common fixtures for all tests.
"""

import pytest

from src.models import FinancialProfile, Portfolio


@pytest.fixture
def sample_financial_profile():
    """Create a sample financial profile for testing."""
    return FinancialProfile(
        age_range="30-39",
        employment_status="employed",
        occupation="Software Engineer",
        annual_income_range="80k-120k",
        income_stability="stable",
        additional_income_sources="None",
        monthly_expenses_range="4k-5k",
        major_expenses="Mortgage: $2000/month",
        total_debt="150k (mortgage)",
        debt_types="Mortgage",
        savings_amount="50k",
        emergency_fund_months="6",
        investments="ETF index funds",
        investment_experience="intermediate",
        primary_goals="Retirement planning",
        short_term_goals="Emergency fund optimization",
        long_term_goals="Retire at 65 with $2M portfolio",
        risk_tolerance="moderate",
        risk_concerns="Market volatility",
        financial_knowledge_level="intermediate",
        geographic_allocation="Global balanced",
        family_dependents="2 children",
        insurance_coverage="Health, Life, Home",
        summary_notes="Good financial discipline",
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return Portfolio(
        primary_asset="SWDA",
        primary_asset_percentage=60.0,
        primary_asset_justification="Global diversified equity exposure",
        secondary_asset="SBXL",
        secondary_asset_percentage=30.0,
        secondary_asset_justification="European bonds for stability",
        tertiary_asset="Gold",
        tertiary_asset_percentage=10.0,
        tertiary_asset_justification="Precious metals hedge",
        quaternary_asset=None,
        quaternary_asset_percentage=None,
        quaternary_asset_justification=None,
        quinary_asset=None,
        quinary_asset_percentage=None,
        quinary_asset_justification=None,
        risk_level="moderate",
        portfolio_reasoning="Balanced allocation for moderate investor",
        key_considerations="Monthly contributions; Annual review; Tax-efficient",
        rebalancing_schedule="Annually or when drift >5%",
    )


@pytest.fixture
def sample_rag_documents():
    """Create sample RAG documents for testing."""
    return [
        {
            "id": "etf1.pdf::chunk_0",
            "source": "dataset/ETFs/etf1.pdf",
            "text": "SWDA is a global equity ETF with low fees and broad diversification.",
            "score": 0.85,
        },
        {
            "id": "etf2.pdf::chunk_0",
            "source": "dataset/ETFs/etf2.pdf",
            "text": "SBXL provides exposure to European bonds with stable returns.",
            "score": 0.78,
        },
        {
            "id": "etf3.pdf::chunk_0",
            "source": "dataset/ETFs/etf3.pdf",
            "text": "Gold ETF offers protection against market downturns.",
            "score": 0.72,
        },
    ]
