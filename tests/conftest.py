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
        annual_income_range="80k-120k",
        income_stability="stable",
        monthly_expenses_range="4k-5k",
        major_expenses="Mortgage: €2000/month",
        total_debt="150k",
        debt_types="Mortgage",
        montly_savings_amount="1k",
        active_investments="20k euros in ETF index funds",
        investment_experience="intermediate",
        goals="Retirement planning, home equity building",
        risk_tolerance="moderate",
        geographic_allocation="Global balanced",
        family_dependents="2 children",
        insurance_coverage="Health, Life, Home insurance",
        summary_notes="User shows good financial discipline and awareness",
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    from src.models.portfolio import Asset

    return Portfolio(
        assets=[
            Asset(
                symbol="SWDA",
                percentage=60.0,
                justification="Global diversified equity exposure for long-term growth",
            ),
            Asset(
                symbol="SBXL",
                percentage=30.0,
                justification="European bonds for stability and income",
            ),
            Asset(
                symbol="Gold",
                percentage=10.0,
                justification="Precious metals hedge for portfolio protection",
            ),
        ],
        risk_level="moderate",
        portfolio_reasoning="Balanced allocation combines growth potential with downside protection",
        key_considerations=[
            "Regular monthly contributions recommended",
            "Review allocation annually",
            "Consider tax-efficient placement",
        ],
        rebalancing_schedule="Annually or when allocations drift >5%",
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
