"""
Financial Profile Data Model.

This module defines the Pydantic model for extracting structured
financial information from user conversations.
"""

from typing import Optional

from pydantic import BaseModel, Field


class FinancialProfile(BaseModel):
    """Structured financial profile extracted from user conversation."""

    # Personal Information
    age_range: str = Field(
        default="18-65",
        description="Age range of the user (e.g., '25-34', '35-44', etc.)",
    )
    employment_status: str = Field(
        default="Employed",
        description="Employment status (e.g., 'employed', 'self-employed', 'retired', etc.)",
    )
    occupation: Optional[str] = Field(
        default="None", description="Current occupation or industry"
    )

    # Income Information
    annual_income_range: str = Field(
        default="None",
        description="Annual income range (e.g., '30k-50k', '50k-100k', '100k+', etc.)",
    )
    income_stability: str = Field(
        default="None",
        description="Income stability rating (e.g., 'stable', 'moderate', 'unstable')",
    )
    additional_income_sources: Optional[str] = Field(
        default="None",
        description="Any additional income sources (investments, side business, etc.)",
    )

    # Expenses and Debts
    monthly_expenses_range: str = Field(
        default="None",
        description="Estimated monthly expenses range (e.g., '2k-3k', '3k-5k', etc.)",
    )
    major_expenses: Optional[str] = Field(
        default="None",
        description="Major recurring expenses (mortgage, rent, car payment, etc.)",
    )
    total_debt: str = Field(
        default="None",
        description="Total outstanding debt (e.g., 'minimal', '10k-50k', '50k-100k', etc.)",
    )
    debt_types: Optional[str] = Field(
        default="None",
        description="Types of debt (mortgage, credit card, student loans, etc.)",
    )

    # Assets and Savings
    savings_amount: str = Field(
        default="None",
        description="Amount in savings (e.g., 'none', '1k-5k', '5k-20k', '20k+', etc.)",
    )
    emergency_fund_months: Optional[str] = Field(
        default="None",
        description="Number of months of expenses covered by emergency fund (e.g., '1', '3', '6', etc.)",
    )
    investments: Optional[str] = Field(
        default="None",
        description="Investment portfolio details (stocks, ETFs, crypto, etc.)",
    )
    investment_experience: str = Field(
        default="Beginner",
        description="Investment experience level (beginner, intermediate, advanced)",
    )

    # Financial Goals
    primary_goals: Optional[str] = Field(
        default="Savings",
        description="Primary financial goals (e.g., 'retirement planning', 'home purchase', 'wealth building', etc.)",
    )
    short_term_goals: Optional[str] = Field(
        default="Savings", description="Short-term goals (next 1-2 years)"
    )
    long_term_goals: Optional[str] = Field(
        default="Savings", description="Long-term goals (5+ years)"
    )

    # Risk Profile
    risk_tolerance: str = Field(
        default="Conservative",
        description="Risk tolerance level (conservative, moderate, aggressive)",
    )
    risk_concerns: Optional[str] = Field(
        default="None", description="Any specific financial concerns or risks"
    )

    # Financial Knowledge
    financial_knowledge_level: str = Field(
        default="Beginner",
        description="Financial knowledge level (beginner, intermediate, advanced)",
    )

    # Investment Geography
    geographic_allocation: Optional[str] = Field(
        default="Global balanced",
        description="Geographic investment preference (emerging markets, USA, Europe, global balanced, etc.)",
    )

    # Other Information
    family_dependents: Optional[str] = Field(
        default="None", description="Number of dependents or family situation"
    )
    insurance_coverage: Optional[str] = Field(
        default="None",
        description="Types of insurance coverage (health, life, property, etc.)",
    )
    summary_notes: Optional[str] = Field(
        default="None", description="Any additional important notes or observations"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "financial_profile": {
                "age_range": "30-39",
                "employment_status": "employed",
                "occupation": "Software Engineer",
                "annual_income_range": "80k-120k",
                "income_stability": "stable",
                "additional_income_sources": "None mentioned",
                "monthly_expenses_range": "4k-5k",
                "major_expenses": "Mortgage: $2000/month",
                "total_debt": "150k (mortgage)",
                "debt_types": "Mortgage",
                "savings_amount": "50k",
                "emergency_fund_months": "6",
                "investments": "ETF index funds",
                "investment_experience": "intermediate",
                "primary_goals": "Retirement planning, home equity building",
                "short_term_goals": "Emergency fund optimization",
                "long_term_goals": "Retire at 65 with $2M portfolio",
                "risk_tolerance": "moderate",
                "risk_concerns": "Market volatility concerns",
                "financial_knowledge_level": "intermediate",
                "geographic_allocation": "Global balanced",
                "family_dependents": "2 children",
                "insurance_coverage": "Health, Life, Home insurance",
                "summary_notes": "User shows good financial discipline and awareness",
            }
        }
