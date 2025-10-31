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
    investments: Optional[str] = Field(
        default="None",
        description="Investment portfolio details (stocks, ETFs, crypto, etc.)",
    )
    investment_experience: str = Field(
        default="Beginner",
        description="Investment experience level (beginner, intermediate, advanced)",
    )

    # Financial Goals
    goals: Optional[str] = Field(
        default="Savings",
        description="Primary financial goals (e.g., 'retirement planning', 'home purchase', 'wealth building', etc.)",
    )

    # Risk Profile
    risk_tolerance: str = Field(
        default="Conservative",
        description="Risk tolerance level (conservative, moderate, aggressive)",
    )
    risk_concerns: Optional[str] = Field(
        default="None", description="Any specific financial concerns or risks"
    )

    # Family and Insurance
    family_dependents: Optional[str] = Field(
        default="None", description="Number of dependents or family situation"
    )
    insurance_coverage: Optional[str] = Field(
        default="None",
        description="Types of insurance coverage (health, life, property, etc.)",
    )

    # Investment Geography
    geographic_allocation: Optional[str] = Field(
        default="Global balanced",
        description="Geographic investment preference (emerging markets, USA, Europe, global balanced, etc.)",
    )

    # Additional Notes
    summary_notes: Optional[str] = Field(
        default="None", description="Any additional important notes or observations"
    )
