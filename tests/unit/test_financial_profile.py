"""
Tests for Financial Profile model.

Tests the FinancialProfile Pydantic model.
"""

import pytest
from pydantic import ValidationError

from src.models import FinancialProfile


class TestFinancialProfileCreation:
    """Tests for FinancialProfile model creation and validation."""

    def test_create_valid_profile(self, sample_financial_profile):
        """Test creating a valid financial profile."""
        assert sample_financial_profile is not None
        assert sample_financial_profile.age_range == "30-39"
        assert sample_financial_profile.employment_status == "employed"
        assert sample_financial_profile.annual_income_range == "80k-120k"

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        profile = FinancialProfile(
            age_range="25-34",
            employment_status="self-employed",
            annual_income_range="50k-100k",
            income_stability="moderate",
            monthly_expenses_range="2k-3k",
            total_debt="minimal",
            monthly_savings_amount="20k",
            investment_experience="beginner",
            risk_tolerance="conservative",
        )
        assert profile.summary_notes == "None"

    def test_geographic_allocation_default(self):
        """Test that geographic_allocation has correct default."""
        profile = FinancialProfile(
            age_range="30-39",
            employment_status="employed",
            annual_income_range="80k-120k",
            income_stability="stable",
            monthly_expenses_range="4k-5k",
            total_debt="minimal",
            monthly_savings_amount="50k",
            investment_experience="intermediate",
            risk_tolerance="moderate",
        )
        assert profile.geographic_allocation == "Global balanced"

    def test_custom_geographic_allocation(self):
        """Test setting custom geographic allocation."""
        profile = FinancialProfile(
            age_range="30-39",
            employment_status="employed",
            annual_income_range="80k-120k",
            income_stability="stable",
            monthly_expenses_range="4k-5k",
            total_debt="minimal",
            monthly_savings_amount="50k",
            investment_experience="intermediate",
            risk_tolerance="moderate",
            geographic_allocation="Emerging markets",
        )
        assert profile.geographic_allocation == "Emerging markets"

    def test_model_dump(self, sample_financial_profile):
        """Test converting profile to dictionary."""
        profile_dict = sample_financial_profile.model_dump()
        assert isinstance(profile_dict, dict)
        assert "age_range" in profile_dict
        assert profile_dict["age_range"] == "30-39"

    def test_model_dump_json(self, sample_financial_profile):
        """Test converting profile to JSON."""
        profile_json = sample_financial_profile.model_dump_json()
        assert isinstance(profile_json, str)
        assert "age_range" in profile_json
        assert "30-39" in profile_json


class TestFinancialProfileValidation:
    """Tests for FinancialProfile validation."""

    def test_income_stability_values(self):
        """Test various income stability values."""
        for stability in ["stable", "moderate", "unstable"]:
            profile = FinancialProfile(
                age_range="30-39",
                employment_status="employed",
                annual_income_range="80k-120k",
                income_stability=stability,
                monthly_expenses_range="4k-5k",
                total_debt="minimal",
                monthly_savings_amount="50k",
                investment_experience="intermediate",
                risk_tolerance="moderate",
            )
            assert profile.income_stability == stability

    def test_risk_tolerance_values(self):
        """Test various risk tolerance values."""
        for risk in ["conservative", "moderate", "aggressive"]:
            profile = FinancialProfile(
                age_range="30-39",
                employment_status="employed",
                annual_income_range="80k-120k",
                income_stability="stable",
                monthly_expenses_range="4k-5k",
                total_debt="minimal",
                monthly_savings_amount="50k",
                investment_experience="intermediate",
                risk_tolerance=risk,
            )
            assert profile.risk_tolerance == risk

    def test_investment_experience_values(self):
        """Test various investment experience values."""
        for exp in ["beginner", "intermediate", "advanced"]:
            profile = FinancialProfile(
                age_range="30-39",
                employment_status="employed",
                annual_income_range="80k-120k",
                income_stability="stable",
                monthly_expenses_range="4k-5k",
                total_debt="minimal",
                monthly_savings_amount="50k",
                investment_experience=exp,
                risk_tolerance="moderate",
            )
            assert profile.investment_experience == exp
