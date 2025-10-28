"""
Tests for Portfolio model.

Tests the Portfolio Pydantic model.
"""

import pytest
from pydantic import ValidationError

from src.models import Portfolio


class TestPortfolioCreation:
    """Tests for Portfolio model creation and validation."""

    def test_create_valid_portfolio(self, sample_portfolio):
        """Test creating a valid portfolio."""
        assert sample_portfolio is not None
        assert sample_portfolio.primary_asset == "SWDA"
        assert sample_portfolio.primary_asset_percentage == 60.0
        assert sample_portfolio.risk_level == "moderate"

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            Portfolio(
                primary_asset="SWDA",
                primary_asset_percentage=60.0,
                # Missing primary_asset_justification
                risk_level="moderate",
                portfolio_reasoning="Test",
                key_considerations="Test",
            )

    def test_optional_secondary_asset(self):
        """Test optional secondary asset fields."""
        portfolio = Portfolio(
            primary_asset="SWDA",
            primary_asset_percentage=100.0,
            primary_asset_justification="Full allocation",
            risk_level="moderate",
            portfolio_reasoning="Single asset portfolio",
            key_considerations="Simple strategy",
            # Optional secondary fields not set
        )
        assert portfolio.secondary_asset is None
        assert portfolio.secondary_asset_percentage is None
        assert portfolio.secondary_asset_justification is None

    def test_full_portfolio_with_all_assets(self, sample_portfolio):
        """Test portfolio with all 5 asset types."""
        assert sample_portfolio.primary_asset is not None
        assert sample_portfolio.secondary_asset is not None
        assert sample_portfolio.tertiary_asset is not None
        assert sample_portfolio.quaternary_asset is None
        assert sample_portfolio.quinary_asset is None

    def test_portfolio_percentages(self):
        """Test portfolio with specific percentages."""
        portfolio = Portfolio(
            primary_asset="SWDA",
            primary_asset_percentage=50.0,
            primary_asset_justification="Global equity",
            secondary_asset="SBXL",
            secondary_asset_percentage=30.0,
            secondary_asset_justification="Bonds",
            tertiary_asset="Gold",
            tertiary_asset_percentage=20.0,
            tertiary_asset_justification="Hedge",
            risk_level="moderate",
            portfolio_reasoning="Balanced portfolio",
            key_considerations="Diversified",
        )
        assert portfolio.primary_asset_percentage == 50.0
        assert portfolio.secondary_asset_percentage == 30.0
        assert portfolio.tertiary_asset_percentage == 20.0

    def test_model_dump(self, sample_portfolio):
        """Test converting portfolio to dictionary."""
        portfolio_dict = sample_portfolio.model_dump()
        assert isinstance(portfolio_dict, dict)
        assert "primary_asset" in portfolio_dict
        assert portfolio_dict["primary_asset"] == "SWDA"
        assert portfolio_dict["risk_level"] == "moderate"

    def test_model_dump_exclude_none(self, sample_portfolio):
        """Test converting portfolio to dict excluding None values."""
        portfolio_dict = sample_portfolio.model_dump(exclude_none=True)
        assert "quaternary_asset" not in portfolio_dict
        assert "quinary_asset" not in portfolio_dict
        assert "primary_asset" in portfolio_dict


class TestPortfolioValidation:
    """Tests for Portfolio validation."""

    def test_risk_level_values(self):
        """Test various risk level values."""
        for risk in ["conservative", "moderate", "aggressive"]:
            portfolio = Portfolio(
                primary_asset="SWDA",
                primary_asset_percentage=100.0,
                primary_asset_justification="Test",
                risk_level=risk,
                portfolio_reasoning="Test portfolio",
                key_considerations="Test",
            )
            assert portfolio.risk_level == risk

    def test_percentage_values(self):
        """Test portfolio with different percentage values."""
        portfolio = Portfolio(
            primary_asset="Asset1",
            primary_asset_percentage=25.5,
            primary_asset_justification="Quarter allocation",
            secondary_asset="Asset2",
            secondary_asset_percentage=75.5,
            secondary_asset_justification="Three quarters",
            risk_level="moderate",
            portfolio_reasoning="Test",
            key_considerations="Test",
        )
        assert portfolio.primary_asset_percentage == 25.5
        assert portfolio.secondary_asset_percentage == 75.5

    def test_asset_justification_required(self):
        """Test that asset justifications are required when asset is defined."""
        portfolio = Portfolio(
            primary_asset="SWDA",
            primary_asset_percentage=60.0,
            primary_asset_justification="Global equity",
            secondary_asset="SBXL",
            secondary_asset_percentage=40.0,
            secondary_asset_justification="Bonds",
            risk_level="moderate",
            portfolio_reasoning="Test",
            key_considerations="Test",
        )
        assert portfolio.secondary_asset_justification == "Bonds"

    def test_key_considerations_parsing(self):
        """Test key considerations with semicolon separation."""
        considerations = "Rebalance quarterly; Review annually; Tax-efficient placement"
        portfolio = Portfolio(
            primary_asset="SWDA",
            primary_asset_percentage=100.0,
            primary_asset_justification="Test",
            risk_level="moderate",
            portfolio_reasoning="Test",
            key_considerations=considerations,
        )
        assert "Rebalance quarterly" in portfolio.key_considerations
        assert "Review annually" in portfolio.key_considerations
        assert "Tax-efficient placement" in portfolio.key_considerations

    def test_rebalancing_schedule_optional(self):
        """Test that rebalancing schedule is optional."""
        portfolio = Portfolio(
            primary_asset="SWDA",
            primary_asset_percentage=100.0,
            primary_asset_justification="Test",
            risk_level="moderate",
            portfolio_reasoning="Test",
            key_considerations="Test",
            rebalancing_schedule=None,
        )
        assert portfolio.rebalancing_schedule is None

    def test_with_rebalancing_schedule(self):
        """Test portfolio with rebalancing schedule."""
        schedule = "Annually or when drift >5%"
        portfolio = Portfolio(
            primary_asset="SWDA",
            primary_asset_percentage=100.0,
            primary_asset_justification="Test",
            risk_level="moderate",
            portfolio_reasoning="Test",
            key_considerations="Test",
            rebalancing_schedule=schedule,
        )
        assert portfolio.rebalancing_schedule == schedule
