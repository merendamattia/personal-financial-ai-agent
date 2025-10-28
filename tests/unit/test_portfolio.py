"""
Tests for Portfolio model.

Tests the Portfolio Pydantic model with nested Asset allocations.
"""

import pytest
from pydantic import ValidationError

from src.models import Portfolio
from src.models.portfolio import Asset, RiskLevel


class TestAssetCreation:
    """Tests for Asset model creation."""

    def test_create_valid_asset(self):
        """Test creating a valid asset."""
        asset = Asset(
            symbol="SWDA",
            percentage=60.0,
            justification="Global diversified equity exposure",
        )
        assert asset.symbol == "SWDA"
        assert asset.percentage == 60.0
        assert asset.justification == "Global diversified equity exposure"

    def test_asset_percentage_validation(self):
        """Test asset percentage must be between 0-100."""
        # Valid: 0 <= percentage <= 100
        asset = Asset(symbol="TEST", percentage=0, justification="Test")
        assert asset.percentage == 0

        asset = Asset(symbol="TEST", percentage=100, justification="Test")
        assert asset.percentage == 100

        # Invalid: percentage > 100
        with pytest.raises(ValidationError):
            Asset(symbol="TEST", percentage=101, justification="Test")

        # Invalid: percentage < 0
        with pytest.raises(ValidationError):
            Asset(symbol="TEST", percentage=-1, justification="Test")

    def test_asset_required_fields(self):
        """Test that all asset fields are required."""
        # Missing symbol
        with pytest.raises(ValidationError):
            Asset(percentage=50, justification="Test")

        # Missing percentage
        with pytest.raises(ValidationError):
            Asset(symbol="TEST", justification="Test")

        # Missing justification
        with pytest.raises(ValidationError):
            Asset(symbol="TEST", percentage=50)


class TestPortfolioCreation:
    """Tests for Portfolio model creation."""

    def test_create_valid_portfolio(self, sample_portfolio):
        """Test creating a valid portfolio."""
        assert sample_portfolio is not None
        assert len(sample_portfolio.assets) == 3
        assert sample_portfolio.assets[0].symbol == "SWDA"
        assert sample_portfolio.risk_level == "moderate"

    def test_portfolio_with_single_asset(self):
        """Test portfolio with single asset."""
        portfolio = Portfolio(
            assets=[
                Asset(
                    symbol="SWDA",
                    percentage=100.0,
                    justification="Full allocation in global ETF",
                )
            ],
            risk_level="moderate",
            portfolio_reasoning="Simple one-asset strategy",
            key_considerations=["Monitor global market conditions"],
            rebalancing_schedule="Annually",
        )
        assert len(portfolio.assets) == 1
        assert portfolio.assets[0].percentage == 100.0

    def test_portfolio_with_multiple_assets(self, sample_portfolio):
        """Test portfolio with multiple assets."""
        assert len(sample_portfolio.assets) == 3
        assert sample_portfolio.assets[0].symbol == "SWDA"
        assert sample_portfolio.assets[1].symbol == "SBXL"
        assert sample_portfolio.assets[2].symbol == "Gold"

    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing assets
        with pytest.raises(ValidationError):
            Portfolio(
                risk_level="moderate",
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )

        # Missing risk_level
        with pytest.raises(ValidationError):
            Portfolio(
                assets=[Asset(symbol="TEST", percentage=100.0, justification="Test")],
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )

    def test_asset_list_constraints(self):
        """Test that asset list has min/max constraints."""
        # At least 1 asset required
        with pytest.raises(ValidationError):
            Portfolio(
                assets=[],
                risk_level="moderate",
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )

        # Max 10 assets allowed
        too_many_assets = [
            Asset(
                symbol=f"ASSET{i}",
                percentage=100 / 11,
                justification=f"Asset {i}",
            )
            for i in range(11)
        ]
        with pytest.raises(ValidationError):
            Portfolio(
                assets=too_many_assets,
                risk_level="moderate",
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )

    def test_key_considerations_constraints(self):
        """Test that key_considerations has at least 1 item."""
        # At least 1 consideration required
        with pytest.raises(ValidationError):
            Portfolio(
                assets=[Asset(symbol="TEST", percentage=100.0, justification="Test")],
                risk_level="moderate",
                portfolio_reasoning="Test",
                key_considerations=[],
                rebalancing_schedule="Annually",
            )


class TestPortfolioValidation:
    """Tests for Portfolio validation."""

    def test_percentage_sum_must_equal_100(self):
        """Test that asset percentages must sum to 100%."""
        # Valid: sum = 100
        portfolio = Portfolio(
            assets=[
                Asset(symbol="A", percentage=60.0, justification="Test A"),
                Asset(symbol="B", percentage=40.0, justification="Test B"),
            ],
            risk_level="moderate",
            portfolio_reasoning="Test",
            key_considerations=["Test"],
            rebalancing_schedule="Annually",
        )
        assert portfolio is not None

        # Valid: sum = 100.5 (within tolerance)
        portfolio = Portfolio(
            assets=[
                Asset(symbol="A", percentage=50.25, justification="Test A"),
                Asset(symbol="B", percentage=50.25, justification="Test B"),
            ],
            risk_level="moderate",
            portfolio_reasoning="Test",
            key_considerations=["Test"],
            rebalancing_schedule="Annually",
        )
        assert portfolio is not None

        # Invalid: sum < 99 (e.g., 98.5)
        with pytest.raises(ValueError, match="must sum to 100%"):
            Portfolio(
                assets=[
                    Asset(symbol="A", percentage=50.0, justification="Test A"),
                    Asset(symbol="B", percentage=48.5, justification="Test B"),
                ],
                risk_level="moderate",
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )

        # Invalid: sum > 101 (e.g., 101.5)
        with pytest.raises(ValueError, match="must sum to 100%"):
            Portfolio(
                assets=[
                    Asset(symbol="A", percentage=51.0, justification="Test A"),
                    Asset(symbol="B", percentage=50.5, justification="Test B"),
                ],
                risk_level="moderate",
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )

    def test_risk_level_enum(self):
        """Test risk_level enum values."""
        for risk in RiskLevel:
            portfolio = Portfolio(
                assets=[
                    Asset(
                        symbol="TEST",
                        percentage=100.0,
                        justification="Test",
                    )
                ],
                risk_level=risk,
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )
            assert portfolio.risk_level == risk

    def test_risk_level_enum_values(self):
        """Test specific risk level values."""
        assert RiskLevel.CONSERVATIVE.value == "conservative"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.AGGRESSIVE.value == "aggressive"

    def test_invalid_risk_level(self):
        """Test that invalid risk level raises error."""
        with pytest.raises(ValidationError):
            Portfolio(
                assets=[Asset(symbol="TEST", percentage=100.0, justification="Test")],
                risk_level="invalid_risk",
                portfolio_reasoning="Test",
                key_considerations=["Test"],
                rebalancing_schedule="Annually",
            )


class TestPortfolioSerialization:
    """Tests for Portfolio serialization."""

    def test_model_dump(self, sample_portfolio):
        """Test converting portfolio to dictionary."""
        portfolio_dict = sample_portfolio.model_dump()
        assert isinstance(portfolio_dict, dict)
        assert "assets" in portfolio_dict
        assert len(portfolio_dict["assets"]) == 3
        assert portfolio_dict["risk_level"] == "moderate"

    def test_model_dump_json(self, sample_portfolio):
        """Test converting portfolio to JSON string."""
        json_str = sample_portfolio.model_dump_json()
        assert isinstance(json_str, str)
        assert "SWDA" in json_str
        assert "moderate" in json_str

    def test_model_dump_with_nested_assets(self, sample_portfolio):
        """Test that nested assets are properly serialized."""
        portfolio_dict = sample_portfolio.model_dump()
        assets = portfolio_dict["assets"]
        assert len(assets) == 3
        assert assets[0]["symbol"] == "SWDA"
        assert assets[0]["percentage"] == 60.0
        assert "justification" in assets[0]

    def test_model_validate_from_dict(self):
        """Test creating portfolio from dictionary."""
        portfolio_data = {
            "assets": [
                {"symbol": "SWDA", "percentage": 100.0, "justification": "Test"}
            ],
            "risk_level": "moderate",
            "portfolio_reasoning": "Test reasoning",
            "key_considerations": ["Consider rebalancing"],
            "rebalancing_schedule": "Annually",
        }
        portfolio = Portfolio.model_validate(portfolio_data)
        assert portfolio.assets[0].symbol == "SWDA"
        assert portfolio.risk_level == "moderate"


class TestPortfolioExample:
    """Test portfolio against the example in json_schema_extra."""

    def test_example_portfolio_structure(self, sample_portfolio):
        """Verify sample portfolio matches expected structure."""
        # Assets structure
        assert len(sample_portfolio.assets) == 3
        assert all(isinstance(asset, Asset) for asset in sample_portfolio.assets)

        # Risk level
        assert sample_portfolio.risk_level in [
            RiskLevel.CONSERVATIVE,
            RiskLevel.MODERATE,
            RiskLevel.AGGRESSIVE,
        ]

        # Reasoning and considerations
        assert isinstance(sample_portfolio.portfolio_reasoning, str)
        assert isinstance(sample_portfolio.key_considerations, list)
        assert len(sample_portfolio.key_considerations) > 0

        # Rebalancing schedule
        assert isinstance(sample_portfolio.rebalancing_schedule, str)
