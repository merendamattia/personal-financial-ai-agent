"""
Portfolio Model Module.

This module defines the Portfolio class that represents
a structured investment portfolio recommendation.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskLevel(str, Enum):
    """Portfolio risk levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Asset(BaseModel):
    """Individual asset allocation within a portfolio."""

    symbol: str = Field(
        ..., description="Asset symbol or name (e.g., 'SWDA', 'SBXL', 'Gold')"
    )
    percentage: float = Field(
        ..., ge=0, le=100, description="Percentage allocation for this asset (0-100)"
    )
    justification: str = Field(
        ..., description="Why this asset was chosen for this client profile"
    )


class Portfolio(BaseModel):
    """Structured portfolio recommendation with nested asset allocations."""

    # Asset allocations
    assets: List[Asset] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of assets in the portfolio (min 1, max 10)",
    )

    # Portfolio strategy
    risk_level: RiskLevel = Field(
        ..., description="Risk level (conservative, moderate, aggressive)"
    )
    portfolio_reasoning: str = Field(
        ...,
        description="Overall explanation of the investment strategy based on client's financial situation",
    )

    # Considerations and recommendations
    key_considerations: List[str] = Field(
        ...,
        min_items=1,
        description="Key considerations for this portfolio (e.g., 'Regular monthly contributions recommended', 'Review allocation annually')",
    )
    rebalancing_schedule: str = Field(
        ...,
        description="Recommended rebalancing schedule (e.g., 'Annually or when allocations drift >5%')",
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "assets": [
                    {
                        "symbol": "SWDA",
                        "percentage": 60,
                        "justification": "Global diversified equity exposure for long-term growth",
                    },
                    {
                        "symbol": "SBXL",
                        "percentage": 30,
                        "justification": "European bonds for stability and income",
                    },
                    {
                        "symbol": "Gold",
                        "percentage": 10,
                        "justification": "Precious metals hedge for portfolio protection",
                    },
                ],
                "risk_level": "moderate",
                "portfolio_reasoning": "This balanced allocation combines growth potential with downside protection, suitable for a moderate investor with intermediate experience",
                "key_considerations": [
                    "Regular monthly contributions recommended",
                    "Review allocation annually",
                    "Consider tax-efficient placement",
                ],
                "rebalancing_schedule": "Annually or when allocations drift >5%",
            }
        },
    )

    @model_validator(mode="after")
    def validate_total_percentage(self):
        """Validate that asset percentages sum to approximately 100."""
        total = sum(asset.percentage for asset in self.assets)
        if not (99 <= total <= 101):
            raise ValueError(
                f"Asset percentages must sum to 100%, got {total}% instead"
            )
        return self
