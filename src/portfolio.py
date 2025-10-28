"""
Portfolio Model Module.

This module defines the Portfolio class that represents
a structured investment portfolio recommendation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Portfolio(BaseModel):
    """Structured portfolio recommendation."""

    # Primary allocation (main assets)
    primary_asset: str = Field(
        ..., description="Primary asset symbol or name (e.g., 'SWDA', etc.)"
    )
    primary_asset_percentage: float = Field(
        ..., description="Percentage allocation for primary asset (0-100)"
    )
    primary_asset_justification: str = Field(
        ..., description="Why this primary asset was chosen for this client profile"
    )

    # Secondary allocation
    secondary_asset: Optional[str] = Field(
        default=None, description="Secondary asset symbol or name"
    )
    secondary_asset_percentage: Optional[float] = Field(
        default=None, description="Percentage allocation for secondary asset (0-100)"
    )
    secondary_asset_justification: Optional[str] = Field(
        default=None, description="Why this secondary asset was chosen"
    )

    # Tertiary allocation
    tertiary_asset: Optional[str] = Field(
        default=None, description="Tertiary asset symbol or name"
    )
    tertiary_asset_percentage: Optional[float] = Field(
        default=None, description="Percentage allocation for tertiary asset (0-100)"
    )
    tertiary_asset_justification: Optional[str] = Field(
        default=None, description="Why this tertiary asset was chosen"
    )

    # Quaternary allocation
    quaternary_asset: Optional[str] = Field(
        default=None, description="Quaternary asset symbol or name"
    )
    quaternary_asset_percentage: Optional[float] = Field(
        default=None, description="Percentage allocation for quaternary asset (0-100)"
    )
    quaternary_asset_justification: Optional[str] = Field(
        default=None, description="Why this quaternary asset was chosen"
    )

    # Quinary allocation
    quinary_asset: Optional[str] = Field(
        default=None, description="Quinary asset symbol or name"
    )
    quinary_asset_percentage: Optional[float] = Field(
        default=None, description="Percentage allocation for quinary asset (0-100)"
    )
    quinary_asset_justification: Optional[str] = Field(
        default=None, description="Why this quinary asset was chosen"
    )

    # Portfolio strategy
    risk_level: str = Field(
        ..., description="Risk level (conservative, moderate, aggressive)"
    )
    portfolio_reasoning: str = Field(
        ...,
        description="Overall explanation of the investment strategy based on client's financial situation",
    )

    # Considerations and recommendations
    key_considerations: str = Field(
        ...,
        description="Key considerations separated by semicolons (e.g., 'consideration1; consideration2; consideration3')",
    )
    rebalancing_schedule: Optional[str] = Field(
        default=None, description="Recommended rebalancing schedule"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "primary_asset": "SWDA",
                "primary_asset_percentage": 60,
                "primary_asset_justification": "Global diversified equity exposure for long-term growth",
                "secondary_asset": "SBXL",
                "secondary_asset_percentage": 30,
                "secondary_asset_justification": "European bonds for stability and income",
                "tertiary_asset": "Gold",
                "tertiary_asset_percentage": 10,
                "tertiary_asset_justification": "Precious metals hedge for portfolio protection",
                "quaternary_asset": None,
                "quaternary_asset_percentage": None,
                "quaternary_asset_justification": None,
                "quinary_asset": None,
                "quinary_asset_percentage": None,
                "quinary_asset_justification": None,
                "risk_level": "moderate",
                "portfolio_reasoning": "This balanced allocation combines growth potential with downside protection, suitable for a moderate investor",
                "key_considerations": "Regular monthly contributions recommended; Review allocation annually; Consider tax-efficient placement",
                "rebalancing_schedule": "Annually or when allocations drift >5%",
            }
        }
