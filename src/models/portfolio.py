"""
Portfolio Model Module.

This module defines the Portfolio class that represents
a structured investment portfolio recommendation.
"""

import math
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

    # @model_validator(mode="after")
    # def validate_total_percentage(self):
    #     """Validate that asset percentages sum to approximately 100."""
    #     total = sum(asset.percentage for asset in self.assets)
    #     if not (99 <= total <= 101):
    #         raise ValueError(
    #             f"Asset percentages must sum to 100%, got {total}% instead"
    #         )
    #     return self

    # TODO: completare nuova versione del model validator e eliminare la vecchia qui sopra
    @model_validator(mode="after")
    def normalize_total_percentage(self):
        """
        Auto-correct asset percentages to ensure they sum to exactly 100%.
        Instead of raising an error, specifically re-proportions the values.
        """
        if not self.assets:
            return self

        total = sum(asset.percentage for asset in self.assets)

        # Se il totale è 0, non possiamo normalizzare (evitiamo divisione per zero)
        if total == 0:
            # In questo caso estremo, assegniamo tutto al primo asset o lanciamo errore
            # Qui scegliamo di distribuire equamente per non rompere l'app
            share = 100.0 / len(self.assets)
            for asset in self.assets:
                asset.percentage = round(share, 2)
            return self

        # Se la somma è già corretta (con tolleranza), non facciamo nulla
        if 99.0 <= total <= 101.0:
            return self

        # LOGICA DI NORMALIZZAZIONE
        # Esempio: Se il totale è 60%, il fattore è 100/60 = 1.666...
        factor = 100.0 / total

        for asset in self.assets:
            # Scaliamo ogni asset
            asset.percentage = round(asset.percentage * factor, 2)

        # Controllo finale per arrotondamenti (es. somma 99.99 o 100.01)
        new_total = sum(asset.percentage for asset in self.assets)
        diff = 100.0 - new_total

        if diff != 0:
            # Aggiungiamo/togliamo la piccola differenza al primo asset (spesso il più grande)
            self.assets[0].percentage = round(self.assets[0].percentage + diff, 2)

        return self
