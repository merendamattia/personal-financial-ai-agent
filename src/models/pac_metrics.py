"""
PAC Metrics Data Model.

This module defines the Pydantic model for extracting structured
PAC (Piano di Accumulo del Capitale) metrics from financial profiles.
"""

from pydantic import BaseModel, Field


class PACMetrics(BaseModel):
    """Structured PAC metrics extracted from financial profile."""

    initial_investment: float = Field(
        default=5000,
        description="Initial investment amount in euros (e.g., existing savings or lump sum to invest)",
    )
    monthly_savings: float = Field(
        default=200,
        description="Monthly savings capacity in euros (how much can be invested monthly)",
    )
