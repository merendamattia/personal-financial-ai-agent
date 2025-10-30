"""
Financial Tools Data Models.

This module defines Pydantic models for the return types of financial tools.
These models ensure type safety and serialization compatibility with the Agent.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SymbolResolution(BaseModel):
    """Result of symbol resolution."""

    success: bool = Field(..., description="Whether symbol was found")
    found_symbol: Optional[str] = Field(
        default=None, description="The resolved ticker symbol"
    )
    company_name: Optional[str] = Field(
        default=None, description="Name of the company or fund"
    )
    symbol_type: Optional[str] = Field(
        default=None, description="Type of asset (Equity, ETF, etc.)"
    )
    currency: Optional[str] = Field(default=None, description="Trading currency")
    exchange: Optional[str] = Field(default=None, description="Exchange name")
    error: Optional[str] = Field(
        default=None, description="Error message if unsuccessful"
    )
