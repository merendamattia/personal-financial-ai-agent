"""
Financial Tools Data Models.

This module defines Pydantic models for the return types of financial tools.
These models ensure type safety and serialization compatibility with the Agent.
"""

from typing import List, Optional, Union

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


class YearReturn(BaseModel):
    """Return for a specific year."""

    year: int = Field(..., description="Number of years")
    percentage: Union[float, str] = Field(..., description="Return percentage or 'N/A'")


class FinancialAnalysisResponse(BaseModel):
    """Response from unified financial analysis tool."""

    success: bool = Field(..., description="Whether analysis was successful")
    ticker: str = Field(..., description="Original ticker symbol")
    resolved_symbol: Optional[str] = Field(
        default=None, description="Resolved ticker symbol with exchange"
    )
    company_name: Optional[str] = Field(
        default=None, description="Company or fund name"
    )
    start_date: Optional[str] = Field(default=None, description="Start date of data")
    end_date: Optional[str] = Field(default=None, description="End date of data")
    years_available: Optional[float] = Field(
        default=None, description="Years of data available"
    )
    data_points: Optional[int] = Field(
        default=None, description="Number of trading days"
    )
    returns: Optional[List[YearReturn]] = Field(
        default=None, description="Array of returns for each year"
    )
    total_return: Optional[float] = Field(
        default=None, description="Total return percentage"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if unsuccessful"
    )
