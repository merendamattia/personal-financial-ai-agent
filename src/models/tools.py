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


class HistoricalPricesResponse(BaseModel):
    """Response from historical prices retrieval."""

    success: bool = Field(..., description="Whether data was retrieved successfully")
    symbol: str = Field(..., description="The asset symbol")
    prices: List[Dict[str, Any]] = Field(
        default=[],
        description="Historical price data as list of dicts with 'date' and 'close_price'",
    )
    data_points: int = Field(default=0, description="Number of trading days")
    years_available: float = Field(default=0.0, description="Years of data available")
    start_date: Optional[str] = Field(
        default=None, description="Start date of the dataset"
    )
    end_date: Optional[str] = Field(default=None, description="End date of the dataset")
    error: Optional[str] = Field(
        default=None, description="Error message if unsuccessful"
    )


class ReturnsMetrics(BaseModel):
    """Investment returns for different time periods."""

    one_year: Union[float, str] = Field(
        default=None,
        description="1-year simple return (%) or 'N/A' if insufficient data",
    )
    three_year: Union[float, str] = Field(
        default=None,
        description="3-year annualized return (%) or 'N/A' if insufficient data",
    )
    five_year: Union[float, str] = Field(
        default=None,
        description="5-year annualized return (%) or 'N/A' if insufficient data",
    )
    seven_year: Union[float, str] = Field(
        default=None,
        description="7-year annualized return (%) or 'N/A' if insufficient data",
    )
    ten_year: Union[float, str] = Field(
        default=None,
        description="10-year annualized return (%) or 'N/A' if insufficient data",
    )
    fifteen_year: Union[float, str] = Field(
        default=None,
        description="15-year annualized return (%) or 'N/A' if insufficient data",
    )
    twenty_year: Union[float, str] = Field(
        default=None,
        description="20-year annualized return (%) or 'N/A' if insufficient data",
    )
    total_return: Optional[float] = Field(
        default=None, description="Total return over entire period (%)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "1_year": self.one_year,
            "3_year": self.three_year,
            "5_year": self.five_year,
            "7_year": self.seven_year,
            "10_year": self.ten_year,
            "15_year": self.fifteen_year,
            "20_year": self.twenty_year,
            "total_return": self.total_return,
        }


class CalculateReturnsResponse(BaseModel):
    """Response from returns calculation."""

    success: bool = Field(..., description="Whether calculation was successful")
    returns: ReturnsMetrics = Field(
        default_factory=ReturnsMetrics, description="Returns metrics"
    )
    total_data_points: int = Field(default=0, description="Number of data points used")
    date_range: Dict[str, str] = Field(
        default_factory=dict, description="Start and end dates"
    )
    years_available: float = Field(default=0.0, description="Years of data analyzed")
    error: Optional[str] = Field(
        default=None, description="Error message if unsuccessful"
    )
