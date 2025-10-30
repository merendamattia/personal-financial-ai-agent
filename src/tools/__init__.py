"""
Financial Tools Package.

This package contains tools for retrieving and analyzing financial asset data.

Available tools:
- search_and_resolve_symbol: Find the correct ticker symbol
- get_historical_prices: Retrieve historical price data
- calculate_returns: Calculate investment returns from price data
"""

from .prices import get_historical_prices
from .returns import calculate_returns
from .symbols import search_and_resolve_symbol

__all__ = [
    "get_historical_prices",
    "calculate_returns",
    "search_and_resolve_symbol",
]
