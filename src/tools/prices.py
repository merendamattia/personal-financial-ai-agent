"""
Price retrieval tools for financial assets.

This module provides tools for fetching historical price data from Yahoo Finance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

import yfinance as yf
from datapizza.tools import tool

# Configure logger
logger = logging.getLogger(__name__)


@tool
def get_historical_prices(asset_symbol: str, years: int = None) -> Dict[str, Any]:
    """
    Retrieves ALL available historical closing prices for a specified asset.

    This tool fetches ALL available price data from Yahoo Finance for the given
    asset symbol. If years is specified, it fetches at least that many years of data.
    Otherwise, it retrieves the complete historical dataset available.

    Args:
        asset_symbol: The ticker symbol of the asset (e.g., 'SWDA', 'AAPL', 'BND')
        years: Optional minimum number of years to retrieve. If None, fetches all available data.

    Returns:
        dict: Contains:
            - 'success': bool indicating if data was retrieved successfully
            - 'symbol': the asset symbol
            - 'years_requested': years parameter if specified
            - 'prices': List of dicts with 'date' and 'close_price' keys
            - 'data_points': number of trading days in the period
            - 'years_available': number of years of data retrieved
            - 'start_date': earliest date in the dataset
            - 'end_date': latest date in the dataset
            - 'error': error message if unsuccessful

    Example:
        >>> get_historical_prices('SWDA')
        {
            'success': True,
            'symbol': 'SWDA',
            'prices': [
                {'date': '2015-01-02', 'close_price': 45.23},
                {'date': '2015-01-05', 'close_price': 45.15},
                ...
            ],
            'data_points': 2516,
            'years_available': 10.8,
            'start_date': '2015-01-02',
            'end_date': '2025-10-30'
        }
    """
    logger.info("Fetching all available historical prices for %s", asset_symbol)

    try:
        # Calculate date range - use a very old start date to get all available data
        end_date = datetime.now()

        if years:
            # If years specified, use that as minimum
            start_date = end_date - timedelta(days=years * 365)
            logger.debug("Fetching at least %d years of data", years)
        else:
            # Otherwise, go back 50 years to capture as much data as possible
            start_date = end_date - timedelta(days=50 * 365)
            logger.debug("Fetching maximum available historical data (up to 50 years)")

        logger.debug(
            "Date range: %s to %s",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        # Fetch data from Yahoo Finance
        logger.debug("Downloading data from Yahoo Finance for %s", asset_symbol)
        data = yf.download(
            asset_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            logger.warning("No data retrieved for symbol %s", asset_symbol)
            return {
                "success": False,
                "symbol": asset_symbol,
                "error": f"No data found for symbol {asset_symbol}. Please check the ticker symbol.",
            }

        # Convert to list of dicts with date and close_price
        prices = []
        for date, row in data.iterrows():
            close_price = row["Close"]
            if hasattr(close_price, "item"):
                close_price = close_price.item()
            prices.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "close_price": float(round(close_price, 2)),
                }
            )

        # Calculate years of data available
        first_date = datetime.strptime(prices[0]["date"], "%Y-%m-%d")
        last_date = datetime.strptime(prices[-1]["date"], "%Y-%m-%d")
        years_available = (last_date - first_date).days / 365.25

        logger.info(
            "Retrieved %d trading days for %s (%.1f years)",
            len(prices),
            asset_symbol,
            years_available,
        )

        return {
            "success": True,
            "symbol": asset_symbol,
            "years_requested": years,
            "prices": prices,
            "data_points": len(prices),
            "years_available": round(years_available, 2),
            "start_date": prices[0]["date"] if prices else None,
            "end_date": prices[-1]["date"] if prices else None,
        }

    except Exception as e:
        logger.error(
            "Error fetching historical prices for %s: %s", asset_symbol, str(e)
        )
        return {
            "success": False,
            "symbol": asset_symbol,
            "error": f"Failed to fetch data for {asset_symbol}: {str(e)}",
        }
