"""
Price retrieval tools for financial assets.

This module provides tools for fetching historical price data from Yahoo Finance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

import yfinance as yf
from datapizza.tools import tool

from src.models.tools import HistoricalPricesResponse, PricePoint, SymbolResolution

# Configure logger
logger = logging.getLogger(__name__)


@tool
def get_historical_prices(
    asset_symbol: str, years: int = None
) -> HistoricalPricesResponse:
    """
    Retrieves ALL available historical closing prices for a specified asset.

    This tool fetches ALL available price data from Yahoo Finance for the given
    asset symbol. If years is specified, it fetches at least that many years of data.
    Otherwise, it retrieves the complete historical dataset available.

    Args:
        asset_symbol: The ticker symbol of the asset (e.g., 'SWDA', 'AAPL', 'BND')
        years: Optional minimum number of years to retrieve. If None, fetches all available data.

    Returns:
        HistoricalPricesResponse: Structured response containing:
            - success: bool indicating if data was retrieved successfully
            - symbol: the asset symbol
            - prices: List of PricePoint objects with 'date' and 'close_price'
            - data_points: number of trading days in the period
            - years_available: number of years of data retrieved
            - start_date: earliest date in the dataset
            - end_date: latest date in the dataset
            - error: error message if unsuccessful

    Example:
        >>> get_historical_prices('SWDA')
        HistoricalPricesResponse(
            success=True,
            symbol='SWDA',
            prices=[
                PricePoint(date='2015-01-02', close_price=45.23),
                PricePoint(date='2015-01-05', close_price=45.15),
                ...
            ],
            data_points=2516,
            years_available=10.8,
            start_date='2015-01-02',
            end_date='2025-10-30'
        )
    """
    # Try to resolve the symbol to handle variations (e.g., SWDA -> SWDA.L)
    symbol_resolution = _search_and_resolve_symbol(asset_symbol)
    if symbol_resolution.success and symbol_resolution.found_symbol:
        resolved_symbol = symbol_resolution.found_symbol
        logger.info("Resolved symbol: %s -> %s", asset_symbol, resolved_symbol)
        asset_symbol = resolved_symbol

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
            return HistoricalPricesResponse(
                success=False,
                symbol=asset_symbol,
                error=f"No data found for symbol {asset_symbol}. Please check the ticker symbol.",
            )

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

        return HistoricalPricesResponse(
            success=True,
            symbol=asset_symbol,
            prices=prices,
            data_points=len(prices),
            years_available=round(years_available, 2),
            start_date=prices[0]["date"] if prices else None,
            end_date=prices[-1]["date"] if prices else None,
        )

    except Exception as e:
        logger.error(
            "Error fetching historical prices for %s: %s", asset_symbol, str(e)
        )
        return HistoricalPricesResponse(
            success=False,
            symbol=asset_symbol,
            error=f"Failed to fetch data for {asset_symbol}: {str(e)}",
        )


def _search_and_resolve_symbol(symbol: str) -> SymbolResolution:
    """
    Searches for the correct ticker symbol when given an invalid or partial symbol.

    When a ticker symbol is not found or is incorrect, this tool attempts to find
    the correct symbol by:
    1. Testing if the symbol is valid directly
    2. If not found, searching Yahoo Finance for similar symbols
    3. Returning the most likely correct symbol with details

    Args:
        symbol: The ticker symbol to search for (e.g., 'SWDA', 'AAPL', 'BND')

    Returns:
        SymbolResolution: Structured response containing:
            - success: bool indicating if a symbol was resolved
            - found_symbol: the correct ticker symbol found
            - company_name: the name of the company/fund
            - symbol_type: type of asset (Stock, ETF, Fund, etc.)
            - currency: the currency in which the asset trades
            - exchange: the exchange where it trades
            - error: error message if unsuccessful
    """
    logger.info("Searching for symbol: %s", symbol)

    try:
        variations = [
            f"{symbol.upper()}.MI",  # Borsa Italiana (Milan)
            f"{symbol.upper()}.DE",  # Deutsche Börse (XETRA - Germany)
            f"{symbol.upper()}.PA",  # Euronext Paris (France)
            f"{symbol.upper()}.AS",  # Euronext Amsterdam (Netherlands)
            f"{symbol.upper()}.BR",  # Euronext Brussels (Belgium)
            f"{symbol.upper()}.LS",  # Euronext Lisbon (Portugal)
            f"{symbol.upper()}.SW",  # SIX Swiss Exchange (Switzerland)
            f"{symbol.upper()}.L",  # London Stock Exchange (United Kingdom)
            f"{symbol.upper()}.CO",  # Nasdaq Copenhagen (Denmark)
            f"{symbol.upper()}.ST",  # Nasdaq Stockholm (Sweden)
            f"{symbol.upper()}.HE",  # Nasdaq Helsinki (Finland)
            f"{symbol.upper()}.OL",  # Oslo Børs (Norway)
            f"{symbol.upper()}.IR",  # Irish Stock Exchange (Ireland)
            f"{symbol.upper()}.VI",  # Vienna Stock Exchange (Austria)
            f"{symbol.upper()}.WA",  # Warsaw Stock Exchange (Poland)
            f"{symbol.upper()}.PR",  # Prague Stock Exchange (Czech Republic)
            f"{symbol.upper()}.BD",  # Budapest Stock Exchange (Hungary)
            f"{symbol.upper()}.AT",  # Athens Stock Exchange (Greece)
            f"{symbol.upper()}.IS",  # Borsa Istanbul (Turkey)
            symbol.upper(),  # fallback: no suffix (e.g. NASDAQ/NYSE)
        ]

        for variant in variations:
            try:
                ticker_variant = yf.Ticker(variant)
                info_variant = ticker_variant.info

                if info_variant and "longName" in info_variant:
                    logger.info(
                        "Found symbol variant: %s for input: %s", variant, symbol
                    )
                    return SymbolResolution(
                        success=True,
                        found_symbol=variant,
                        company_name=info_variant.get(
                            "longName", info_variant.get("shortName", "Unknown")
                        ),
                        symbol_type=info_variant.get("quoteType", "Unknown"),
                        currency=info_variant.get("currency", "Unknown"),
                        exchange=info_variant.get("exchange", "Unknown"),
                    )
            except Exception as e:
                logger.debug("Variant %s failed: %s", variant, str(e))
                continue

        # If no variants worked, return error with suggestions
        logger.warning("No valid symbol found for: %s", symbol)
        return SymbolResolution(
            success=False,
            error=f"Symbol '{symbol}' not found. Please verify the ticker symbol.",
        )

    except Exception as e:
        logger.error("Error searching for symbol %s: %s", symbol, str(e))
        return SymbolResolution(
            success=False,
            error=f"Failed to search for symbol {symbol}: {str(e)}",
        )
