"""
Unified financial analysis tool.

This module provides a single tool that combines price retrieval and return calculation
for a comprehensive financial asset analysis.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Union

import yfinance as yf
from datapizza.tools import tool

from src.models.tools import FinancialAnalysisResponse, SymbolResolution, YearReturn

# Configure logger
logger = logging.getLogger(__name__)


@tool
def analyze_financial_asset(ticker: str, years: int = 5) -> str:
    """
    Comprehensive financial asset analysis tool.

    This tool provides a complete analysis of a financial asset by:
    1. Resolving the ticker symbol across global exchanges
    2. Retrieving all available historical price data
    3. Calculating returns for 1, 2, 3, 4, and 5 years (or specified years)
    4. Returning a structured JSON response with all key metrics

    Args:
        ticker: The stock/ETF ticker symbol (e.g., 'SWDA', 'AAPL', 'BND')
        years: Number of years to analyze (default: 5). Returns will be calculated
               for each year from 1 to the specified number of years.

    Returns:
        str: A JSON string containing the analysis response with:
            - success: Whether analysis succeeded
            - ticker: Original ticker
            - resolved_symbol: Resolved symbol with exchange
            - company_name: Company/fund name
            - start_date: Data start date
            - end_date: Data end date
            - years_available: Years of data available
            - data_points: Number of trading days
            - returns: Dict of returns for each year
            - total_return: Total return percentage
            - error: Error message if unsuccessful
    """
    try:
        logger.info("Starting analysis for %s with %d years", ticker, years)

        # Step 1: Resolve symbol
        symbol_resolution = _search_and_resolve_symbol(ticker)
        if not symbol_resolution.success:
            error_msg = symbol_resolution.error or f"Could not resolve symbol: {ticker}"
            logger.error(error_msg)
            response = FinancialAnalysisResponse(
                success=False, ticker=ticker, error=error_msg
            )
            return response.model_dump_json()

        resolved_symbol = symbol_resolution.found_symbol
        company_name = symbol_resolution.company_name
        logger.info("Resolved %s to %s (%s)", ticker, resolved_symbol, company_name)

        # Step 2: Retrieve historical prices
        prices_response = _get_historical_prices_internal(resolved_symbol, years)
        if not prices_response["success"]:
            error_msg = prices_response.get("error", "Failed to retrieve prices")
            logger.error(error_msg)
            response = FinancialAnalysisResponse(
                success=False,
                ticker=ticker,
                resolved_symbol=resolved_symbol,
                error=error_msg,
            )
            return response.model_dump_json()

        prices_list = prices_response["prices"]
        data_points = prices_response["data_points"]
        years_available = prices_response["years_available"]
        start_date = prices_response["start_date"]
        end_date = prices_response["end_date"]

        logger.info(
            "Retrieved %d trading days (%d years) from %s to %s",
            data_points,
            years_available,
            start_date,
            end_date,
        )

        # Step 3: Calculate returns
        returns_response = _calculate_returns_internal(prices_list, years)
        if not returns_response["success"]:
            error_msg = returns_response.get("error", "Failed to calculate returns")
            logger.error(error_msg)
            response = FinancialAnalysisResponse(
                success=False,
                ticker=ticker,
                resolved_symbol=resolved_symbol,
                company_name=company_name,
                error=error_msg,
            )
            return response.model_dump_json()

        returns_data = returns_response["returns"]
        total_return = returns_response["total_return"]

        logger.info("Returns calculated successfully")

        # Convert returns dict to array of YearReturn objects
        returns_array = []
        for year_key, value in returns_data.items():
            # Extract year number from key like "1_year", "2_year", etc.
            year = int(year_key.split("_")[0])
            returns_array.append(YearReturn(year=year, percentage=value))

        # Sort by year
        returns_array.sort(key=lambda x: x.year)

        # Step 4: Build and return the structured response as JSON
        response = FinancialAnalysisResponse(
            success=True,
            ticker=ticker,
            resolved_symbol=resolved_symbol,
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            years_available=years_available,
            data_points=data_points,
            returns=returns_array,
            total_return=total_return,
        )

        logger.info("Analysis completed successfully")
        return response.model_dump_json()

    except Exception as e:
        logger.error("Error during analysis: %s", str(e), exc_info=True)
        response = FinancialAnalysisResponse(
            success=False, ticker=ticker, error=f"Error during analysis: {str(e)}"
        )
        return response.model_dump_json()


def _search_and_resolve_symbol(symbol: str) -> SymbolResolution:
    """
    Search and resolve ticker symbol across global exchanges.

    Args:
        symbol: The ticker symbol to resolve

    Returns:
        SymbolResolution: Object with success status and symbol details
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

        # If no variants worked, return error
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


def _get_historical_prices_internal(
    asset_symbol: str, years: int = None
) -> Dict[str, Any]:
    """
    Retrieve historical price data from Yahoo Finance.

    Args:
        asset_symbol: The resolved ticker symbol
        years: Number of years to retrieve

    Returns:
        Dict with success status, prices list, and metadata
    """
    logger.info("Fetching historical prices for %s", asset_symbol)

    try:
        end_date = datetime.now()

        if years:
            start_date = end_date - timedelta(days=years * 365)
            logger.debug("Fetching %d years of data", years)
        else:
            start_date = end_date - timedelta(days=50 * 365)
            logger.debug("Fetching maximum available historical data (up to 50 years)")

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
            "prices": prices,
            "data_points": len(prices),
            "years_available": round(years_available, 2),
            "start_date": prices[0]["date"],
            "end_date": prices[-1]["date"],
        }

    except Exception as e:
        logger.error("Error fetching prices for %s: %s", asset_symbol, str(e))
        return {
            "success": False,
            "error": f"Failed to fetch data for {asset_symbol}: {str(e)}",
        }


def _calculate_returns_internal(
    prices_list: list, years_requested: int = 5
) -> Dict[str, Any]:
    """
    Calculate returns for each year in the dataset.

    Args:
        prices_list: List of dicts with 'date' and 'close_price'
        years_requested: Number of years to calculate returns for

    Returns:
        Dict with success status, returns dict, and metadata
    """
    logger.info("Calculating returns from %d price points", len(prices_list))

    try:
        if not prices_list or len(prices_list) < 2:
            logger.warning("Insufficient data points")
            return {
                "success": False,
                "error": "At least 2 price points required for return calculation",
            }

        # Sort prices by date
        sorted_prices = sorted(prices_list, key=lambda x: x["date"])
        closing_prices = [price["close_price"] for price in sorted_prices]
        dates = [price["date"] for price in sorted_prices]

        # Get first and last prices
        first_price = closing_prices[0]
        last_price = closing_prices[-1]
        first_date = datetime.strptime(dates[0], "%Y-%m-%d")
        last_date = datetime.strptime(dates[-1], "%Y-%m-%d")

        # Calculate total return
        total_return = ((last_price - first_price) / first_price) * 100
        total_years = (last_date - first_date).days / 365.25

        logger.debug(
            "Total return: %.2f%%, Total years: %.2f", total_return, total_years
        )

        def calculate_cagr(start_price, end_price, years):
            """Calculate Compound Annual Growth Rate."""
            if years <= 0:
                return None
            return (((end_price / start_price) ** (1 / years)) - 1) * 100

        def get_price_at_date(target_date):
            """Find price closest to target date, or first available price."""
            for i in range(len(sorted_prices) - 1, -1, -1):
                price_date = datetime.strptime(dates[i], "%Y-%m-%d")
                if price_date <= target_date:
                    return closing_prices[i]
            # If target date is before all available data, return oldest price
            return closing_prices[0] if sorted_prices else None

        # Calculate returns for years 1 through N
        max_years = years_requested  # Use requested years, not available years
        returns_dict = {}

        for year in range(1, max_years + 1):
            target_date = last_date - timedelta(days=year * 365)
            period_price = get_price_at_date(target_date)

            if period_price is not None:
                if year == 1:
                    # 1-year is simple return
                    period_return = ((last_price - period_price) / period_price) * 100
                else:
                    # Annualized return for year > 1
                    period_return = calculate_cagr(period_price, last_price, year)

                returns_dict[f"{year}_year"] = (
                    round(period_return, 2) if period_return is not None else "N/A"
                )
                logger.debug(
                    "%d-year return: %.2f%%",
                    year,
                    returns_dict[f"{year}_year"]
                    if isinstance(returns_dict[f"{year}_year"], (int, float))
                    else None,
                )
            else:
                returns_dict[f"{year}_year"] = "N/A"
                logger.debug("%d-year: insufficient data", year)

        logger.info("Returns calculated successfully")

        return {
            "success": True,
            "returns": returns_dict,
            "total_return": round(total_return, 2),
            "total_data_points": len(prices_list),
            "date_range": {"start": dates[0], "end": dates[-1]},
            "years_available": round(total_years, 2),
        }

    except Exception as e:
        logger.error("Error calculating returns: %s", str(e), exc_info=True)
        return {
            "success": False,
            "error": f"Failed to calculate returns: {str(e)}",
        }
