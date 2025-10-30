"""
Symbol resolution tools for financial assets.

This module provides tools for searching and resolving ticker symbols.
"""

import logging
from typing import Any, Dict

import yfinance as yf
from datapizza.tools import tool

from src.models.tools import SymbolResolution

# Configure logger
logger = logging.getLogger(__name__)


@tool
def search_and_resolve_symbol(symbol: str) -> SymbolResolution:
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
