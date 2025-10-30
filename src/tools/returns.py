"""
Return calculation tools for financial analysis.

This module provides tools for calculating investment returns from price data.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from datapizza.tools import tool

from src.models.tools import CalculateReturnsResponse, ReturnsMetrics

# Configure logger
logger = logging.getLogger(__name__)


@tool
def calculate_returns(prices: str) -> CalculateReturnsResponse:
    """
    Calculates various return metrics from a list of historical prices.

    This tool takes historical price data and computes annualized returns for multiple periods:
    - 1-year return
    - 3-year annualized return
    - 5-year annualized return
    - 7-year annualized return
    - 10-year annualized return
    - 15-year annualized return
    - 20-year annualized return
    - Total return (from first to last price)

    The annualized return is calculated using the Compound Annual Growth Rate (CAGR) formula:
    CAGR = (End Value / Start Value) ^ (1 / Number of Years) - 1

    Returns "N/A" for periods where sufficient data is not available.

    Args:
        prices: JSON string containing list of dicts with 'date' and 'close_price' keys.
                Example: '[{"date": "2020-01-01", "close_price": 100.0}, ...]'
                Typically generated from get_historical_prices()

    Returns:
        CalculateReturnsResponse: Structured response containing:
            - success: bool indicating if calculation was successful
            - returns: ReturnsMetrics object with calculated returns
            - total_data_points: number of prices in the input
            - date_range: dict with 'start' and 'end' dates
            - years_available: total years of data available
            - error: error message if unsuccessful

    Example:
        >>> calculate_returns('[{"date": "2015-01-02", "close_price": 45.23}, ...]')
        CalculateReturnsResponse(
            success=True,
            total_data_points=2516,
            date_range={'start': '2015-01-02', 'end': '2025-10-30'},
            years_available=10.8,
            returns=ReturnsMetrics(
                one_year=12.45,
                three_year=8.32,
                five_year=7.15,
                ...
            )
        )
    """
    logger.info("Calculating returns from price data")

    try:
        # Parse JSON string to list
        if isinstance(prices, str):
            prices_list = json.loads(prices)
        else:
            prices_list = prices

        logger.info("Received %d prices", len(prices_list) if prices_list else 0)
        if prices_list:
            logger.debug("First price: %s", prices_list[0])
            logger.debug("Last price: %s", prices_list[-1])

        if not prices_list or len(prices_list) < 2:
            logger.warning("Insufficient data points for return calculation")
            return CalculateReturnsResponse(
                success=False,
                error="At least 2 price points required for return calculation",
            )

        # Sort prices by date to ensure correct order
        sorted_prices = sorted(prices_list, key=lambda x: x["date"])
        logger.debug("Prices sorted by date")

        # Extract closing prices
        closing_prices = [price["close_price"] for price in sorted_prices]
        dates = [price["date"] for price in sorted_prices]

        # Get first and last prices
        first_price = closing_prices[0]
        last_price = closing_prices[-1]
        first_date = datetime.strptime(dates[0], "%Y-%m-%d")
        last_date = datetime.strptime(dates[-1], "%Y-%m-%d")

        logger.debug(
            "Price range: %.2f (start) to %.2f (end)",
            first_price,
            last_price,
        )

        # Calculate total return
        total_return = ((last_price - first_price) / first_price) * 100
        logger.debug("Total return: %.2f%%", total_return)

        # Calculate time in years
        total_years = (last_date - first_date).days / 365.25
        logger.debug("Total years: %.2f", total_years)

        # Helper function to calculate CAGR
        def calculate_cagr(start_price, end_price, years):
            if years <= 0:
                return None
            return (((end_price / start_price) ** (1 / years)) - 1) * 100

        # Helper function to find price at a given date in the past
        def get_price_at_date(target_date):
            for i in range(len(sorted_prices) - 1, -1, -1):
                price_date = datetime.strptime(dates[i], "%Y-%m-%d")
                if price_date <= target_date:
                    return closing_prices[i]
            return None

        # Periods to calculate (in years)
        periods = [1, 3, 5, 7, 10, 15, 20]

        # Initialize ReturnsMetrics
        returns_metrics = ReturnsMetrics()

        # Calculate returns for each period
        period_mapping = {
            1: "one_year",
            3: "three_year",
            5: "five_year",
            7: "seven_year",
            10: "ten_year",
            15: "fifteen_year",
            20: "twenty_year",
        }

        for period in periods:
            target_date = last_date - timedelta(days=period * 365)
            period_price = get_price_at_date(target_date)

            if period_price is not None:
                if period == 1:
                    # 1-year return is simple return, not annualized
                    period_return = ((last_price - period_price) / period_price) * 100
                else:
                    # Annualized return for periods > 1 year
                    period_return = calculate_cagr(period_price, last_price, period)

                rounded_return = (
                    round(period_return, 2) if period_return is not None else "N/A"
                )
                setattr(returns_metrics, period_mapping[period], rounded_return)
                logger.debug(
                    "%d-year return: %.2f%%",
                    period,
                    rounded_return
                    if isinstance(rounded_return, (int, float))
                    else None,
                )
            else:
                setattr(returns_metrics, period_mapping[period], "N/A")
                logger.debug("%d-year: insufficient data", period)

        # Set total return
        returns_metrics.total_return = round(total_return, 2)

        logger.info("Returns calculated successfully")

        return CalculateReturnsResponse(
            success=True,
            returns=returns_metrics,
            total_data_points=len(prices_list),
            date_range={"start": dates[0], "end": dates[-1]},
            years_available=round(total_years, 2),
        )

    except Exception as e:
        logger.error("Error calculating returns: %s", str(e), exc_info=True)
        return CalculateReturnsResponse(
            success=False,
            error=f"Failed to calculate returns: {str(e)}",
        )
