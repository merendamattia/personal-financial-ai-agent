"""
Charts and Visualization Module.

This module provides functions to generate financial charts and visualizations
using Streamlit components and the financial profile data extracted from
user conversations.
"""

import logging
import os
from typing import Optional

import pandas as pd
import streamlit as st

from .models import FinancialProfile

# Configure logger
logger = logging.getLogger(__name__)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _log_level, logging.INFO))


def parse_range_to_value(range_str: str) -> float:
    """
    Parse a range string to extract a numeric value.

    Args:
        range_str: String like "1000-2000€", "20000€+", "50k-100k", etc.

    Returns:
        Midpoint or average value from the range
    """
    logger.debug("Parsing range string: %s", range_str)

    try:
        # Remove common currency symbols and whitespace
        cleaned = (
            range_str.replace("€", "").replace("k", "000").replace("+", "").strip()
        )

        # Split by hyphen or comma
        if "-" in cleaned:
            parts = cleaned.split("-")
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            value = (low + high) / 2
            logger.debug("Extracted range: %f - %f, midpoint: %f", low, high, value)
            return value
        else:
            # Single value
            value = float(cleaned)
            logger.debug("Extracted single value: %f", value)
            return value
    except (ValueError, IndexError) as e:
        logger.warning("Could not parse range '%s': %s", range_str, str(e))
        return 0


def create_income_vs_expenses_chart(profile: FinancialProfile) -> None:
    """
    Create a pie chart showing the relationship between income and expenses using Streamlit.

    Args:
        profile: FinancialProfile object with financial data
    """
    logger.debug("Creating income vs expenses pie chart")

    try:
        income = parse_range_to_value(profile.annual_income_range)
        expenses = parse_range_to_value(profile.monthly_expenses_range) * 12

        if income <= 0 or expenses <= 0:
            logger.warning("Invalid income or expenses for pie chart")
            st.warning("Insufficient data for income vs expenses chart")
            return

        # Calculate savings
        savings = income - expenses

        # Create dataframe for Streamlit pie chart
        data = {
            "Category": ["Expenses", "Savings"],
            "Amount": [expenses, max(savings, 0)],
        }
        df = pd.DataFrame(data)

        st.subheader("💰 Annual Income Distribution")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(
                {
                    "data": [
                        {
                            "labels": df["Category"].tolist(),
                            "values": df["Amount"].tolist(),
                            "type": "pie",
                            "marker": {"colors": ["#ff9999", "#90EE90"]},
                        }
                    ],
                    "layout": {
                        "title": f"Annual Income: €{income:,.0f}",
                        "height": 400,
                    },
                },
                use_container_width=True,
            )

        with col2:
            st.metric("Annual Income", f"€{income:,.0f}")
            st.metric("Annual Expenses", f"€{expenses:,.0f}")
            st.metric("Annual Savings", f"€{max(savings, 0):,.0f}")

        logger.info("Income vs expenses pie chart created successfully")

    except Exception as e:
        logger.error("Failed to create income vs expenses chart: %s", str(e))
        st.error("Error creating income vs expenses chart")


def create_debt_composition_chart(profile: FinancialProfile) -> None:
    """
    Create a pie chart showing debt composition using Streamlit.

    Args:
        profile: FinancialProfile object with financial data
    """
    logger.debug("Creating debt composition pie chart")

    try:
        if not profile.total_debt or profile.total_debt.lower() in [
            "none",
            "minimal",
            "no",
            "nessuno",
        ]:
            logger.debug("No significant debt to display")
            return

        st.subheader("📊 Debt Composition")

        # Create simple debt breakdown
        if (
            "mortgage" in profile.total_debt.lower()
            or "mutuo" in profile.total_debt.lower()
        ):
            data = {
                "Type": ["Mortgage/Mutuo", "Other Debt"],
                "Percentage": [70, 30],
            }
        elif "credit" in (profile.debt_types or "").lower():
            data = {
                "Type": ["Credit Cards", "Other Debt"],
                "Percentage": [60, 40],
            }
        else:
            data = {"Type": ["Total Debt"], "Percentage": [100]}

        df = pd.DataFrame(data)

        st.plotly_chart(
            {
                "data": [
                    {
                        "labels": df["Type"].tolist(),
                        "values": df["Percentage"].tolist(),
                        "type": "pie",
                        "marker": {"colors": ["#ff6b6b", "#ffa500", "#ffcc99"]},
                    }
                ],
                "layout": {
                    "title": f"Debt Breakdown: {profile.total_debt}",
                    "height": 400,
                },
            },
            use_container_width=True,
        )

        logger.info("Debt composition pie chart created successfully")

    except Exception as e:
        logger.error("Failed to create debt composition chart: %s", str(e))
        st.error("Error creating debt composition chart")


def create_savings_breakdown_chart(profile: FinancialProfile) -> None:
    """
    Create a pie chart showing savings and emergency fund breakdown using Streamlit.

    Args:
        profile: FinancialProfile object with financial data
    """
    logger.debug("Creating savings breakdown pie chart")

    try:
        savings_amount = parse_range_to_value(profile.savings_amount)

        if savings_amount <= 0:
            logger.debug("No savings to display")
            return

        st.subheader("🏦 Savings Breakdown")

        # Calculate emergency fund and other savings
        emergency_months = (
            int("".join(filter(str.isdigit, profile.emergency_fund_months or "0"))) or 0
        )
        monthly_expenses = parse_range_to_value(profile.monthly_expenses_range)
        emergency_fund = emergency_months * monthly_expenses
        other_savings = max(savings_amount - emergency_fund, 0)

        data = {
            "Type": ["Emergency Fund", "Other Savings"],
            "Amount": [emergency_fund, other_savings],
        }
        df = pd.DataFrame(data)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(
                {
                    "data": [
                        {
                            "labels": df["Type"].tolist(),
                            "values": df["Amount"].tolist(),
                            "type": "pie",
                            "marker": {"colors": ["#87CEEB", "#FFD700"]},
                        }
                    ],
                    "layout": {
                        "title": f"Total Savings: €{savings_amount:,.0f}",
                        "height": 400,
                    },
                },
                use_container_width=True,
            )

        with col2:
            st.metric("Total Savings", f"€{savings_amount:,.0f}")
            st.metric("Emergency Fund", f"€{emergency_fund:,.0f}")
            st.metric("Other Savings", f"€{other_savings:,.0f}")
            st.metric("Emergency Coverage", f"{emergency_months} months")

        logger.info("Savings breakdown pie chart created successfully")

    except Exception as e:
        logger.error("Failed to create savings breakdown chart: %s", str(e))
        st.error("Error creating savings breakdown chart")


def create_investment_allocation_chart(profile: FinancialProfile) -> None:
    """
    Create a pie chart showing investment allocation using Streamlit.

    Args:
        profile: FinancialProfile object with financial data
    """
    logger.debug("Creating investment allocation pie chart")

    try:
        if not profile.investments or profile.investments.lower() in [
            "none",
            "no",
            "nessuno",
        ]:
            logger.debug("No investments to display")
            return

        st.subheader("📈 Investment Allocation")

        # Parse investment types (simple heuristic)
        investments_lower = profile.investments.lower()

        # Create investment allocation
        investment_types = []
        if "etf" in investments_lower or "fondo" in investments_lower:
            investment_types.append(("ETF/Fondi", 50))
        if "azioni" in investments_lower or "stock" in investments_lower:
            investment_types.append(("Azioni", 30))
        if "immobili" in investments_lower or "real estate" in investments_lower:
            investment_types.append(("Immobili", 40))
        if "cripto" in investments_lower or "crypto" in investments_lower:
            investment_types.append(("Criptovalute", 10))

        if not investment_types:
            investment_types = [("Investimenti Vari", 100)]

        # Normalize to 100%
        total = sum(v for _, v in investment_types)
        normalized = [(k, (v / total) * 100) for k, v in investment_types]

        labels = [inv[0] for inv in normalized]
        values = [inv[1] for inv in normalized]

        st.plotly_chart(
            {
                "data": [
                    {
                        "labels": labels,
                        "values": values,
                        "type": "pie",
                        "marker": {
                            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
                        },
                    }
                ],
                "layout": {
                    "title": f"Investment Experience: {profile.investment_experience}",
                    "height": 400,
                },
            },
            use_container_width=True,
        )

        logger.info("Investment allocation pie chart created successfully")

    except Exception as e:
        logger.error("Failed to create investment allocation chart: %s", str(e))
        st.error("Error creating investment allocation chart")


def display_financial_charts(profile: FinancialProfile) -> None:
    """
    Display all financial charts in Streamlit.

    Args:
        profile: FinancialProfile object with financial data
    """
    logger.debug("Displaying financial charts")

    st.divider()
    st.header("Financial Analysis Charts")

    # Income vs Expenses
    create_income_vs_expenses_chart(profile)

    st.divider()

    # Savings Breakdown
    create_savings_breakdown_chart(profile)

    st.divider()

    # Debt Composition
    create_debt_composition_chart(profile)

    st.divider()

    # Investment Allocation
    create_investment_allocation_chart(profile)

    logger.info("All financial charts displayed successfully")
