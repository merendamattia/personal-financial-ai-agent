"""
Main test module for financial tools.

Run this directly to test all three tools.
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import (
    calculate_returns,
    get_historical_prices,
    search_and_resolve_symbol,
)


def main():
    """Run all tests."""
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    symbols = ["SWDA", "CSSPX", "ISAC", "SMEA", "CSEMAS", "CRPA", "DTLA", "IE3E"]

    for symbol in symbols:
        # Test 1: Search for symbol
        print("\n" + "=" * 60)
        print(f"TEST 1: Search for Symbol ({symbol})")
        print("=" * 60)
        symbol_result = search_and_resolve_symbol(symbol)
        if symbol_result["success"]:
            print(f"✅ Symbol found: {symbol_result['found_symbol']}")
            print(f"   Company: {symbol_result['company_name']}")
            print(f"   Type: {symbol_result['symbol_type']}")
            print(f"   Currency: {symbol_result['currency']}")
            print(f"   Exchange: {symbol_result['exchange']}")
            symbol_to_use = symbol_result["found_symbol"]
        else:
            print(f"❌ Symbol search failed: {symbol_result['error']}")

        # Test 2: Get historical prices
        print("\n" + "=" * 60)
        print(f"TEST 2: Fetch ALL historical prices for {symbol_to_use}")
        print("=" * 60)
        historical_data = get_historical_prices(symbol_to_use)

        if historical_data["success"]:
            print(f"✅ Retrieved {historical_data['data_points']} trading days")
            print(f"   Data available: {historical_data['years_available']} years")
            print(
                f"   Date range: {historical_data['start_date']} to {historical_data['end_date']}"
            )

            # Test 3: Calculate returns
            print("\n" + "=" * 60)
            print("TEST 3: Calculate Returns (1, 3, 5, 7, 10, 15, 20 years)")
            print("=" * 60)
            # Convert prices to JSON string for the tool
            prices_json = json.dumps(historical_data["prices"])
            returns = calculate_returns(prices_json)

            if returns["success"]:
                print(f"✅ Returns calculated successfully:")
                print(f"   Years available: {returns['years_available']}")
                print(f"\n   📊 Return Metrics:")
                for period in [1, 3, 5, 7, 10, 15, 20]:
                    key = f"{period}_year"
                    value = returns["returns"].get(key, "N/A")
                    if isinstance(value, str) and value == "N/A":
                        print(f"      {period:2d}-year: {value} (insufficient data)")
                    else:
                        print(f"      {period:2d}-year: {value:7.2f}%")
                print(f"      Total : {returns['returns']['total_return']:7.2f}%")
            else:
                print(f"❌ Return calculation failed: {returns['error']}")
        else:
            print(f"❌ Historical data retrieval failed: {historical_data['error']}")


if __name__ == "__main__":
    from src.core import ChatBotAgent

    # agent = ChatBotAgent(provider="google")
    # print(agent.run("Qual è il ritorno storico del fondo SWDA negli ultimi 5 anni?", "required"))
    main()
