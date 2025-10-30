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

from src.tools import calculate_returns, get_historical_prices


def main():
    """Run all tests."""
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    symbols = ["SWDA", "CSSPX", "ISAC", "SMEA", "CSEMAS", "CRPA", "DTLA", "IE3E"]

    for symbol in symbols:
        print("\n" + "=" * 60)
        print(f"TEST 1: Fetch ALL historical prices for {symbol}")
        print("=" * 60)
        historical_data = get_historical_prices(symbol)

        if historical_data.success:
            print(f"✅ Retrieved {historical_data.data_points} trading days")
            print(f"   Data available: {historical_data.years_available} years")
            print(
                f"   Date range: {historical_data.start_date} to {historical_data.end_date}"
            )

            print("\n" + "=" * 60)
            print("TEST 2: Calculate Returns (1, 3, 5, 7, 10, 15, 20 years)")
            print("=" * 60)
            # Convert prices to JSON string for the tool
            prices_json = json.dumps(historical_data.prices)
            returns = calculate_returns(prices_json)

            if returns.success:
                print(f"✅ Returns calculated successfully:")
                print(f"   Years available: {returns.years_available}")
                print(f"\n   📊 Return Metrics:")
                for period in [1, 3, 5, 7, 10, 15, 20]:
                    attr_name = {
                        1: "one_year",
                        3: "three_year",
                        5: "five_year",
                        7: "seven_year",
                        10: "ten_year",
                        15: "fifteen_year",
                        20: "twenty_year",
                    }[period]
                    value = getattr(returns.returns, attr_name)
                    if isinstance(value, str) and value == "N/A":
                        print(f"      {period:2d}-year: N/A (insufficient data)")
                    else:
                        print(f"      {period:2d}-year: {value:7.2f}%")
                print(f"      Total: {returns.returns.total_return:7.2f}%")
            else:
                print(f"❌ Return calculation failed: {returns.error}")
        else:
            print(f"❌ Historical data retrieval failed: {historical_data.error}")


if __name__ == "__main__":
    main()
