"""
Main test module for financial tools.

Run this directly to test the unified analyze_financial_asset tool.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools import analyze_financial_asset


def main():
    """Run all tests."""
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test cases: (symbol, years)
    test_cases = [
        ("SWDA", 5),
        ("CSSPX", 3),
        ("ISAC", 4),
        ("SMEA", 2),
        ("CSEMAS", 5),
        ("CRPA", 3),
        ("DTLA", 5),
        ("IE3E", 3),
    ]

    for symbol, years in test_cases:
        print("\n" + "=" * 80)
        print(f"Analyzing {symbol} for {years} years")
        print("=" * 80)

        result = analyze_financial_asset(symbol, years=years)
        print(result)
        print()


if __name__ == "__main__":
    main()
