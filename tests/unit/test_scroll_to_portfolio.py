"""Test scroll to portfolio functionality."""

import re
from pathlib import Path

import pytest

# Get path to app.py relative to this test file
APP_PATH = Path(__file__).parent.parent.parent / "app.py"


def test_scroll_to_portfolio_delay_constant():
    """Test that SCROLL_TO_PORTFOLIO_DELAY_MS constant is defined in app.py."""
    with open(APP_PATH, "r") as f:
        content = f.read()

    # Verify the constant exists and has a reasonable value
    assert "SCROLL_TO_PORTFOLIO_DELAY_MS" in content

    # Extract the value (regex for the assignment)
    match = re.search(r"SCROLL_TO_PORTFOLIO_DELAY_MS\s*=\s*(\d+)", content)
    assert match is not None
    delay_value = int(match.group(1))
    assert delay_value > 0
    assert delay_value <= 1000  # Should be less than 1 second


def test_initialize_session_state_includes_scroll_flag():
    """Test that _initialize_session_state includes scroll_to_portfolio flag."""
    with open(APP_PATH, "r") as f:
        content = f.read()

    # Verify the function exists
    assert "def _initialize_session_state():" in content

    # Verify scroll_to_portfolio is in the session state defaults
    assert "scroll_to_portfolio" in content


def test_scroll_to_portfolio_session_state_key_in_defaults():
    """Test that scroll_to_portfolio is in the session state defaults."""
    # Read the app.py file and verify the key is in the defaults
    with open(APP_PATH, "r") as f:
        content = f.read()

    # Verify the key exists in the session state defaults (flexible whitespace matching)
    assert re.search(r'["\']scroll_to_portfolio["\']', content) is not None

    # Verify it's initialized to False (flexible whitespace matching)
    assert (
        re.search(
            r'["\']scroll_to_portfolio["\']\s*:\s*\([^)]*False', content, re.DOTALL
        )
        is not None
    )


def test_scroll_javascript_contains_required_elements():
    """Test that the JavaScript scroll code contains required elements."""
    with open(APP_PATH, "r") as f:
        content = f.read()

    # Verify the HTML anchor exists
    assert 'id="portfolio-section"' in content

    # Verify the JavaScript scrolling logic exists
    assert "scrollIntoView" in content
    assert "getElementById('portfolio-section')" in content

    # Verify the scroll flag is set when portfolio is generated
    assert "st.session_state.scroll_to_portfolio = True" in content

    # Verify the scroll flag is reset after scrolling
    assert "st.session_state.scroll_to_portfolio = False" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
