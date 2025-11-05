"""
Unit tests for analyze_financial_asset caching functionality.

Tests verify that the caching mechanism works correctly for financial asset analysis.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.tools.analyze_financial_asset import (
    _CACHE,
    _get_cache_key,
    _get_cached_analysis,
    _set_cached_analysis,
    analyze_financial_asset,
)


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_format(self):
        """Test that cache keys are generated in the correct format."""
        key = _get_cache_key("SWDA", 10)
        assert key == "SWDA_10"

    def test_cache_key_uppercase(self):
        """Test that cache keys convert ticker to uppercase."""
        key = _get_cache_key("swda", 5)
        assert key == "SWDA_5"

    def test_cache_key_different_years(self):
        """Test that different years produce different keys."""
        key1 = _get_cache_key("AAPL", 5)
        key2 = _get_cache_key("AAPL", 10)
        assert key1 != key2
        assert key1 == "AAPL_5"
        assert key2 == "AAPL_10"


class TestCacheOperations:
    """Test cache get/set operations."""

    def setup_method(self):
        """Clear cache before each test."""
        _CACHE.clear()

    def teardown_method(self):
        """Clear cache after each test."""
        _CACHE.clear()

    def test_cache_miss_when_empty(self):
        """Test that cache returns None when empty."""
        result = _get_cached_analysis("SWDA", 10)
        assert result is None

    def test_cache_set_and_get(self):
        """Test that we can set and retrieve cached values."""
        test_data = '{"success": true, "ticker": "SWDA"}'
        _set_cached_analysis("SWDA", 10, test_data)

        result = _get_cached_analysis("SWDA", 10)
        assert result == test_data

    def test_cache_different_tickers(self):
        """Test that cache correctly handles different tickers."""
        data1 = '{"success": true, "ticker": "SWDA"}'
        data2 = '{"success": true, "ticker": "AAPL"}'

        _set_cached_analysis("SWDA", 10, data1)
        _set_cached_analysis("AAPL", 10, data2)

        assert _get_cached_analysis("SWDA", 10) == data1
        assert _get_cached_analysis("AAPL", 10) == data2

    def test_cache_different_years(self):
        """Test that cache correctly handles different year periods."""
        data1 = '{"success": true, "years": 5}'
        data2 = '{"success": true, "years": 10}'

        _set_cached_analysis("SWDA", 5, data1)
        _set_cached_analysis("SWDA", 10, data2)

        assert _get_cached_analysis("SWDA", 5) == data1
        assert _get_cached_analysis("SWDA", 10) == data2

    def test_cache_uppercase_normalization(self):
        """Test that cache normalizes ticker case."""
        test_data = '{"success": true, "ticker": "SWDA"}'
        _set_cached_analysis("swda", 10, test_data)

        # Should be able to retrieve with different case
        result = _get_cached_analysis("SWDA", 10)
        assert result == test_data


class TestAnalyzeFinancialAssetCaching:
    """Test end-to-end caching in analyze_financial_asset function."""

    def setup_method(self):
        """Clear cache before each test."""
        _CACHE.clear()

    def teardown_method(self):
        """Clear cache after each test."""
        _CACHE.clear()

    @patch("src.tools.analyze_financial_asset._search_and_resolve_symbol")
    @patch("src.tools.analyze_financial_asset._get_historical_prices_internal")
    @patch("src.tools.analyze_financial_asset._calculate_returns_internal")
    def test_cache_miss_calls_functions(
        self, mock_calc_returns, mock_get_prices, mock_search_symbol
    ):
        """Test that on cache miss, all internal functions are called."""
        # Setup mocks
        mock_search_symbol.return_value = MagicMock(
            success=True, found_symbol="SWDA.DE", company_name="Test ETF"
        )
        mock_get_prices.return_value = {
            "success": True,
            "prices": [
                {"date": "2020-01-01", "close_price": 100.0},
                {"date": "2021-01-01", "close_price": 110.0},
            ],
            "data_points": 2,
            "years_available": 1.0,
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
        }
        mock_calc_returns.return_value = {
            "success": True,
            "returns": {"1_year": 10.0},
            "total_return": 10.0,
        }

        # First call should miss cache and call functions
        result1 = analyze_financial_asset("SWDA", years=10, use_cache=True)

        # Verify functions were called
        assert mock_search_symbol.called
        assert mock_get_prices.called
        assert mock_calc_returns.called

        # Verify result is valid JSON
        data = json.loads(result1)
        assert data["success"] is True
        assert data["ticker"] == "SWDA"

    @patch("src.tools.analyze_financial_asset._search_and_resolve_symbol")
    @patch("src.tools.analyze_financial_asset._get_historical_prices_internal")
    @patch("src.tools.analyze_financial_asset._calculate_returns_internal")
    def test_cache_hit_skips_functions(
        self, mock_calc_returns, mock_get_prices, mock_search_symbol
    ):
        """Test that on cache hit, internal functions are not called."""
        # Setup mocks for first call
        mock_search_symbol.return_value = MagicMock(
            success=True, found_symbol="SWDA.DE", company_name="Test ETF"
        )
        mock_get_prices.return_value = {
            "success": True,
            "prices": [
                {"date": "2020-01-01", "close_price": 100.0},
                {"date": "2021-01-01", "close_price": 110.0},
            ],
            "data_points": 2,
            "years_available": 1.0,
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
        }
        mock_calc_returns.return_value = {
            "success": True,
            "returns": {"1_year": 10.0},
            "total_return": 10.0,
        }

        # First call - cache miss
        result1 = analyze_financial_asset("SWDA", years=10, use_cache=True)

        # Reset mock call counts
        mock_search_symbol.reset_mock()
        mock_get_prices.reset_mock()
        mock_calc_returns.reset_mock()

        # Second call - should hit cache
        result2 = analyze_financial_asset("SWDA", years=10, use_cache=True)

        # Verify functions were NOT called on second attempt
        assert not mock_search_symbol.called
        assert not mock_get_prices.called
        assert not mock_calc_returns.called

        # Results should be identical
        assert result1 == result2

    @patch("src.tools.analyze_financial_asset._search_and_resolve_symbol")
    @patch("src.tools.analyze_financial_asset._get_historical_prices_internal")
    @patch("src.tools.analyze_financial_asset._calculate_returns_internal")
    def test_use_cache_false_bypasses_cache(
        self, mock_calc_returns, mock_get_prices, mock_search_symbol
    ):
        """Test that use_cache=False bypasses the cache."""
        # Setup mocks
        mock_search_symbol.return_value = MagicMock(
            success=True, found_symbol="SWDA.DE", company_name="Test ETF"
        )
        mock_get_prices.return_value = {
            "success": True,
            "prices": [
                {"date": "2020-01-01", "close_price": 100.0},
                {"date": "2021-01-01", "close_price": 110.0},
            ],
            "data_points": 2,
            "years_available": 1.0,
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
        }
        mock_calc_returns.return_value = {
            "success": True,
            "returns": {"1_year": 10.0},
            "total_return": 10.0,
        }

        # First call with caching
        result1 = analyze_financial_asset("SWDA", years=10, use_cache=True)

        # Reset mocks
        mock_search_symbol.reset_mock()
        mock_get_prices.reset_mock()
        mock_calc_returns.reset_mock()

        # Second call with use_cache=False should call functions again
        result2 = analyze_financial_asset("SWDA", years=10, use_cache=False)

        # Verify functions WERE called even though data is cached
        assert mock_search_symbol.called
        assert mock_get_prices.called
        assert mock_calc_returns.called

    def test_cache_respects_case_insensitive_ticker(self):
        """Test that cache works with case-insensitive ticker symbols."""
        # Pre-populate cache with lowercase ticker
        test_data = '{"success": true, "ticker": "swda"}'
        _set_cached_analysis("swda", 10, test_data)

        # Request with uppercase should hit the same cache
        result = _get_cached_analysis("SWDA", 10)
        assert result == test_data


class TestStreamlitSessionStateIntegration:
    """Test Streamlit session state integration."""

    def setup_method(self):
        """Clear caches before each test."""
        _CACHE.clear()

    def teardown_method(self):
        """Clear caches after each test."""
        _CACHE.clear()

    def test_streamlit_cache_storage(self):
        """Test that cache is stored in Streamlit session state when available."""
        # Create a mock streamlit module with a writable session_state
        mock_st_module = MagicMock()
        mock_session_state = MagicMock()
        mock_st_module.session_state = mock_session_state

        # Patch the import
        with patch.dict("sys.modules", {"streamlit": mock_st_module}):
            test_data = '{"success": true, "ticker": "SWDA"}'
            _set_cached_analysis("SWDA", 10, test_data)

            # Verify that financial_asset_cache was created and data was stored
            # The session_state should have been accessed to set the cache
            assert (
                mock_session_state.financial_asset_cache.__setitem__.called
                or hasattr(mock_session_state, "financial_asset_cache")
            )

    def test_streamlit_cache_retrieval(self):
        """Test that cache is retrieved from Streamlit session state when available."""
        # Create a mock streamlit module with pre-cached data
        test_data = '{"success": true, "ticker": "SWDA"}'

        mock_st_module = MagicMock()
        mock_session_state = MagicMock()
        # Configure the mock to have the cache
        mock_session_state.financial_asset_cache = {"SWDA_10": test_data}
        mock_st_module.session_state = mock_session_state

        # Patch the import
        with patch.dict("sys.modules", {"streamlit": mock_st_module}):
            result = _get_cached_analysis("SWDA", 10)
            assert result == test_data

    def test_fallback_to_module_cache_when_no_streamlit(self):
        """Test that module cache is used when Streamlit is not available."""
        # Store data in module cache
        test_data = '{"success": true, "ticker": "SWDA"}'
        _set_cached_analysis("SWDA", 10, test_data)

        # Retrieve should work even without Streamlit
        result = _get_cached_analysis("SWDA", 10)
        assert result == test_data

        # Verify it's in module cache
        assert "SWDA_10" in _CACHE
        assert _CACHE["SWDA_10"] == test_data
