"""
DEPRECATED: This module has been moved to src/models/financial_profile.py

This file is kept for backwards compatibility. Please import from src.models instead:
    from src.models import FinancialProfile
"""

# Re-export from new location for backwards compatibility
from .models import FinancialProfile

__all__ = ["FinancialProfile"]
