"""Test RAG and Portfolio Generation with Google Provider."""

import json
import sys

from src.core import ChatbotAgent


def create_test_profile() -> dict:
    """Create a realistic financial profile for testing."""
    profile = {
        # Personal Information
        "age_range": "35-44",
        "employment_status": "employed",
        "occupation": "Software Engineer",
        # Income Information
        "annual_income_range": "100k-150k",
        "income_stability": "stable",
        "additional_income_sources": "Freelance consulting projects",
        # Expenses and Debts
        "monthly_expenses_range": "5k-7k",
        "major_expenses": "Mortgage (3k), Car payment (500), Insurance (300)",
        "total_debt": "250k-300k",
        "debt_types": "Mortgage, car loan",
        # Assets and Savings
        "savings_amount": "50k-100k",
        "emergency_fund_months": "6",
        "investments": "Current: 150k in 401k, 50k in personal brokerage account",
        "investment_experience": "intermediate",
        # Financial Goals
        "primary_goals": "Retirement planning, wealth building, home equity growth",
        "short_term_goals": "Build investment portfolio to 250k in next 2 years",
        "long_term_goals": "Retire by 60, have 2M in net worth",
        # Risk Profile
        "risk_tolerance": "moderate",
        "risk_concerns": "Market volatility, inflation impact on retirement",
        # Financial Knowledge
        "financial_knowledge_level": "intermediate",
        # Investment Geography
        "geographic_allocation": "Global balanced (70% USA, 30% International)",
    }
    return profile


def test_portfolio_generation():
    """Test portfolio generation with Google provider."""
    print("\n" + "=" * 80)
    print("TEST: PORTFOLIO GENERATION WITH GOOGLE")
    print("=" * 80)

    try:
        # Create ChatbotAgent with Google provider
        print("Initializing ChatbotAgent with Google provider...")
        agent = ChatbotAgent(
            name="FinancialAdvisor",
            provider="google",
        )
        print("ChatbotAgent initialized with Google provider")

        # Check RAG retriever
        if agent._rag_retriever:
            print("RAG retriever is available")
        else:
            print("RAG retriever not initialized")
            return False

        # Create test profile
        profile = create_test_profile()
        print(f"\nTest profile created with {len(profile)} fields")
        print("Profile summary:")
        print(f"  - Age: {profile['age_range']}")
        print(f"  - Income: {profile['annual_income_range']}")
        print(f"  - Risk Tolerance: {profile['risk_tolerance']}")
        print(f"  - Investment Experience: {profile['investment_experience']}")

        # Generate portfolio
        print("\nGenerating portfolio with RAG context...")
        portfolio = agent.generate_balanced_portfolio(profile)

        print("Portfolio generated successfully!")
        print("\nPortfolio Output:")
        print(json.dumps(portfolio, indent=2))

        # Validate structure
        if "portfolio_allocation" in portfolio:
            print(
                f"\nPortfolio allocation: {list(portfolio['portfolio_allocation'].keys())}"
            )
        if "risk_level" in portfolio:
            print(f"Risk level: {portfolio['risk_level']}")
        if "reasoning" in portfolio:
            print(f"Reasoning provided (length: {len(portfolio['reasoning'])} chars)")

        return True

    except Exception as e:
        print(f"Portfolio generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING RAG ASSET RETRIEVER & PORTFOLIO GENERATION")
    print("=" * 80)

    results = []

    results.append(("Portfolio Generation", test_portfolio_generation()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
