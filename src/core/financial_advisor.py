"""
Financial Advisor Agent Module.

Agent specialized for portfolio analysis and generation using RAG.
"""

import json
import logging
from typing import Optional

from ..models import FinancialProfile, Portfolio
from ..retrieval import RAGAssetRetriever
from ..tools import analyze_financial_asset
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FinancialAdvisorAgent(BaseAgent):
    """
    Financial Advisor Agent for portfolio generation and analysis.

    Specializes in:
    - RAG-augmented financial advice
    - Portfolio generation based on financial profiles
    - Financial profile extraction from conversations
    """

    def __init__(
        self,
        name: Optional[str] = "FinancialAdvisorAgent",
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        planning_interval: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Initialize the FinancialAdvisorAgent.

        Args:
            name: Agent name
            system_prompt: System prompt for the agent
            provider: LLM provider ('ollama', 'google', 'openai')
            api_url: API URL (for Ollama)
            model: Model name to use
            planning_interval: Planning interval (default: 1)
            max_steps: Maximum steps (default: 1)
        """
        # Call parent init
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            provider=provider,
            api_url=api_url,
            model=model,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        # Load prompt templates specific to financial advisor
        self.rag_query_builder_prompt = self._load_prompt_template("rag_query_builder")
        self.portfolio_extraction_prompt = self._load_prompt_template(
            "portfolio_extraction"
        )

        # Initialize RAG retriever for asset data
        logger.debug("Initializing RAG asset retriever")
        self._rag_retriever = self._initialize_rag_retriever()

    def _get_default_system_prompt(self) -> str:
        """
        Get financial advisor system prompt.

        Returns:
            System prompt text
        """
        return self._load_system_prompt()

    def _get_default_tools(self) -> list:
        """
        Financial advisor has asset analysis tool.

        Returns:
            List with analyze_financial_asset tool
        """
        logger.debug("Setting up financial advisor tools")
        return [analyze_financial_asset]

    def _get_default_planning_interval(self) -> int:
        """
        Get default planning interval for FinancialAdvisorAgent.

        Returns:
            Planning interval: 1
        """
        return 1

    def _get_default_max_steps(self) -> int:
        """
        Get default max steps for FinancialAdvisorAgent.

        Returns:
            Max steps: 3
        """
        return 3

    def _initialize_rag_retriever(self) -> Optional[RAGAssetRetriever]:
        """
        Initialize the RAG retriever for asset data.

        Returns:
            RAGAssetRetriever instance or None if initialization fails
        """
        try:
            rag_retriever = RAGAssetRetriever()
            rag_retriever.build_or_load_index()
            logger.info("RAG asset retriever initialized successfully")
            return rag_retriever
        except Exception as e:
            logger.warning("Failed to initialize RAG retriever: %s", str(e))
            return None

    # ==================== Financial Profile ====================

    def extract_financial_profile(self, conversation_summary: str) -> FinancialProfile:
        """
        Extract structured financial profile from conversation summary.

        Uses datapizza's structured_response to extract financial information
        from the conversation summary into a structured FinancialProfile object.

        Args:
            conversation_summary: The conversation summary text to process

        Returns:
            FinancialProfile object with extracted financial information

        Raises:
            Exception: If the extraction fails
        """
        logger.debug("Extracting financial profile from summary")

        try:
            extraction_prompt = f"""Extract ONLY the following financial profile fields from the conversation summary.
For each field, extract the exact information mentioned. If not mentioned, use the default value provided.

EXTRACT THESE FIELDS ONLY (no extra fields):
- age_range: Age range (e.g., '25-34', '35-44', etc.) [default: '18-65']
- employment_status: Employment status (e.g., 'employed', 'self-employed', 'retired') [default: 'Employed']
- occupation: Current occupation or industry [default: 'None']
- annual_income_range: Income range (e.g., '30k-50k', '50k-100k', '100k+') [default: 'None']
- income_stability: Income stability (e.g., 'stable', 'moderate', 'unstable') [default: 'None']
- additional_income_sources: Additional income sources [default: 'None']
- monthly_expenses_range: Monthly expenses range (e.g., '2k-3k', '3k-5k') [default: 'None']
- major_expenses: Major recurring expenses (mortgage, rent, car payment, etc.) [default: 'None']
- total_debt: Total outstanding debt (e.g., 'minimal', '10k-50k', '50k-100k') [default: 'None']
- debt_types: Types of debt (mortgage, credit card, student loans, etc.) [default: 'None']
- savings_amount: Amount in savings (e.g., 'none', '1k-5k', '5k-20k', '20k+') [default: 'None']
- emergency_fund_months: Number of months of expenses in emergency fund [default: 'None']
- investments: Investment portfolio details (stocks, ETFs, crypto, etc.) [default: 'None']
- investment_experience: Investment experience (beginner, intermediate, advanced) [default: 'Beginner']
- primary_goals: Primary financial goals [default: 'Savings']
- short_term_goals: Short-term goals (next 1-2 years) [default: 'Savings']
- long_term_goals: Long-term goals (5+ years) [default: 'Savings']
- risk_tolerance: Risk tolerance (conservative, moderate, aggressive) [default: 'Conservative']
- risk_concerns: Specific financial concerns or risks [default: 'None']
- financial_knowledge_level: Financial knowledge (beginner, intermediate, advanced) [default: 'Beginner']
- geographic_allocation: Geographic investment preference [default: 'Global balanced']
- family_dependents: Number of dependents or family situation [default: 'None']
- insurance_coverage: Types of insurance coverage [default: 'None']
- summary_notes: Any additional important notes [default: 'None']

Conversation Summary:
{conversation_summary}

Extract ONLY these fields. Do not add any extra fields."""

            logger.debug("Calling structured_response with FinancialProfile model")

            response = self._client.structured_response(
                input=extraction_prompt,
                output_cls=FinancialProfile,
            )

            logger.debug("Structured response received")
            logger.debug("Response structured data: %s", response.structured_data)

            if hasattr(response, "structured_data") and response.structured_data:
                profile = response.structured_data[0]
                logger.info("Financial profile extracted successfully")
                logger.debug("Extracted profile data: %s", profile)
                return profile
            else:
                logger.error("No structured data in response")
                raise ValueError("No structured data returned from extraction")

        except Exception as e:
            logger.error(
                "Failed to extract financial profile: %s", str(e), exc_info=True
            )
            raise RuntimeError(f"Failed to extract financial profile: {e}") from e

    # ==================== Portfolio Generation ====================

    def generate_balanced_portfolio(self, financial_profile: dict) -> dict:
        """
        Generate a balanced investment portfolio based on the financial profile.

        Uses RAG to retrieve relevant asset information from ETF PDFs, then generates
        a customized portfolio allocation using the LLM with both the financial profile
        and asset data as context.

        Args:
            financial_profile: Dictionary with financial profile information

        Returns:
            dict: Portfolio recommendation as dictionary with structured data

        Raises:
            ValueError: If financial_profile is None or empty
            RuntimeError: If LLM fails or RAG retriever unavailable
        """
        if not financial_profile:
            logger.error("Financial profile cannot be None or empty")
            raise ValueError("Financial profile cannot be None or empty")

        logger.info("Generating balanced portfolio with RAG context")
        logger.debug("Profile keys: %s", list(financial_profile.keys()))

        try:
            profile_json = json.dumps(financial_profile, indent=2)

            asset_context = ""
            if self._rag_retriever:
                logger.debug("Retrieving asset information via RAG")

                risk_tolerance = financial_profile.get("risk_tolerance", "Conservative")
                investment_experience = financial_profile.get(
                    "investment_experience", "Beginner"
                )
                goals = financial_profile.get("primary_goals", "Savings")
                time_horizon = financial_profile.get("long_term_goals", "None")
                geographic_allocation = financial_profile.get(
                    "geographic_allocation", "Global balanced"
                )

                query = self.rag_query_builder_prompt.format(
                    risk_tolerance=risk_tolerance,
                    investment_experience=investment_experience,
                    time_horizon=time_horizon,
                    financial_goals=goals,
                    geographic_allocation=geographic_allocation,
                )

                logger.debug("RAG query: %s", query)

                try:
                    retrieved_assets = self._rag_retriever.retrieve(query, k=15)
                    logger.info("Retrieved %d asset documents", len(retrieved_assets))

                    asset_texts = []
                    for asset in retrieved_assets:
                        asset_texts.append(
                            f"[Asset: {asset['id']} (relevance: {asset['score']:.2f})]\n{asset['text'][:300]}..."
                        )
                    asset_context = "\n---\n".join(asset_texts)
                    logger.debug(
                        "Asset context prepared, length: %d", len(asset_context)
                    )
                except Exception as e:
                    logger.warning("RAG retrieval failed: %s", str(e))
            else:
                logger.warning("RAG retriever not available")

            extraction_prompt = self.portfolio_extraction_prompt.format(
                client_profile=profile_json, asset_context=asset_context
            )

            logger.debug("Sending structured portfolio generation request to LLM")
            logger.debug("Prompt length: %d characters", len(extraction_prompt))

            response = self._client.structured_response(
                input=extraction_prompt,
                output_cls=Portfolio,
            )

            logger.debug("Structured response received")

            if hasattr(response, "structured_data") and response.structured_data:
                portfolio = response.structured_data[0]
                logger.info("Portfolio generated successfully with RAG context")
                logger.info("Risk level: %s", portfolio.risk_level)

                portfolio_dict = portfolio.model_dump(mode="json")
                logger.debug(
                    "Returning portfolio dictionary with keys: %s",
                    list(portfolio_dict.keys()),
                )

                return portfolio_dict
            else:
                logger.error("No structured data in response")
                raise ValueError(
                    "No structured data returned from portfolio generation"
                )

        except Exception as e:
            logger.error("Failed to generate portfolio: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to generate portfolio: {e}") from e

    def analyze_asset(self, ticker: str, years: int = 10) -> dict:
        """
        Analyze a financial asset and extract structured return data.

        Directly calls the analyze_financial_asset tool and parses the JSON response.
        No need for agent orchestration since the tool does the work directly.

        Args:
            ticker: The stock/ETF ticker symbol (e.g., 'SWDA', 'IX5A')
            years: Number of years to analyze (default: 10)

        Returns:
            dict: Asset analysis data as dictionary with returns, dates, and metrics

        Raises:
            Exception: If the analysis fails
        """
        logger.debug("Analyzing asset %s for %d years", ticker, years)

        try:
            # Call the tool directly - it returns JSON string with all analysis
            logger.debug("Calling analyze_financial_asset tool for %s", ticker)
            result_json = analyze_financial_asset(ticker=ticker, years=years)

            logger.debug("Tool returned JSON, parsing...")

            # Parse the JSON string returned by the tool
            analysis_dict = json.loads(result_json)

            logger.info("Asset analysis completed for %s", ticker)
            logger.debug("Analysis keys: %s", list(analysis_dict.keys()))

            return analysis_dict

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from tool result for %s: %s", ticker, str(e)
            )
            raise RuntimeError(f"Invalid JSON from tool for {ticker}: {e}") from e
        except Exception as e:
            logger.error(
                "Failed to analyze asset %s: %s", ticker, str(e), exc_info=True
            )
            raise RuntimeError(f"Failed to analyze asset {ticker}: {e}") from e
