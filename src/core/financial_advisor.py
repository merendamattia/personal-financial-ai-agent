"""
Financial Advisor Agent Module.

Agent specialized for portfolio analysis and generation using RAG.
"""

import json
import logging
from typing import Optional

from ..models import FinancialProfile, PACMetrics, Portfolio
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
        self.financial_profile_extraction_prompt = self._load_prompt_template(
            "financial_profile_extraction"
        )
        self.pac_metrics_extraction_prompt = self._load_prompt_template(
            "pac_metrics_extraction"
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
        Financial advisor has NO tools.

        Returns:
            Empty list - no tools for Financial advisor
        """
        logger.debug("Financial advisor initialized with no tools")
        return []

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
            extraction_prompt = self.financial_profile_extraction_prompt.format(
                conversation_summary=conversation_summary
            )

            logger.debug(
                "Calling structured_response with FinancialProfile model with prompt: %s",
                extraction_prompt,
            )

            try:
                response = self._client.structured_response(
                    input=extraction_prompt,
                    output_cls=FinancialProfile,
                    memory=self._memory,
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
            except ValueError as ve:
                if "extra_forbidden" in str(
                    ve
                ) or "Extra inputs are not permitted" in str(ve):
                    logger.error(
                        "LLM generated extra fields that could not be filtered: %s",
                        str(ve),
                    )
                    logger.warning("Returning default FinancialProfile")
                    return FinancialProfile()
                else:
                    raise

        except Exception as e:
            logger.error(
                "Failed to extract financial profile: %s", str(e), exc_info=True
            )
            raise RuntimeError(f"Failed to extract financial profile: {e}") from e

    # ==================== PAC Metrics Extraction ====================

    def extract_pac_metrics(self, financial_profile: dict) -> PACMetrics:
        """
        Extract PAC (Piano di Accumulo del Capitale) metrics from financial profile.

        Uses structured response to get:
        - Initial investment amount (from savings)
        - Monthly savings capacity (from income - expenses)

        Args:
            financial_profile: Dictionary with financial profile information

        Returns:
            PACMetrics object with initial_investment and monthly_savings

        Raises:
            RuntimeError: If extraction fails
        """
        logger.debug("Extracting PAC metrics from financial profile")

        try:
            extraction_prompt = self.pac_metrics_extraction_prompt.format(
                financial_profile=json.dumps(financial_profile, indent=2)
            )
            logger.debug(
                "Calling structured_response with PACMetrics model with prompt: %s",
                extraction_prompt,
            )

            response = self._client.structured_response(
                input=extraction_prompt,
                output_cls=PACMetrics,
                memory=self._memory,
            )

            logger.debug("Structured response received")

            if hasattr(response, "structured_data") and response.structured_data:
                metrics = response.structured_data[0]
                logger.info(
                    "PAC metrics extracted successfully: Initial €%d, Monthly €%.0f",
                    metrics.initial_investment,
                    metrics.monthly_savings,
                )
                return metrics
            else:
                logger.error("No structured data in response")
                logger.warning("Returning default PAC metrics")
                return PACMetrics()

        except Exception as e:
            logger.error("Failed to extract PAC metrics: %s", str(e), exc_info=True)
            logger.warning("Returning default PAC metrics due to extraction failure")
            return PACMetrics()

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

                query = self.rag_query_builder_prompt.format(
                    financial_profile=profile_json,
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
