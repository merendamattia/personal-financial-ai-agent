"""
Financial AI Agent Streamlit Application.

This module provides a web interface for interacting with the financial
AI agent powered by datapizza-ai and multiple LLM providers.
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from src.clients import list_providers
from src.core import ChatbotAgent, FinancialAdvisorAgent
from src.models import FinancialProfile, Portfolio

MONTECARLO_MIN_ASSET_VOLATILITY = float(
    os.getenv("MONTECARLO_MIN_ASSET_VOLATILITY", 0.1)
)
MONTECARLO_SIMULATION_SCENARIOS = int(
    os.getenv("MONTECARLO_SIMULATION_SCENARIOS", 1000)
)
MONTECARLO_SIMULATION_YEARS = int(os.getenv("MONTECARLO_SIMULATION_YEARS", 20))
MONTECARLO_DEFAULT_INITIAL_INVESTMENT = int(
    os.getenv("MONTECARLO_DEFAULT_INITIAL_INVESTMENT", 1000)
)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.debug("Loading environment variables")
load_dotenv()
logger.info("Environment variables loaded")

# Page configuration
st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_ollama_available() -> bool:
    """
    Check if Ollama is available (localhost or docker container).

    Returns:
        True if Ollama is available, False otherwise
    """
    logger.debug("Checking Ollama availability")

    # Try localhost first (for local Ollama)
    try:
        logger.debug("Trying localhost:11434")
        response = requests.get("http://localhost:11434/", timeout=2)
        is_available = "Ollama is running" in response.text
        logger.info("Ollama available at localhost:11434")
        return is_available
    except Exception as e:
        logger.debug("localhost:11434 failed: %s", str(e))

    # Try docker container address (for docker-compose)
    try:
        logger.debug("Trying ollama:11434 (docker)")
        response = requests.get("http://ollama:11434/", timeout=2)
        is_available = "Ollama is running" in response.text
        logger.info("Ollama available at ollama:11434 (docker)")
        return is_available
    except Exception as e:
        logger.debug("ollama:11434 failed: %s", str(e))

    logger.warning("Ollama not available on both addresses")
    return False


def get_available_providers() -> list:
    """
    Get list of available providers.
    Ollama is only available if it's running.

    Returns:
        List of available provider names
    """
    logger.debug("Getting available providers")

    all_providers = list_providers()
    logger.debug("All registered providers: %s", all_providers)

    available = []

    for provider in all_providers:
        if provider == "ollama":
            if is_ollama_available():
                available.append(provider)
                logger.debug("Ollama is available")
            else:
                logger.debug("Ollama is not available")
        else:
            # Google and OpenAI are always available (if API keys are set, they'll work)
            available.append(provider)
            logger.debug("Provider %s is available", provider)

    logger.info("Available providers: %s", available)
    return available


def load_profile_from_json(uploaded_file) -> FinancialProfile:
    """
    Load a financial profile from an uploaded JSON file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        FinancialProfile object if valid, None otherwise
    """
    try:
        import json

        logger.debug("Loading financial profile from JSON file: %s", uploaded_file.name)

        # Read the file content
        file_content = uploaded_file.read().decode("utf-8")
        data = json.loads(file_content)

        # Validate and create FinancialProfile
        profile = FinancialProfile(**data)

        logger.info("Financial profile loaded successfully from JSON")
        return profile
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON format: %s", str(e))
        st.error(f"Invalid JSON format: {str(e)}")
        return None
    except Exception as e:
        logger.error("Failed to load profile from JSON: %s", str(e))
        st.error(f"Failed to load profile: {str(e)}")
        return None


@st.cache_resource
def initialize_chatbot(provider: str) -> ChatbotAgent:
    """
    Initialize the chatbot agent (pure conversation, no tools).

    Args:
        provider: The LLM provider to use

    Returns:
        Initialized ChatbotAgent instance
    """
    logger.debug("Initializing chatbot agent with provider: %s", provider)

    try:
        # Use a more meaningful agent name
        agent = ChatbotAgent(name="ChatbotAgent", provider=provider)
        logger.info(
            "Chatbot agent initialized successfully with provider: %s", provider
        )
        return agent
    except Exception as e:
        logger.error(
            "Failed to initialize chatbot agent with provider %s: %s",
            provider,
            str(e),
            exc_info=True,
        )
        raise


@st.cache_resource
def initialize_financial_advisor(
    provider: str, name: Optional[str] = None
) -> FinancialAdvisorAgent:
    """
    Initialize the financial advisor agent (with RAG and portfolio generation).

    Args:
        provider: The LLM provider to use

    Returns:
        Initialized FinancialAdvisorAgent instance
    """
    logger.debug("Initializing financial advisor agent with provider: %s", provider)

    try:
        agent = FinancialAdvisorAgent(name="FinancialAdvisorAgent", provider=provider)
        logger.info(
            "Financial advisor agent '%s' initialized successfully with provider: %s",
            agent.name,
            provider,
        )
        return agent
    except Exception as e:
        logger.error(
            "Failed to initialize financial advisor agent with provider %s: %s",
            provider,
            str(e),
            exc_info=True,
        )
        raise


def stream_text(text: str, chunk_size: int = 20):
    """
    Stream text with a typing effect by displaying chunks progressively.

    Args:
        text: The text to stream
        chunk_size: Number of characters per chunk (default: 20)
    """
    placeholder = st.empty()
    displayed_text = ""

    for i in range(0, len(text), chunk_size):
        displayed_text += text[i : i + chunk_size]
        placeholder.markdown(displayed_text)
        time.sleep(0.1)  # 100ms delay between chunks for smooth typing effect


def _fetch_and_display_historical_returns(portfolio, financial_advisor_agent):
    """
    Fetch and display historical returns data for portfolio assets.

    Args:
        portfolio: The portfolio dictionary containing assets
        financial_advisor_agent: The FinancialAdvisorAgent instance

    Returns:
        None (displays data using st components)
    """
    logger.debug("Fetching historical returns for portfolio assets")

    # Collect returns data
    returns_data = []

    if "assets" in portfolio and isinstance(portfolio["assets"], list):
        with st.spinner("🔄 Retrieving 10-year historical data for assets..."):
            for asset in portfolio["assets"]:
                asset_symbol = (
                    asset.get("symbol") if isinstance(asset, dict) else asset.symbol
                )

                if asset_symbol:
                    try:
                        logger.debug("Fetching returns for asset: %s", asset_symbol)
                        if asset_symbol.upper() == "BITCOIN":
                            asset_symbol = "BTC-EUR"

                        # Call advisor.analyze_asset to get structured asset data
                        result_data = financial_advisor_agent.analyze_asset(
                            asset_symbol, years=10
                        )

                        logger.debug(
                            "Received data for %s: %s",
                            asset_symbol,
                            result_data,
                        )

                        if result_data.get("success"):
                            logger.debug(
                                "Successfully retrieved data for %s",
                                asset_symbol,
                            )
                            logger.debug("Result data: %s", result_data)

                            # Extract company name for display
                            company_name = result_data.get("company_name", asset_symbol)

                            # Extract returns
                            returns = result_data.get("returns", [])

                            # Helper function to get return for specific year
                            def get_return_for_year(year_val):
                                for ret in returns:
                                    if ret.get("year") == year_val:
                                        val = ret.get("percentage")
                                        return (
                                            f"{val:.2f}%"
                                            if isinstance(val, (int, float))
                                            else "N/A"
                                        )
                                return "N/A"

                            returns_data.append(
                                {
                                    "Asset": f"{company_name} ({asset_symbol})",
                                    "1-Year Return %": get_return_for_year(1),
                                    "3-Year Return %": get_return_for_year(3),
                                    "5-Year Return %": get_return_for_year(5),
                                    "10-Year Return %": get_return_for_year(10),
                                }
                            )
                        else:
                            logger.warning(
                                "Failed to retrieve data for %s: %s",
                                asset_symbol,
                                result_data.get("error"),
                            )
                            returns_data.append(
                                {
                                    "Asset": f"{asset_symbol} (Error)",
                                    "1-Year Return %": "N/A",
                                    "3-Year Return %": "N/A",
                                    "5-Year Return %": "N/A",
                                    "10-Year Return %": "N/A",
                                }
                            )

                    except Exception as e:
                        logger.error(
                            "Error fetching returns for %s: %s",
                            asset_symbol,
                            str(e),
                        )
                        returns_data.append(
                            {
                                "Asset": f"{asset_symbol} (Error)",
                                "1-Year Return %": "N/A",
                                "3-Year Return %": "N/A",
                                "5-Year Return %": "N/A",
                                "10-Year Return %": "N/A",
                            }
                        )

    # Display returns table
    if returns_data:
        df_returns = pd.DataFrame(returns_data)
        st.dataframe(df_returns, width="stretch", hide_index=True)
        logger.info("Returns table displayed successfully")
    else:
        st.info("Unable to retrieve historical data for assets in the portfolio.")


def _display_key_considerations(portfolio):
    """
    Display key considerations from the portfolio.

    Args:
        portfolio: The portfolio dictionary containing key_considerations

    Returns:
        None (displays data using st components)
    """
    if "key_considerations" in portfolio and portfolio["key_considerations"]:
        st.markdown("### 📋 Key Considerations")
        # Handle list of considerations
        if isinstance(portfolio["key_considerations"], list):
            for consideration in portfolio["key_considerations"]:
                if consideration:
                    st.write(f"• {consideration}")
        else:
            # Fallback for string format (old structure)
            considerations = portfolio["key_considerations"].split(";")
            for consideration in considerations:
                consideration = consideration.strip()
                if consideration:
                    st.write(f"• {consideration}")


def _generate_portfolio_for_profile(advisor_agent, profile):
    """
    Generate a balanced portfolio for the given financial profile.

    Args:
        advisor_agent: The FinancialAdvisorAgent instance
        profile: The FinancialProfile object

    Returns:
        The generated Portfolio dict, or None if generation failed
    """
    logger.info("Starting portfolio generation for profile")

    try:
        # Convert FinancialProfile object to dict for agent
        profile_dict = profile.model_dump()

        logger.debug("Converting profile to dict for portfolio generation")

        # Generate portfolio using RAG-enhanced advisor agent with structured response
        portfolio = advisor_agent.generate_balanced_portfolio(profile_dict)

        logger.info("Portfolio generated successfully")
        if portfolio:
            assets_count = (
                len(portfolio.get("assets", [])) if isinstance(portfolio, dict) else 0
            )
            risk = portfolio.get("risk_level") if isinstance(portfolio, dict) else None
            logger.debug(
                "Portfolio structure: assets_count=%d, risk_level=%s",
                assets_count,
                risk,
            )

        return portfolio

    except Exception as portfolio_error:
        logger.error(
            "Failed to generate portfolio: %s",
            str(portfolio_error),
            exc_info=True,
        )
        st.error(f"Failed to generate portfolio: {str(portfolio_error)}")
        return None


def _clear_loaded_profile():
    """
    Clear the loaded financial profile and related state.

    This is called when the user deletes an uploaded JSON file or
    wants to reset the profile without clearing the entire conversation.

    Returns:
        None (modifies st.session_state)
    """
    logger.info("Clearing loaded profile and related state")
    st.session_state.financial_profile = None
    st.session_state.conversation_completed = False
    st.session_state.generated_portfolio = None
    st.session_state.profile_loaded_from_json = False


def _initialize_session_state():
    """
    Initialize all required session state variables.

    Returns:
        None (initializes st.session_state)
    """
    session_state_defaults = {
        "provider": (None, "Initialized provider session state to None"),
        "messages": ([], "Initialized messages session state to empty list"),
        "question_index": (0, "Initialized question_index session state to 0"),
        "agent_initialized": (
            False,
            "Initialized agent_initialized session state to False",
        ),
        "conversation_completed": (
            False,
            "Initialized conversation_completed session state to False",
        ),
        "financial_profile": (
            None,
            "Initialized financial_profile session state to None",
        ),
        "profile_loaded_from_json": (
            False,
            "Initialized profile_loaded_from_json session state to False",
        ),
        "health_check_done": (
            False,
            "Initialized health_check_done session state to False",
        ),
        "agent_is_healthy": (
            False,
            "Initialized agent_is_healthy session state to False",
        ),
        "generated_portfolio": (
            None,
            "Initialized generated_portfolio session state to None",
        ),
        "pac_params": (
            None,
            "Initialized pac_params session state to None",
        ),
        "cached_returns_data": (
            None,
            "Initialized cached_returns_data session state to None",
        ),
    }

    for key, (default_value, debug_message) in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            logger.debug(debug_message)


def _show_provider_selection():
    """
    Display provider selection modal on first load.

    Returns:
        None (displays UI and returns if provider not selected)
    """
    logger.debug("No provider selected, showing provider selection modal")

    _show_header()
    st.subheader("🔧 Select Your LLM Provider")
    st.markdown(
        "Choose which AI service to use for this conversation. You can change this anytime."
    )

    col1, col2, col3 = st.columns(3)
    available_providers = get_available_providers()

    with col1:
        if "ollama" in available_providers:
            if st.button("🦙 Ollama", use_container_width=True, key="ollama_btn"):
                logger.info("Ollama provider selected")
                st.session_state.provider = "ollama"
                st.session_state.agent_initialized = False
                st.rerun()
        else:
            st.button(
                "Ollama (Not Running)",
                disabled=True,
                use_container_width=True,
                help="Start Ollama with: ollama serve",
            )

    with col2:
        if "google" in available_providers:
            if st.button(
                "🌐 Google Gemini", use_container_width=True, key="google_btn"
            ):
                logger.info("Google provider selected")
                st.session_state.provider = "google"
                st.session_state.agent_initialized = False
                st.rerun()
        else:
            st.button(
                "Google (Not Available)",
                disabled=True,
                use_container_width=True,
            )

    with col3:
        if "openai" in available_providers:
            if st.button("✨ OpenAI", use_container_width=True, key="openai_btn"):
                logger.info("OpenAI provider selected")
                st.session_state.provider = "openai"
                st.session_state.agent_initialized = False
                st.rerun()
        else:
            st.button(
                "OpenAI (Not Available)",
                disabled=True,
                use_container_width=True,
            )

    st.divider()
    st.info(
        "💡 **Tip:** Ollama is free and runs locally. "
        "Install from [ollama.com](https://ollama.com/)"
    )


def _show_header():
    """
    Display the application header.
    """
    st.title("💰 Financial AI Agent")
    st.markdown(
        "An intelligent personal financial advisor powered by AI. Get expert financial guidance in your preferred language with support for multiple LLM providers, including local offline inference with Ollama."
    )

    st.markdown(
        """
        [![GitHub](https://img.shields.io/badge/GitHub-merendamattia%2Fpersonal--financial--ai--agent-black?logo=github)](https://github.com/merendamattia/personal-financial-ai-agent)
        [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
        [![Latest Release](https://img.shields.io/github/v/release/merendamattia/personal-financial-ai-agent?label=release)](https://github.com/merendamattia/personal-financial-ai-agent/releases)
        """
    )
    st.divider()


def _show_loading_screen():
    """
    Display loading screen while initializing agent.

    Returns:
        None (displays UI with loading animation)
    """
    logger.debug("Agent not yet initialized, showing loading screen")

    _show_header()

    # Loading animation
    loading_col1, loading_col2, loading_col3 = st.columns([1, 2, 1])
    with loading_col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; animation: spin 2s linear infinite;">⏳</div>
                <h3>Initializing your financial advisor...</h3>
                <p style="color: gray;">Please wait while we load your AI assistant.</p>
            </div>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


def _initialize_agents():
    """
    Initialize chatbot and financial advisor agents.

    Returns:
        Tuple of (chatbot_agent, financial_advisor_agent) or reruns on error
    """
    try:
        logger.debug(
            "Initializing chatbot and financial advisor for provider: %s",
            st.session_state.provider,
        )
        chatbot_agent = initialize_chatbot(st.session_state.provider)
        financial_advisor_agent = initialize_financial_advisor(
            st.session_state.provider
        )
        st.session_state["financial_advisor_agent"] = financial_advisor_agent
        st.session_state["chatbot_agent"] = chatbot_agent
        st.session_state.agent_initialized = True
        logger.info("Agents initialized successfully")
        return chatbot_agent, financial_advisor_agent
    except Exception as e:
        logger.error("Failed to initialize agents: %s", str(e), exc_info=True)
        st.error(f"Failed to initialize agents: {e}")
        st.session_state.provider = None
        st.session_state.agent_initialized = False
        st.rerun()


def _setup_sidebar(chatbot_agent, financial_advisor_agent):
    """
    Setup and display the sidebar with settings and controls.

    Args:
        chatbot_agent: The ChatbotAgent instance
        financial_advisor_agent: The FinancialAdvisorAgent instance

    Returns:
        None (displays sidebar UI)
    """
    with st.sidebar:
        st.header("⚙️ Settings")

        # Provider info - health check only once
        if not st.session_state.health_check_done:
            logger.debug("Performing health check for the first time")
            st.session_state.agent_is_healthy = (
                chatbot_agent.is_healthy() and financial_advisor_agent.is_healthy()
            )
            st.session_state.health_check_done = True
            logger.debug(
                "Health check completed: %s", st.session_state.agent_is_healthy
            )

        if st.session_state.agent_is_healthy:
            logger.debug("Agent health check passed")
            st.success("Agents initialized!")
        else:
            logger.warning("Agent health check failed")
            st.warning("⚠️ Agents running but connection may be unstable")

        # Change provider
        if st.button("🔄 Change Provider", use_container_width=True):
            logger.info("Changing provider requested")
            st.session_state.provider = None
            st.session_state.messages = []
            st.session_state.question_index = 0
            st.session_state.conversation_completed = False
            st.session_state.financial_profile = None
            st.session_state.generated_portfolio = None
            st.session_state.health_check_done = False
            st.session_state.agent_is_healthy = False
            st.session_state.profile_loaded_from_json = False
            st.rerun()

        # Clear history button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            logger.debug("Clearing conversation history")
            chatbot_agent.clear_memory()
            chatbot_agent.reset_questions()
            st.session_state.messages = []
            st.session_state.question_index = 0
            st.session_state.conversation_completed = False
            st.session_state.financial_profile = None
            st.session_state.generated_portfolio = None
            st.session_state.health_check_done = False
            st.session_state.agent_is_healthy = False
            st.session_state.profile_loaded_from_json = False
            st.success("Conversation cleared!")

        # Upload custom profile from JSON
        st.divider()
        st.subheader("📤 Load Custom Profile")
        uploaded_json = st.file_uploader(
            "Upload a financial profile JSON file",
            type=["json"],
            help="Upload a JSON file with your financial profile data",
        )

        # Detect file deletion (X button pressed)
        # When uploaded_json is None but profile_loaded_from_json is True,
        # it means the user removed the file via the X button
        if uploaded_json is None and st.session_state.profile_loaded_from_json:
            logger.info("JSON file removed by user, clearing profile")
            _clear_loaded_profile()
            st.info("Profile cleared. Upload a new file to load a profile.")
            st.rerun()

        # Load profile when file is first uploaded (but don't auto-analyze)
        # Both conditions are required:
        # 1. uploaded_json is not None: User has selected a file
        # 2. not profile_loaded_from_json: This is a new upload, not a re-render
        if uploaded_json is not None and not st.session_state.profile_loaded_from_json:
            logger.debug("JSON file uploaded: %s", uploaded_json.name)
            profile = load_profile_from_json(uploaded_json)

            if profile:
                st.session_state.financial_profile = profile
                st.session_state.profile_loaded_from_json = True
                # Don't set conversation_completed yet - wait for user to click analyze button
                logger.info(
                    "Profile loaded successfully, waiting for user to trigger analysis"
                )
                st.success(
                    "Profile loaded successfully! Click 'Analyze Profile' below to start the analysis."
                )

        # Add explicit button to trigger analysis for loaded profile
        if (
            st.session_state.profile_loaded_from_json
            and not st.session_state.conversation_completed
        ):
            if st.button(
                "🔍 Analyze Profile", use_container_width=True, type="primary"
            ):
                logger.info("User triggered profile analysis")
                st.session_state.conversation_completed = True
                st.session_state.generated_portfolio = None
                st.rerun()

        # Model info
        st.divider()
        st.subheader("📊 Model Information")
        config = chatbot_agent.get_config_summary()
        st.write(f"**Provider:** {config['provider'].upper()}")
        st.write(f"**Model:** {config['model']}")
        st.write(f"**Agent:** {config['name']}")
        logger.debug("Config summary displayed: %s", config)

        # Progress info
        st.divider()
        st.subheader("📋 Assessment Progress")
        progress = chatbot_agent.get_questions_progress()
        st.write(
            f"**Question {progress['current_index'] + 1} of {progress['total_questions']}**"
        )
        st.progress(progress["progress_percentage"] / 100)
        logger.debug("Progress displayed: %s", progress)


def _display_financial_profile_summary():
    """
    Display the financial profile summary information.

    Returns:
        None (displays UI components)
    """
    if st.session_state.financial_profile:
        logger.debug("Displaying extracted financial profile")
        st.subheader("Financial Profile Summary")

        # Convert profile to dictionary for display
        profile_dict = st.session_state.financial_profile.model_dump()

        # Create DataFrame for table (excluding summary_notes)
        table_data = {}
        for key, value in profile_dict.items():
            if key == "summary_notes":  # Skip summary_notes for table
                continue
            # Format key for display (snake_case to Title Case)
            display_key = key.replace("_", " ").title()
            # Format value
            display_value = value if value is not None else "N/A"
            table_data[display_key] = [str(display_value)]

        # Convert to DataFrame and display as table
        df = pd.DataFrame(table_data, index=["Data"]).T
        st.table(df)

        # Display summary notes separately below table
        if profile_dict.get("summary_notes"):
            st.markdown("**📝 Summary Notes**")
            st.write(profile_dict["summary_notes"])

        # Display JSON download option
        st.download_button(
            label="📥 Download Financial Profile (JSON)",
            data=st.session_state.financial_profile.model_dump_json(indent=2),
            file_name="financial_profile.json",
            mime="application/json",
            key="download_profile",
        )


def _display_portfolio_pie_chart(portfolio):
    """
    Display an interactive pie chart showing portfolio asset allocation.

    Args:
        portfolio: The portfolio dictionary containing assets

    Returns:
        None (displays Plotly pie chart)
    """
    logger.debug("Displaying portfolio allocation pie chart")

    if "assets" in portfolio and isinstance(portfolio["assets"], list):
        # Prepare data for pie chart
        asset_symbols = []
        asset_percentages = []

        for asset in portfolio["assets"]:
            symbol = asset.get("symbol") if isinstance(asset, dict) else asset.symbol
            percentage = (
                asset.get("percentage") if isinstance(asset, dict) else asset.percentage
            )
            if symbol and percentage:
                asset_symbols.append(symbol)
                asset_percentages.append(percentage)

        if asset_symbols:
            # Create pie chart
            fig = px.pie(
                values=asset_percentages,
                names=asset_symbols,
                hole=0.3,  # Create donut chart
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Allocation: %{value}%<extra></extra>",
            )

            fig.update_layout(
                height=400,
                showlegend=True,
                margin=dict(l=0, r=0, t=40, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)
            logger.info("Portfolio allocation pie chart displayed successfully")


def _display_expected_returns(portfolio, financial_advisor_agent):
    """
    Display expected returns projections for 1, 3, 5, and 10 years.

    Args:
        portfolio: The portfolio dictionary containing assets
        financial_advisor_agent: The FinancialAdvisorAgent instance

    Returns:
        None (displays projected returns)
    """
    logger.debug("Calculating expected returns for 1/3/5/10 years")

    if "assets" not in portfolio or not isinstance(portfolio["assets"], list):
        return

    # Display disclaimer about past performance
    st.warning(
        "⚠️ **Importante:** I rendimenti passati non sono indicatori affidabili dei rendimenti futuri. "
        "Queste proiezioni si basano su medie storiche e possono variare significativamente."
    )

    # Calculate weighted returns
    total_allocation = 0
    expected_returns = {1: 0, 3: 0, 5: 0, 10: 0}
    available_years = set()

    with st.spinner("📊 Calculating expected returns..."):
        for asset in portfolio["assets"]:
            asset_symbol = (
                asset.get("symbol") if isinstance(asset, dict) else asset.symbol
            )
            asset_percentage = (
                asset.get("percentage") if isinstance(asset, dict) else asset.percentage
            )

            if asset_symbol and asset_percentage:
                try:
                    # Get asset data
                    if asset_symbol.upper() == "BITCOIN":
                        asset_symbol = "BTC-EUR"

                    result_data = financial_advisor_agent.analyze_asset(
                        asset_symbol, years=10
                    )

                    if result_data.get("success"):
                        returns = result_data.get("returns", [])

                        # Calculate average return for each period
                        for period in [1, 3, 5, 10]:
                            period_returns = [
                                ret.get("percentage", 0)
                                for ret in returns
                                if ret.get("year") == period
                            ]
                            if period_returns:
                                expected_returns[period] += (
                                    period_returns[0] * asset_percentage / 100
                                )
                                available_years.add(period)

                        total_allocation += asset_percentage

                except Exception as e:
                    logger.warning(
                        "Failed to calculate returns for %s: %s", asset_symbol, str(e)
                    )

    if total_allocation > 0:
        # Display expected returns in columns
        available_years = sorted(available_years)
        num_cols = len(available_years)

        if num_cols > 0:
            cols = st.columns(num_cols)

            for idx, period in enumerate(available_years):
                with cols[idx]:
                    st.metric(
                        f"{period}-Year Expected Return",
                        f"{expected_returns[period]:.2f}%",
                        delta=None,
                        help="Basato sui rendimenti storici medi ponderati per l'allocazione del portafoglio",
                    )

            logger.info("Expected returns displayed successfully")


def _extract_financial_metrics(
    profile: FinancialProfile, financial_advisor_agent
) -> tuple:
    """
    Extract PAC metrics from financial profile using structured response.

    Args:
        profile: The FinancialProfile object
        financial_advisor_agent: The FinancialAdvisorAgent instance for structured extraction

    Returns:
        Tuple of (initial_investment, monthly_savings) with proper values from LLM
    """
    logger.debug("Extracting PAC metrics from profile using structured response")

    try:
        if profile is None:
            logger.warning("Profile is None, returning defaults")
            return 5000, 200

        # Convert profile to dict for agent
        profile_dict = (
            profile.model_dump() if hasattr(profile, "model_dump") else profile
        )
        logger.debug("Profile dict keys: %s", list(profile_dict.keys()))

        # Use structured response to extract PAC metrics
        pac_metrics = financial_advisor_agent.extract_pac_metrics(profile_dict)

        initial_investment = pac_metrics.initial_investment
        monthly_savings = pac_metrics.monthly_savings

        logger.info(
            "PAC METRICS EXTRACTED - Using structured response: Initial €%d, Monthly €%.0f",
            initial_investment,
            monthly_savings,
        )
        return initial_investment, monthly_savings

    except Exception as e:
        logger.error("Error extracting PAC metrics: %s", str(e))
        logger.warning(
            "PAC METRICS EXTRACTION FAILED - Returning defaults: Initial €5000, Monthly €200"
        )
        return 5000, 200  # Return defaults


def _display_wealth_simulation(
    portfolio, financial_advisor_agent, financial_profile=None
):
    """
    Display long-term wealth simulation with Monte Carlo scenarios and PAC simulation.

    Args:
        portfolio: The portfolio dictionary containing assets
        financial_advisor_agent: The FinancialAdvisorAgent instance
        financial_profile: Optional FinancialProfile object for personalized PAC parameters

    Returns:
        None (displays simulation charts)
    """
    logger.debug("Running wealth simulation with Monte Carlo and PAC")

    if "assets" not in portfolio or not isinstance(portfolio["assets"], list):
        return

    # Display explanation of Long-Term Wealth Projection
    st.markdown(
        """
    **📊 Cos'è questa simulazione?**

    La simulazione utilizza il **Monte Carlo**, un metodo statistico che proietta 1.000 scenari di crescita del portafoglio
    su 20 anni, considerando volatilità storica e rendimenti medi. I tre scenari mostrati rappresentano:
    - **Pessimistico (10° percentile)**: Solo 1 scenario su 10 avrà risultati peggiori
    - **Atteso (50° percentile)**: Il valore più probabile (mediana)
    - **Ottimistico (75° percentile)**: 3 scenari su 4 rimangono sotto questo valore
    """
    )

    try:
        with st.spinner("📈 Running wealth simulation (this may take a moment)..."):
            # Get historical data for all assets
            asset_returns = {}
            asset_volatility = {}
            total_weight = 0

            for asset in portfolio["assets"]:
                asset_symbol = (
                    asset.get("symbol") if isinstance(asset, dict) else asset.symbol
                )
                asset_percentage = (
                    asset.get("percentage")
                    if isinstance(asset, dict)
                    else asset.percentage
                )

                if asset_symbol and asset_percentage:
                    try:
                        if asset_symbol.upper() == "BITCOIN":
                            asset_symbol = "BTC-EUR"
                        elif asset_symbol.upper() == "GOLD":
                            asset_symbol = "SGLD"

                        result_data = financial_advisor_agent.analyze_asset(
                            asset_symbol, years=10
                        )

                        if result_data.get("success"):
                            returns = result_data.get("returns", [])
                            # Simple volatility estimate from returns variance
                            if len(returns) > 1:
                                return_values = [
                                    r.get("percentage", 0) for r in returns
                                ]
                                avg_return = np.mean(return_values)
                                volatility = np.std(return_values)

                                asset_returns[asset_symbol] = avg_return / 100
                                asset_volatility[asset_symbol] = max(
                                    volatility / 100, MONTECARLO_MIN_ASSET_VOLATILITY
                                )
                                total_weight += asset_percentage

                    except Exception as e:
                        logger.warning(
                            "Failed to get data for %s: %s", asset_symbol, str(e)
                        )

            if asset_returns and total_weight > 0:
                # Calculate portfolio metrics using weighted averages
                # This ensures consistency with Expected Returns projections
                portfolio_return = sum(
                    (
                        asset_returns.get(symbol, 0)
                        * (
                            next(
                                (
                                    (
                                        a.get("percentage")
                                        if isinstance(a, dict)
                                        else a.percentage
                                    )
                                    for a in portfolio["assets"]
                                    if (
                                        a.get("symbol")
                                        if isinstance(a, dict)
                                        else a.symbol
                                    )
                                    == symbol
                                ),
                                0,
                            )
                            / 100
                        )
                    )
                    for symbol in asset_returns
                )

                portfolio_volatility = np.sqrt(
                    sum(
                        (
                            (
                                asset_volatility.get(symbol, 0)
                                * (
                                    next(
                                        (
                                            (
                                                a.get("percentage")
                                                if isinstance(a, dict)
                                                else a.percentage
                                            )
                                            for a in portfolio["assets"]
                                            if (
                                                a.get("symbol")
                                                if isinstance(a, dict)
                                                else a.symbol
                                            )
                                            == symbol
                                        ),
                                        0,
                                    )
                                    / 100
                                )
                            )
                            ** 2
                        )
                        for symbol in asset_volatility
                    )
                )

                # Calculate annualized return for accurate Monte Carlo simulation
                # Using the same methodology as Expected Returns: average of historical annual returns
                portfolio_return_annualized = portfolio_return

                # Extract personalized PAC parameters from financial profile if available
                logger.info(
                    "PAC SECTION - financial_profile is: %s",
                    "NOT NONE" if financial_profile else "NONE",
                )
                if financial_profile:
                    logger.info(
                        "PAC SECTION - Extracting metrics from profile using agent"
                    )
                    (
                        initial_investment,
                        monthly_contribution,
                    ) = _extract_financial_metrics(
                        financial_profile, financial_advisor_agent
                    )
                    logger.info(
                        "PAC SECTION - Using personalized PAC parameters from profile: Initial €%d, Monthly €%.0f",
                        initial_investment,
                        monthly_contribution,
                    )
                else:
                    initial_investment = 5000
                    monthly_contribution = 200
                    logger.warning(
                        "PAC SECTION - Profile is NONE, using default PAC parameters: Initial €%d, Monthly €%d",
                        initial_investment,
                        monthly_contribution,
                    )

                # Check if initial_investment is 0 and replace with symbolic value
                if initial_investment == 0:
                    logger.warning(
                        "PAC SECTION - Initial investment is 0, using symbolic value of €%s",
                        MONTECARLO_DEFAULT_INITIAL_INVESTMENT,
                    )
                    initial_investment = MONTECARLO_DEFAULT_INITIAL_INVESTMENT

                time_steps = MONTECARLO_SIMULATION_YEARS * 12  # Monthly steps

                # Run Monte Carlo simulation
                np.random.seed(42)
                simulations = np.zeros((time_steps, MONTECARLO_SIMULATION_SCENARIOS))
                simulations[0] = initial_investment

                monthly_return = portfolio_return_annualized / 12
                monthly_volatility = portfolio_volatility / np.sqrt(12)

                for t in range(1, time_steps):
                    random_returns = np.random.normal(
                        monthly_return,
                        monthly_volatility,
                        MONTECARLO_SIMULATION_SCENARIOS,
                    )
                    simulations[t] = simulations[t - 1] * (1 + random_returns)

                # Calculate percentiles (using 10th, 50th, and 75th)
                percentile_10 = np.percentile(simulations, 10, axis=1)
                percentile_50 = np.percentile(simulations, 50, axis=1)
                percentile_75 = np.percentile(simulations, 75, axis=1)

                # Create time array in years
                time_array = np.linspace(0, MONTECARLO_SIMULATION_YEARS, time_steps)

                # Create figure for lump sum
                fig_lump = go.Figure()

                # Add pessimistic scenario (10th percentile)
                fig_lump.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=percentile_10,
                        name="Pessimistic (10th %ile)",
                        line=dict(color="rgba(255, 0, 0, 0.3)", width=2),
                        fill=None,
                    )
                )

                # Add optimistic scenario (75th percentile)
                fig_lump.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=percentile_75,
                        name="Optimistic (75th %ile)",
                        line=dict(color="rgba(0, 255, 0, 0.3)", width=2),
                        fill="tonexty",
                        fillcolor="rgba(0, 255, 0, 0.1)",
                    )
                )

                # Add median scenario
                fig_lump.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=percentile_50,
                        name="Expected (Median)",
                        line=dict(color="rgb(0, 100, 200)", width=3),
                    )
                )

                fig_lump.update_layout(
                    title="Lump Sum Investment - 20-Year Wealth Projection",
                    xaxis_title="Years",
                    yaxis_title="Portfolio Value (€)",
                    hovermode="x unified",
                    height=450,
                    template="plotly_white",
                    yaxis=dict(tickformat="€,.0f"),
                    margin=dict(l=0, r=0, t=40, b=0),
                )

                st.plotly_chart(fig_lump, use_container_width=True)

                # Display lump sum starting data
                st.markdown("#### Starting Data")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Initial Investment",
                        f"€{initial_investment:,.0f}",
                        help="Starting lump sum amount",
                    )

                with col2:
                    st.metric(
                        "Portfolio Return",
                        f"{portfolio_return_annualized*100:.2f}%",
                        help="Expected annual return based on historical average",
                    )

                with col3:
                    st.metric(
                        "Portfolio Volatility",
                        f"{portfolio_volatility*100:.2f}%",
                        help="Annual volatility (risk) of the portfolio",
                    )

                with col4:
                    st.metric(
                        "Time Horizon",
                        f"{MONTECARLO_SIMULATION_YEARS} years",
                        help=f"Projection period: {MONTECARLO_SIMULATION_YEARS} years with monthly compounding",
                    )

                # Display lump sum statistics
                st.markdown("#### Lump Sum Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Pessimistic Outcome",
                        f"€{percentile_10[-1]:,.0f}",
                        help="10th percentile - worst case scenario",
                    )

                with col2:
                    st.metric(
                        "Expected Outcome",
                        f"€{percentile_50[-1]:,.0f}",
                        help="Median - most likely scenario",
                    )

                with col3:
                    st.metric(
                        "Optimistic Outcome",
                        f"€{percentile_75[-1]:,.0f}",
                        help="75th percentile - best case scenario",
                    )

                # PAC Simulation (Piano di Accumulo - Monthly contributions)
                st.divider()
                st.markdown("#### Piano di Accumulo (PAC) - Monthly Contributions")

                st.markdown(
                    """
                **Cos'è un Piano di Accumulo?**

                Un **PAC (Piano di Accumulo del Capitale)** è una strategia di investimento dove depositi una somma fissa
                (es. €250) ogni mese, indipendentemente dall'andamento del mercato.

                **Perché è importante?**
                - **Riduce il rischio**: Investi a prezzi sia alti che bassi, mediando il costo
                - **Disciplina**: Investimenti regolari senza bisogno di prendere decisioni basate emozioni
                - **Effetto composto**: Il denaro investito ha più tempo per crescere
                - **Psicologicamente più sostenibile**: Meno stress rispetto agli investimenti in unica soluzione
                """
                )

                # PAC simulations (monthly_contribution and initial_investment already set above)
                pac_simulations = np.zeros(
                    (time_steps, MONTECARLO_SIMULATION_SCENARIOS)
                )

                np.random.seed(42)
                for t in range(time_steps):
                    if t == 0:
                        # Start PAC with initial deposit + first monthly contribution
                        pac_simulations[t] = initial_investment + monthly_contribution
                    else:
                        random_returns = np.random.normal(
                            monthly_return,
                            monthly_volatility,
                            MONTECARLO_SIMULATION_SCENARIOS,
                        )
                        pac_simulations[t] = (
                            pac_simulations[t - 1] * (1 + random_returns)
                            + monthly_contribution
                        )

                # Calculate percentiles for PAC (using 10th, 50th, and 75th)
                pac_percentile_10 = np.percentile(pac_simulations, 10, axis=1)
                pac_percentile_50 = np.percentile(pac_simulations, 50, axis=1)
                pac_percentile_75 = np.percentile(pac_simulations, 75, axis=1)

                # Calculate cumulative capital invested over time
                cumulative_invested = np.array(
                    [
                        initial_investment + (monthly_contribution * t)
                        for t in range(time_steps)
                    ]
                )

                # Create figure for PAC
                fig_pac = go.Figure()

                # Add pessimistic scenario
                fig_pac.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=pac_percentile_10,
                        name="Pessimistic (10th %ile)",
                        line=dict(color="rgba(255, 0, 0, 0.3)", width=2),
                        fill=None,
                    )
                )

                # Add optimistic scenario
                fig_pac.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=pac_percentile_75,
                        name="Optimistic (75th %ile)",
                        line=dict(color="rgba(0, 255, 0, 0.3)", width=2),
                        fill="tonexty",
                        fillcolor="rgba(0, 255, 0, 0.1)",
                    )
                )

                # Add median scenario
                fig_pac.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=pac_percentile_50,
                        name="Expected (Median)",
                        line=dict(color="rgb(200, 100, 0)", width=3),
                    )
                )

                # Add cumulative invested capital
                fig_pac.add_trace(
                    go.Scatter(
                        x=time_array,
                        y=cumulative_invested,
                        name="Total Invested",
                        line=dict(color="rgb(50, 50, 150)", width=2, dash="dash"),
                    )
                )

                fig_pac.update_layout(
                    title=f"PAC Investment - Monthly Contribution: €{monthly_contribution:,.0f}",
                    xaxis_title="Years",
                    yaxis_title="Amount (€)",
                    hovermode="x unified",
                    height=450,
                    template="plotly_white",
                    yaxis=dict(tickformat="€,.0f"),
                    margin=dict(l=0, r=0, t=40, b=0),
                )

                st.plotly_chart(fig_pac, use_container_width=True)

                # Display PAC starting data
                st.markdown("#### Starting Data")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "Initial Deposit",
                        f"€{initial_investment:,.0f}",
                        help="Starting amount for PAC",
                    )

                with col2:
                    st.metric(
                        "Monthly Contribution",
                        f"€{monthly_contribution:,.0f}",
                        help="Fixed monthly investment amount",
                    )

                with col3:
                    st.metric(
                        "Portfolio Return",
                        f"{portfolio_return_annualized*100:.2f}%",
                        help="Expected annual return based on historical average",
                    )

                with col4:
                    st.metric(
                        "Portfolio Volatility",
                        f"{portfolio_volatility*100:.2f}%",
                        help="Annual volatility (risk) of the portfolio",
                    )

                with col5:
                    st.metric(
                        "Time Horizon",
                        f"{MONTECARLO_SIMULATION_YEARS} years",
                        help=f"Projection period: {MONTECARLO_SIMULATION_YEARS} years with monthly compounding",
                    )

                # Display PAC statistics
                st.markdown("#### PAC Statistics")

                # Calculate total amount invested
                total_invested = initial_investment + (
                    monthly_contribution * time_steps
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Invested",
                        f"€{total_invested:,.0f}",
                        help=f"Initial: €{initial_investment:,} + {time_steps} months × €{monthly_contribution:,}",
                    )

                with col2:
                    st.metric(
                        "Pessimistic Outcome",
                        f"€{pac_percentile_10[-1]:,.0f}",
                        help="10th percentile - worst case scenario",
                    )

                with col3:
                    st.metric(
                        "Expected Outcome",
                        f"€{pac_percentile_50[-1]:,.0f}",
                        help="Median - most likely scenario",
                    )

                with col4:
                    st.metric(
                        "Optimistic Outcome",
                        f"€{pac_percentile_75[-1]:,.0f}",
                        help="75th percentile - best case scenario",
                    )

                logger.info("Wealth simulation displayed successfully")

    except Exception as e:
        logger.error("Error in wealth simulation: %s", str(e), exc_info=True)
        st.warning("Could not generate wealth simulation. Please try again later.")


def _display_portfolio_details(portfolio, financial_advisor_agent):
    """
    Display all portfolio details including allocation, strategy, risk level, and returns.

    Args:
        portfolio: The portfolio dictionary
        financial_advisor_agent: The FinancialAdvisorAgent instance

    Returns:
        None (displays UI components)
    """
    logger.debug("Displaying generated portfolio")

    # Display portfolio allocation
    st.markdown("### 📈 Portfolio Allocation")

    # Helper function to display asset
    def display_asset(asset_name, percentage, justification):
        if asset_name and percentage:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(asset_name, f"{percentage}%")
            with col2:
                st.caption(justification if justification else "")

    # Display all assets from the portfolio (new structure with nested assets)
    if "assets" in portfolio and isinstance(portfolio["assets"], list):
        for asset in portfolio["assets"]:
            display_asset(
                asset.get("symbol") if isinstance(asset, dict) else asset.symbol,
                (
                    asset.get("percentage")
                    if isinstance(asset, dict)
                    else asset.percentage
                ),
                (
                    asset.get("justification")
                    if isinstance(asset, dict)
                    else asset.justification
                ),
            )

    # Display portfolio pie chart
    _display_portfolio_pie_chart(portfolio)

    # Display overall strategy reasoning
    st.divider()
    if "portfolio_reasoning" in portfolio:
        st.markdown("### 🎯 Investment Strategy")
        st.info(portfolio["portfolio_reasoning"])

    # Display risk level
    if "risk_level" in portfolio:
        # Extract the value from the risk_level (could be enum or string)
        risk_value = portfolio["risk_level"]
        if isinstance(risk_value, str):
            risk_level = risk_value.upper()
        else:
            # Handle enum or other types
            risk_level = str(risk_value).replace("RiskLevel.", "").upper()

        if risk_level == "CONSERVATIVE":
            st.success(f"**Risk Level**: 🛡️ {risk_level}")
        elif risk_level == "MODERATE":
            st.info(f"**Risk Level**: ⚖️ {risk_level}")
        else:
            st.warning(f"**Risk Level**: ⚡ {risk_level}")

    # Display rebalancing schedule
    if "rebalancing_schedule" in portfolio:
        st.markdown(f"**Rebalancing Schedule**: {portfolio['rebalancing_schedule']}")

    # Display key considerations
    _display_key_considerations(portfolio)

    # Display historical returns for assets
    st.markdown("### 📊 Historical Returns (Last 10 Years)")
    _fetch_and_display_historical_returns(portfolio, financial_advisor_agent)

    # Display expected returns
    st.divider()
    st.markdown("### 💰 Expected Returns Projections")
    _display_expected_returns(portfolio, financial_advisor_agent)

    # Display wealth simulation
    st.divider()
    st.markdown("### 🔮 Long-Term Wealth Projection")
    _display_wealth_simulation(
        portfolio,
        financial_advisor_agent,
        financial_profile=st.session_state.financial_profile,
    )
    logger.info("Portfolio display completed")


def main():
    """Main Streamlit application."""
    logger.debug("Starting main Streamlit application")

    # Initialize session state
    _initialize_session_state()

    # Provider selection modal on first load
    if st.session_state.provider is None:
        _show_provider_selection()
        return

    # Show loading screen while initializing agent
    if not st.session_state.agent_initialized:
        _show_loading_screen()
        chatbot_agent, financial_advisor_agent = _initialize_agents()
        st.rerun()

    # Provider has been selected and agent is initialized
    logger.debug(
        "Provider selected and agent initialized: %s", st.session_state.provider
    )

    try:
        logger.debug(
            "Retrieving cached chatbot and financial advisor for provider: %s",
            st.session_state.provider,
        )
        # Prefer stored instances in session_state to avoid re-initialization
        chatbot_agent = st.session_state.get("chatbot_agent") or initialize_chatbot(
            st.session_state.provider
        )
        financial_advisor_agent = st.session_state.get(
            "financial_advisor_agent"
        ) or initialize_financial_advisor(st.session_state.provider)
        # Ensure session_state holds the instances
        st.session_state["chatbot_agent"] = chatbot_agent
        st.session_state["financial_advisor_agent"] = financial_advisor_agent
        logger.info("Agents retrieved successfully")
    except Exception as e:
        logger.error("Failed to retrieve agent: %s", str(e), exc_info=True)
        st.error(f"Failed to retrieve agent: {e}")
        st.session_state.provider = None
        st.session_state.agent_initialized = False
        st.rerun()

    # Header
    _show_header()

    # Sidebar
    _setup_sidebar(chatbot_agent, financial_advisor_agent)

    # Send welcome message on first load
    if len(st.session_state.messages) == 0:
        logger.debug("No messages in session state, sending welcome message")

        try:
            # Get the first question (without advancing yet - we're at index 0)
            first_question = chatbot_agent.get_current_question()
            logger.debug("First question: %s", first_question)

            # Format the welcome prompt template
            welcome_prompt = chatbot_agent.welcome_prompt.format(
                first_question=first_question
            )

            welcome_message = chatbot_agent.chat(welcome_prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_message}
            )

            logger.debug("Welcome message sent")
        except Exception as e:
            logger.error(
                "Failed to generate welcome message: %s", str(e), exc_info=True
            )

    # Display chat messages from history
    logger.debug("Displaying %d messages from history", len(st.session_state.messages))
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Show message if conversation is completed
    if st.session_state.conversation_completed:
        if not st.session_state.generated_portfolio:
            st.toast(
                "Analyzing profile and generating portfolio...",
                icon="🔄",
                duration="long",
            )

        logger.debug("Conversation is completed")

        # Generate and display portfolio
        if st.session_state.financial_profile:
            st.divider()
            st.subheader("💼 AI-Generated Portfolio Recommendation")

            logger.debug("Preparing portfolio generation")
            logger.debug(
                "Generated portfolio state: %s", st.session_state.generated_portfolio
            )

            # Auto-generate portfolio if not already generated
            if not st.session_state.generated_portfolio:
                logger.info("Portfolio not yet generated, auto-generating...")
                with st.spinner(
                    "🔄 Analyzing profile and retrieving relevant ETF assets..."
                ):
                    portfolio = _generate_portfolio_for_profile(
                        financial_advisor_agent, st.session_state.financial_profile
                    )
                    if portfolio:
                        st.session_state.generated_portfolio = portfolio
                        logger.info("Portfolio auto-generated successfully")
                    else:
                        logger.warning("Failed to generate portfolio")
                        st.error(
                            "Could not auto-generate portfolio. Try clicking the button below."
                        )
                        if st.button(
                            "Generate Portfolio with RAG", key="generate_portfolio"
                        ):
                            logger.info("Portfolio generation requested manually")
                            portfolio = _generate_portfolio_for_profile(
                                financial_advisor_agent,
                                st.session_state.financial_profile,
                            )
                            if portfolio:
                                st.session_state.generated_portfolio = portfolio
            else:
                logger.debug("Portfolio already generated, displaying...")

            # Display generated portfolio if available
            if (
                "generated_portfolio" in st.session_state
                and st.session_state.generated_portfolio
            ):
                _display_portfolio_details(
                    st.session_state.generated_portfolio, financial_advisor_agent
                )

                # Show completion message after portfolio
                st.divider()
                st.toast(
                    "Assessment Completed Successfully! Your personalized portfolio analysis is ready.",
                    icon="🎉",
                    duration="long",
                )

                # Celebration animation and message
                st.balloons()
                st.success(
                    "🎉 **Assessment Completed Successfully!** 🎉\n\n"
                    "All your financial questions have been answered and your personalized portfolio analysis is ready! "
                    "Your financial profile and PAC metrics have been extracted and analyzed."
                )

                # Financial Profile in an expanded section
                with st.expander(
                    "📊 View Your Financial Profile & Summary", expanded=False
                ):
                    _display_financial_profile_summary()
                    st.info(
                        "💡 **Next Steps:**\n"
                        "- Review your portfolio allocation above\n"
                        "- Consider consulting with a financial advisor for personalized advice\n"
                        "- Click 'Clear Conversation' to start a new assessment or 'Change Provider' to start over"
                    )

        else:
            logger.debug("No financial profile available to display")
    else:
        # User input - only show if conversation is not completed
        if prompt := st.chat_input("Ask me about your finances..."):
            logger.debug("User input received: %s", prompt[:100])

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Add to session state
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        logger.debug("Getting response from agent")
                        logger.debug(
                            "Current question index before advance: %d",
                            chatbot_agent.current_question_index,
                        )

                        # Check if there are more questions after this one
                        has_more_questions = chatbot_agent.advance_to_next_question()
                        logger.debug(
                            "After advance - has_more_questions: %s, current_index: %d",
                            has_more_questions,
                            chatbot_agent.current_question_index,
                        )

                        if has_more_questions:
                            # There are more questions to ask
                            next_question = chatbot_agent.get_current_question()
                            logger.debug(
                                "Next question available: %s", next_question[:50]
                            )

                            # Format the acknowledge and ask prompt template
                            response_prompt = (
                                chatbot_agent.acknowledge_and_ask_prompt.format(
                                    user_answer=prompt, next_question=next_question
                                )
                            )

                            # Get response from agent
                            response_text = chatbot_agent.chat(response_prompt)
                            logger.debug(
                                "Response generated, length: %d", len(response_text)
                            )

                            # Display response with streaming effect
                            stream_text(response_text)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response_text}
                            )

                            logger.info("Chat interaction completed successfully")
                        else:
                            # All questions have been answered
                            logger.info(
                                "All questions have been answered, generating summary"
                            )

                            # Get the full conversation history from session state
                            conversation_lines = []
                            for msg in st.session_state.messages:
                                role = (
                                    "Utente" if msg["role"] == "user" else "Assistente"
                                )
                                conversation_lines.append(f"{role}: {msg['content']}")

                            conversation_history = "\n\n".join(conversation_lines)
                            logger.debug(
                                "Conversation history prepared, length: %d",
                                len(conversation_history),
                            )

                            # Format the summary prompt with the conversation history
                            summary_prompt = chatbot_agent.summary_prompt.format(
                                user_answer=prompt,
                                conversation_history=conversation_history,
                            )
                            response_text = chatbot_agent.chat(summary_prompt)
                            logger.debug(
                                "Final summary generated, length: %d",
                                len(response_text),
                            )
                            logger.info("All questions completed, summary provided")

                            # Extract structured financial profile using financial advisor
                            logger.debug(
                                "Extracting structured financial profile using financial advisor"
                            )
                            try:
                                # Ensure financial advisor instance is available
                                financial_advisor_agent = st.session_state.get(
                                    "financial_advisor_agent"
                                ) or initialize_financial_advisor(
                                    st.session_state.provider
                                )
                                st.session_state["financial_advisor_agent"] = (
                                    financial_advisor_agent
                                )

                                financial_profile = (
                                    financial_advisor_agent.extract_financial_profile(
                                        response_text
                                    )
                                )
                                logger.info("Financial profile extracted successfully")

                                # Store profile in session state for display
                                st.session_state.financial_profile = financial_profile
                                logger.debug("Profile stored in session state")
                            except Exception as profile_error:
                                logger.warning(
                                    "Failed to extract financial profile: %s",
                                    str(profile_error),
                                )
                                st.session_state.financial_profile = None

                            # Mark conversation as completed for next render
                            st.session_state.conversation_completed = True
                            logger.info("Conversation marked as completed")

                            # Display response with streaming effect
                            stream_text(response_text)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": response_text}
                            )

                            logger.info("Chat interaction completed successfully")
                except Exception as e:
                    logger.error(
                        "Error processing user input: %s", str(e), exc_info=True
                    )
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    logger.info("Starting Financial AI Agent application")
    main()
