"""
Financial AI Agent Streamlit Application.

This module provides a web interface for interacting with the financial
AI agent powered by datapizza-ai and multiple LLM providers.
"""

import logging
import os
import time
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from src.clients import list_providers
from src.core import ChatbotAgent, FinancialAdvisorAgent
from src.models import FinancialProfile, Portfolio

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
    initial_sidebar_state="auto",
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
                # asset_percentage = (
                #     asset.get("percentage")
                #     if isinstance(asset, dict)
                #     else asset.percentage
                # )

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
                                    # "Allocation %": asset_percentage,
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
                                    # "Allocation %": asset_percentage,
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
                                # "Allocation %": asset_percentage,
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
        st.info("📌 Unable to retrieve historical data for assets in the portfolio.")


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

    st.title("💰 Financial AI Agent")
    st.markdown("Your personal financial advisor powered by AI")

    st.divider()
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
            if st.button("🌐 Google Gemini", use_container_width=True, key="google_btn"):
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


def _show_loading_screen():
    """
    Display loading screen while initializing agent.

    Returns:
        None (displays UI with loading animation)
    """
    logger.debug("Agent not yet initialized, showing loading screen")

    st.title("💰 Financial AI Agent")
    st.markdown("Your personal financial advisor powered by AI")

    st.divider()

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

        if uploaded_json is not None and not st.session_state.profile_loaded_from_json:
            logger.debug("JSON file uploaded: %s", uploaded_json.name)
            profile = load_profile_from_json(uploaded_json)

            if profile:
                st.session_state.financial_profile = profile
                st.session_state.conversation_completed = True
                st.session_state.generated_portfolio = None
                st.session_state.profile_loaded_from_json = True
                logger.info(
                    "Profile loaded successfully, auto-generation will happen in display section"
                )
                st.success("Profile loaded successfully!")
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
        st.divider()
        st.subheader("📊 Financial Profile Summary")

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
                asset.get("percentage")
                if isinstance(asset, dict)
                else asset.percentage,
                asset.get("justification")
                if isinstance(asset, dict)
                else asset.justification,
            )

    # Display overall strategy reasoning
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
    st.title("💰 Financial AI Agent")
    st.markdown("Your personal financial advisor powered by AI")

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

            # Now advance to the next question index for the first user response
            chatbot_agent.advance_to_next_question()
            logger.debug("Welcome message sent, advanced to question index 1")
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
        logger.debug("Conversation is completed, showing completion message")
        st.info(
            "**Assessment completed!** All your financial questions have been answered and the summary has been provided. "
            "Click 'Clear Conversation' to start a new assessment or 'Change Provider' to start over."
        )

        # Display financial profile
        _display_financial_profile_summary()

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
                        st.rerun()
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
                                st.rerun()
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
                                financial_advisor = st.session_state.get(
                                    "financial_advisor"
                                ) or initialize_financial_advisor(
                                    st.session_state.provider
                                )
                                st.session_state[
                                    "financial_advisor"
                                ] = financial_advisor

                                financial_profile = (
                                    financial_advisor.extract_financial_profile(
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
