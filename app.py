"""
Financial AI Agent Streamlit Application.

This module provides a web interface for interacting with the financial
AI agent powered by datapizza-ai and multiple LLM providers.
"""

import logging
import os
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from src.clients import list_providers
from src.core import ChatBotAgent
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
        st.error(f"❌ Invalid JSON format: {str(e)}")
        return None
    except Exception as e:
        logger.error("Failed to load profile from JSON: %s", str(e))
        st.error(f"❌ Failed to load profile: {str(e)}")
        return None


@st.cache_resource
def initialize_agent(provider: str) -> ChatBotAgent:
    """
    Initialize the financial chatbot agent with the selected provider.

    Args:
        provider: The LLM provider to use

    Returns:
        Initialized ChatBotAgent instance
    """
    logger.debug("Initializing agent with provider: %s", provider)

    try:
        agent = ChatBotAgent(provider=provider)
        logger.info("Agent initialized successfully with provider: %s", provider)
        return agent
    except Exception as e:
        logger.error(
            "Failed to initialize agent with provider %s: %s",
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


def generate_portfolio_for_profile(agent, profile):
    """
    Generate a balanced portfolio for the given financial profile.

    Args:
        agent: The ChatBotAgent instance
        profile: The FinancialProfile object

    Returns:
        The generated Portfolio object, or None if generation failed
    """
    logger.info("Starting portfolio generation for profile")

    try:
        # Convert FinancialProfile object to dict for agent
        profile_dict = profile.model_dump()

        logger.debug("Converting profile to dict for portfolio generation")

        # Generate portfolio using RAG-enhanced agent with structured response
        portfolio = agent.generate_balanced_portfolio(profile_dict)

        logger.info("Portfolio generated successfully")
        if portfolio:
            assets_count = len(portfolio.assets) if hasattr(portfolio, "assets") else 0
            risk = portfolio.risk_level if hasattr(portfolio, "risk_level") else None
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
        st.error(f"❌ Failed to generate portfolio: {str(portfolio_error)}")
        return None


def main():
    """Main Streamlit application."""
    logger.debug("Starting main Streamlit application")

    # Initialize session state for provider selection
    if "provider" not in st.session_state:
        st.session_state.provider = None
        logger.debug("Initialized provider session state to None")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.debug("Initialized messages session state to empty list")
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
        logger.debug("Initialized question_index session state to 0")
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
        logger.debug("Initialized agent_initialized session state to False")
    if "conversation_completed" not in st.session_state:
        st.session_state.conversation_completed = False
        logger.debug("Initialized conversation_completed session state to False")
    if "financial_profile" not in st.session_state:
        st.session_state.financial_profile = None
        logger.debug("Initialized financial_profile session state to None")
    if "profile_loaded_from_json" not in st.session_state:
        st.session_state.profile_loaded_from_json = False
        logger.debug("Initialized profile_loaded_from_json session state to False")
    if "health_check_done" not in st.session_state:
        st.session_state.health_check_done = False
        logger.debug("Initialized health_check_done session state to False")
    if "agent_is_healthy" not in st.session_state:
        st.session_state.agent_is_healthy = False
        logger.debug("Initialized agent_is_healthy session state to False")

    # Provider selection modal on first load
    if st.session_state.provider is None:
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
        return

    # Show loading screen while initializing agent
    if not st.session_state.agent_initialized:
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

        try:
            logger.debug(
                "Initializing agent for provider: %s", st.session_state.provider
            )
            agent = initialize_agent(st.session_state.provider)
            logger.info("Agent initialized successfully")
            st.session_state.agent_initialized = True
            st.rerun()
        except Exception as e:
            logger.error("Failed to initialize agent: %s", str(e), exc_info=True)
            st.error(f"Failed to initialize agent: {e}")
            st.session_state.provider = None
            st.session_state.agent_initialized = False
            st.rerun()

    # Provider has been selected and agent is initialized
    logger.debug(
        "Provider selected and agent initialized: %s", st.session_state.provider
    )

    try:
        logger.debug("Getting cached agent for provider: %s", st.session_state.provider)
        agent = initialize_agent(st.session_state.provider)
        logger.info("Agent retrieved successfully")
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
    with st.sidebar:
        st.header("⚙️ Settings")

        # Provider info - health check only once
        if not st.session_state.health_check_done:
            logger.debug("Performing health check for the first time")
            st.session_state.agent_is_healthy = agent.is_healthy()
            st.session_state.health_check_done = True
            logger.debug(
                "Health check completed: %s", st.session_state.agent_is_healthy
            )

        if st.session_state.agent_is_healthy:
            logger.debug("Agent health check passed")
            st.success("✅ Agent initialized!")
        else:
            logger.warning("Agent health check failed")
            st.warning("⚠️ Agent running but connection may be unstable")

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
            agent.clear_memory()
            agent.reset_questions()
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
                st.success("✅ Profile loaded successfully!")
                st.rerun()

        # Model info
        st.divider()
        st.subheader("📊 Model Information")
        config = agent.get_config_summary()
        st.write(f"**Provider:** {config['provider'].upper()}")
        st.write(f"**Model:** {config['model']}")
        st.write(f"**Agent:** {config['name']}")
        logger.debug("Config summary displayed: %s", config)

        # Progress info
        st.divider()
        st.subheader("📋 Assessment Progress")
        progress = agent.get_questions_progress()
        st.write(
            f"**Question {progress['current_index'] + 1} of {progress['total_questions']}**"
        )
        st.progress(progress["progress_percentage"] / 100)
        logger.debug("Progress displayed: %s", progress)

    # Send welcome message on first load
    if len(st.session_state.messages) == 0:
        logger.debug("No messages in session state, sending welcome message")

        try:
            # Get the first question (without advancing yet - we're at index 0)
            first_question = agent.get_current_question()
            logger.debug("First question: %s", first_question)

            # Format the welcome prompt template
            welcome_prompt = agent.welcome_prompt.format(first_question=first_question)

            welcome_message = agent.chat(welcome_prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_message}
            )

            # Now advance to the next question index for the first user response
            agent.advance_to_next_question()
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
            "✅ **Assessment completed!** All your financial questions have been answered and the summary has been provided. "
            "Click 'Clear Conversation' to start a new assessment or 'Change Provider' to start over."
        )

        # Display financial profile if available
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

            # Generate balanced portfolio based on profile
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
                    portfolio = generate_portfolio_for_profile(
                        agent, st.session_state.financial_profile
                    )
                    if portfolio:
                        st.session_state.generated_portfolio = portfolio
                        logger.info("Portfolio auto-generated successfully")
                        st.rerun()
                    else:
                        logger.warning("Failed to generate portfolio")
                        st.error(
                            "❌ Could not auto-generate portfolio. Try clicking the button below."
                        )
                        if st.button(
                            "Generate Portfolio with RAG", key="generate_portfolio"
                        ):
                            logger.info("Portfolio generation requested manually")
                            portfolio = generate_portfolio_for_profile(
                                agent, st.session_state.financial_profile
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
                portfolio = st.session_state.generated_portfolio

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
                            asset.get("symbol")
                            if isinstance(asset, dict)
                            else asset.symbol,
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
                    st.markdown(
                        f"**Rebalancing Schedule**: {portfolio['rebalancing_schedule']}"
                    )

                # Display key considerations
                if (
                    "key_considerations" in portfolio
                    and portfolio["key_considerations"]
                ):
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

                logger.info("Portfolio display completed")

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
                            agent.current_question_index,
                        )

                        # Check if there are more questions AFTER this one
                        has_more_questions = agent.advance_to_next_question()
                        logger.debug(
                            "After advance - has_more_questions: %s, current_index: %d",
                            has_more_questions,
                            agent.current_question_index,
                        )

                        if has_more_questions:
                            # There are more questions to ask
                            next_question = agent.get_current_question()
                            logger.debug(
                                "Next question available: %s", next_question[:50]
                            )

                            # Format the acknowledge and ask prompt template
                            response_prompt = agent.acknowledge_and_ask_prompt.format(
                                user_answer=prompt, next_question=next_question
                            )

                            # Get response from agent
                            response_text = agent.chat(response_prompt)
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
                            summary_prompt = agent.summary_prompt.format(
                                user_answer=prompt,
                                conversation_history=conversation_history,
                            )
                            response_text = agent.chat(summary_prompt)
                            logger.debug(
                                "Final summary generated, length: %d",
                                len(response_text),
                            )
                            logger.info("All questions completed, summary provided")

                            # Extract structured financial profile
                            logger.debug("Extracting structured financial profile")
                            try:
                                financial_profile = agent.extract_financial_profile(
                                    response_text
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
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    logger.info("Starting Financial AI Agent application")
    main()
