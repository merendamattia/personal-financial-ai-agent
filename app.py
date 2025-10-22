"""
Financial AI Agent Streamlit Application.

This module provides a web interface for interacting with the financial
AI agent powered by datapizza-ai and multiple LLM providers.
"""

import logging
import os

import requests
import streamlit as st
from dotenv import load_dotenv

from src.chatbot_agent import ChatBotAgent
from src.clients import list_providers

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
    initial_sidebar_state="expanded",
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
    Check if Ollama is running at localhost:11434.

    Returns:
        True if Ollama is available, False otherwise
    """
    logger.debug("Checking Ollama availability at localhost:11434")

    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        is_available = "Ollama is running" in response.text
        logger.info("Ollama availability check: %s", is_available)
        return is_available
    except (requests.exceptions.RequestException, Exception) as e:
        logger.warning("Ollama not available: %s", str(e))
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
        st.stop()

    # Provider has been selected, initialize agent
    logger.debug("Provider selected: %s", st.session_state.provider)

    try:
        logger.debug("Initializing agent for selected provider")
        agent = initialize_agent(st.session_state.provider)
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize agent: %s", str(e), exc_info=True)
        st.error(f"❌ Failed to initialize agent: {e}")
        st.session_state.provider = None
        st.rerun()

    # Header
    st.title("💰 Financial AI Agent")
    st.markdown("Your personal financial advisor powered by AI")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Provider info
        if agent.is_healthy():
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
            st.rerun()

        # Clear history button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            logger.debug("Clearing conversation history")
            agent.clear_memory()
            st.session_state.messages = []
            st.success("Conversation cleared!")

        # Model info
        st.divider()
        st.subheader("📊 Model Information")
        config = agent.get_config_summary()
        st.write(f"**Provider:** {config['provider'].upper()}")
        st.write(f"**Model:** {config['model']}")
        st.write(f"**Agent:** {config['name']}")
        logger.debug("Config summary displayed: %s", config)

    # Send welcome message on first load
    if len(st.session_state.messages) == 0:
        logger.debug("No messages in session state, sending welcome message")

        try:
            welcome_message = agent.chat(
                "Give a brief and friendly greeting as a financial advisor assistant. Write max 2 sentences in plain text (do not use JSON format or anything else)."
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_message}
            )
            logger.debug("Welcome message added to session")
        except Exception as e:
            logger.error(
                "Failed to generate welcome message: %s", str(e), exc_info=True
            )

    # Display chat messages from history
    logger.debug("Displaying %d messages from history", len(st.session_state.messages))
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
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
                    # Get response from agent
                    response_text = agent.chat(prompt)

                    logger.debug("Response generated, length: %d", len(response_text))

                    # Display response
                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                    logger.info("Chat interaction completed successfully")
            except Exception as e:
                logger.error("Error processing user input: %s", str(e), exc_info=True)
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


if __name__ == "__main__":
    logger.info("Starting Financial AI Agent application")
    main()
