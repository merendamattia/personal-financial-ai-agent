"""
Financial AI Agent Streamlit Application.

This module provides a web interface for interacting with the financial
AI agent powered by datapizza-ai and multiple LLM providers.
"""

import requests
import streamlit as st
from dotenv import load_dotenv

from src.chatbot_agent import ChatBotAgent
from src.clients import list_providers

# Load environment variables
load_dotenv()

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
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        return "Ollama is running" in response.text
    except (requests.exceptions.RequestException, Exception):
        return False


def get_available_providers() -> list:
    """
    Get list of available providers.
    Ollama is only available if it's running.

    Returns:
        List of available provider names
    """
    all_providers = list_providers()
    available = []

    for provider in all_providers:
        if provider == "ollama":
            if is_ollama_available():
                available.append(provider)
        else:
            # Google and OpenAI are always available (if API keys are set, they'll work)
            available.append(provider)

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
    return ChatBotAgent(provider=provider)


def main():
    """Main Streamlit application."""

    # Initialize session state for provider selection
    if "provider" not in st.session_state:
        st.session_state.provider = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Provider selection modal on first load
    if st.session_state.provider is None:
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
                if st.button("🏠 Ollama (Local)", use_container_width=True):
                    st.session_state.provider = "ollama"
                    st.rerun()
            else:
                st.button(
                    "🏠 Ollama (Not Running)",
                    disabled=True,
                    use_container_width=True,
                    help="Start Ollama with: ollama serve",
                )

        with col2:
            if "google" in available_providers:
                if st.button("🔴 Google Gemini", use_container_width=True):
                    st.session_state.provider = "google"
                    st.rerun()
            else:
                st.button(
                    "🔴 Google (Not Available)",
                    disabled=True,
                    use_container_width=True,
                )

        with col3:
            if "openai" in available_providers:
                if st.button("⚪ OpenAI GPT", use_container_width=True):
                    st.session_state.provider = "openai"
                    st.rerun()
            else:
                st.button(
                    "⚪ OpenAI (Not Available)",
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
    try:
        agent = initialize_agent(st.session_state.provider)
    except Exception as e:
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
            st.success("✅ Agent initialized!")
        else:
            st.warning("⚠️ Agent running but connection may be unstable")

        # Change provider
        if st.button("🔄 Change Provider", use_container_width=True):
            st.session_state.provider = None
            st.session_state.messages = []
            st.rerun()

        # Clear history button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
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

    # Send welcome message on first load
    if len(st.session_state.messages) == 0:
        welcome_message = agent.chat(
            "Give a brief and friendly greeting as a financial advisor assistant. Write max 2 sentences."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": welcome_message}
        )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask me about your finances..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    # Get response from agent
                    response_text = agent.chat(prompt)

                    # Display response
                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


if __name__ == "__main__":
    main()
