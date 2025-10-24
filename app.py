"""
Financial AI Agent Streamlit Application.

This module provides a web interface for interacting with the financial
AI agent powered by datapizza-ai and multiple LLM providers.
"""

import logging
import os
import time

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
            st.session_state.question_index = 0
            st.session_state.conversation_completed = False
            st.rerun()

        # Clear history button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            logger.debug("Clearing conversation history")
            agent.clear_memory()
            agent.reset_questions()
            st.session_state.messages = []
            st.session_state.question_index = 0
            st.session_state.conversation_completed = False
            st.success("Conversation cleared!")

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
            logger.debug("First question: %s", first_question[:200])

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
