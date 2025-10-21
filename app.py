"""
Financial AI Agent Streamlit Application.

This module provides a web interface for interacting with the financial
AI agent powered by datapizza-ai and Ollama.
"""

import streamlit as st
from dotenv import load_dotenv

from src.chatbot_agent import ChatBotAgent

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


@st.cache_resource
def initialize_agent() -> ChatBotAgent:
    """
    Initialize the financial chatbot agent (cached).

    Returns:
        Initialized ChatBotAgent instance
    """
    return ChatBotAgent()


def main():
    """Main Streamlit application."""

    # Header
    st.title("💰 Financial AI Agent")
    st.markdown("Your personal financial advisor powered by AI and Ollama")

    # Initialize agent first
    try:
        agent = initialize_agent()
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Initialize agent
        if agent.is_healthy():
            st.success("✅ Agent initialized!")
        else:
            st.warning("⚠️ Agent running but Ollama may not be available")

        # Clear history button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            agent.clear_memory()
            st.session_state.messages = []
            st.success("Conversation cleared!")

        # Model info
        st.divider()
        st.subheader("📊 Model Information")
        config = agent.get_config_summary()
        st.write(f"**Model:** {config['model']}")
        st.write(f"**Agent:** {config['name']}")
        st.write(f"**API:** {config['api_url']}")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Send welcome message on first load
    if len(st.session_state.messages) == 0:
        welcome_message = agent.chat(
            "Give a brief and friendly greeting as a financial advisor assistant. Write max 2 sentences in a markdown format."
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
                    # Get response from agent (memory is updated automatically)
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
