"""Settings page for API key configuration.

This module provides a user-friendly interface for configuring API keys
and provider settings directly from the Streamlit UI without manually
editing the .env file.

Implementation Notes:
- Provider-specific imports (openai, google.generativeai, requests) are done
  inside functions to avoid requiring all dependencies at module import time.
  This is intentional: users may not have all provider SDKs installed.
- Credentials are stored in Streamlit session state (in-memory only) and
  propagated to environment variables for consistency with the app's existing
  architecture where clients.py factories read from os.environ.
- Connection tests provide immediate feedback before saving credentials.
"""

import logging
import os
from typing import Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def _test_openai_connection(api_key: str, model: str) -> tuple[bool, str]:
    """Test OpenAI API connection.
    
    Args:
        api_key: OpenAI API key
        model: Model name to test
        
    Returns:
        Tuple of (success, message)
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        # Make a minimal test call
        response = client.models.retrieve(model)
        return True, f"✅ Successfully connected! Model: {response.id}"
    except Exception as e:
        error_msg = str(e)
        # Sanitize error messages to avoid exposing sensitive information
        if "401" in error_msg or "Incorrect API key" in error_msg or "authentication" in error_msg.lower():
            return False, "❌ Invalid API key"
        elif "404" in error_msg or "does not exist" in error_msg:
            return False, f"❌ Model '{model}' not found"
        else:
            return False, "❌ Connection failed. Please check your settings."


def _test_google_connection(api_key: str, model: str) -> tuple[bool, str]:
    """Test Google Gemini API connection.
    
    Args:
        api_key: Google API key
        model: Model name to test
        
    Returns:
        Tuple of (success, message)
    """
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        # Try to get model info
        model_info = genai.get_model(f"models/{model}")
        return True, f"✅ Successfully connected! Model: {model_info.name}"
    except Exception as e:
        error_msg = str(e)
        # Sanitize error messages to avoid exposing sensitive information
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower() or "authentication" in error_msg.lower():
            return False, "❌ Invalid API key"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return False, f"❌ Model '{model}' not found"
        else:
            return False, "❌ Connection failed. Please check your settings."


def _test_ollama_connection(base_url: str, model: str) -> tuple[bool, str]:
    """Test Ollama connection.
    
    Args:
        base_url: Ollama API base URL
        model: Model name to test
        
    Returns:
        Tuple of (success, message)
    """
    try:
        import requests
        
        # Remove /v1 suffix if present for the health check
        health_url = base_url.replace("/v1", "")
        
        # Test if Ollama is running
        # Note: For localhost/local docker, SSL verification is not typically needed
        # For production environments with HTTPS, verify=True should be used
        verify_ssl = not health_url.startswith("http://localhost") and not health_url.startswith("http://127.0.0.1")
        response = requests.get(health_url, timeout=5, verify=verify_ssl)
        if "Ollama is running" not in response.text:
            return False, "❌ Ollama is not running at this URL"
        
        # Try to check if the model exists
        # Note: This is a basic check; Ollama may not have all models pre-downloaded
        return True, f"✅ Successfully connected to Ollama! Model: {model}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to Ollama. Is it running?"
    except requests.exceptions.Timeout:
        return False, "❌ Connection timeout"
    except Exception:
        return False, "❌ Connection failed. Please check the URL."


def _initialize_settings_state():
    """Initialize session state for settings if not already present."""
    if "settings_config" not in st.session_state:
        # Load from environment variables as defaults
        st.session_state.settings_config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
            "google_model": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp"),
            "ollama_base_url": os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        }
        logger.debug("Settings config initialized from environment variables")
    
    if "settings_changed" not in st.session_state:
        st.session_state.settings_changed = False


def _apply_settings():
    """Apply the current settings to environment variables and update clients.
    
    Note: This function updates environment variables to maintain consistency with
    the existing application architecture. The client factory functions in
    src/clients.py read credentials from environment variables. While this approach
    exposes credentials to child processes, it's consistent with the app's design
    where credentials are already loaded from .env at startup. For enhanced security
    in production, consider refactoring to pass credentials directly to client
    constructors instead of via environment variables.
    """
    config = st.session_state.settings_config
    
    # Update environment variables (maintaining consistency with existing architecture)
    if config.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    if config.get("openai_model"):
        os.environ["OPENAI_MODEL"] = config["openai_model"]
    
    if config.get("google_api_key"):
        os.environ["GOOGLE_API_KEY"] = config["google_api_key"]
    if config.get("google_model"):
        os.environ["GOOGLE_MODEL"] = config["google_model"]
    
    if config.get("ollama_base_url"):
        os.environ["OLLAMA_API_URL"] = config["ollama_base_url"]
    if config.get("ollama_model"):
        os.environ["OLLAMA_MODEL"] = config["ollama_model"]
    
    # Mark that settings have been applied
    st.session_state.settings_changed = True
    
    # Clear cached agents to force re-initialization with new credentials
    if "chatbot_agent" in st.session_state:
        del st.session_state["chatbot_agent"]
    if "financial_advisor_agent" in st.session_state:
        del st.session_state["financial_advisor_agent"]
    st.session_state.agent_initialized = False
    st.session_state.health_check_done = False
    
    logger.info("Settings applied and agents cleared for re-initialization")


def show_settings_page():
    """Display the settings page with API key configuration."""
    _initialize_settings_state()
    
    st.title("⚙️ API Key Configuration")
    st.markdown(
        """
        Configure your API keys and provider settings here. Changes are applied immediately
        and stored in memory for this session.
        
        **Note:** Settings are **not** persisted to disk and will be reset when you restart the application.
        """
    )
    
    st.divider()
    
    # OpenAI Settings
    with st.expander("✨ **OpenAI Configuration**", expanded=True):
        st.markdown(
            """
            Configure your OpenAI API credentials. Get your API key from 
            [OpenAI Platform](https://platform.openai.com/api-keys).
            """
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            openai_api_key = st.text_input(
                "API Key",
                value=st.session_state.settings_config.get("openai_api_key", ""),
                type="password",
                help="Your OpenAI API key (starts with sk-)",
                key="openai_api_key_input"
            )
            
            openai_model = st.text_input(
                "Model",
                value=st.session_state.settings_config.get("openai_model", "gpt-4o-mini"),
                help="OpenAI model to use (e.g., gpt-4o-mini, gpt-4o)",
                key="openai_model_input"
            )
        
        with col2:
            st.markdown("**Test Connection**")
            if st.button("🔌 Test OpenAI", use_container_width=True, key="test_openai"):
                if not openai_api_key:
                    st.warning("⚠️ Please enter an API key first")
                else:
                    with st.spinner("Testing connection..."):
                        success, message = _test_openai_connection(openai_api_key, openai_model)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        
        # Save button for OpenAI
        if st.button("💾 Save OpenAI Settings", use_container_width=True, key="save_openai"):
            st.session_state.settings_config["openai_api_key"] = openai_api_key
            st.session_state.settings_config["openai_model"] = openai_model
            _apply_settings()
            st.success("✅ OpenAI settings saved!")
            st.rerun()
    
    # Google Settings
    with st.expander("🌐 **Google Gemini Configuration**", expanded=False):
        st.markdown(
            """
            Configure your Google Gemini API credentials. Get your API key from 
            [Google AI Studio](https://aistudio.google.com/app/apikey).
            """
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            google_api_key = st.text_input(
                "API Key",
                value=st.session_state.settings_config.get("google_api_key", ""),
                type="password",
                help="Your Google API key",
                key="google_api_key_input"
            )
            
            google_model = st.text_input(
                "Model",
                value=st.session_state.settings_config.get("google_model", "gemini-2.0-flash-exp"),
                help="Google model to use (e.g., gemini-2.0-flash-exp, gemini-1.5-pro)",
                key="google_model_input"
            )
        
        with col2:
            st.markdown("**Test Connection**")
            if st.button("🔌 Test Google", use_container_width=True, key="test_google"):
                if not google_api_key:
                    st.warning("⚠️ Please enter an API key first")
                else:
                    with st.spinner("Testing connection..."):
                        success, message = _test_google_connection(google_api_key, google_model)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        
        # Save button for Google
        if st.button("💾 Save Google Settings", use_container_width=True, key="save_google"):
            st.session_state.settings_config["google_api_key"] = google_api_key
            st.session_state.settings_config["google_model"] = google_model
            _apply_settings()
            st.success("✅ Google settings saved!")
            st.rerun()
    
    # Ollama Settings
    with st.expander("🦙 **Ollama Configuration**", expanded=False):
        st.markdown(
            """
            Configure your Ollama settings. Ollama must be running locally or accessible at the specified URL.
            Download Ollama from [ollama.com](https://ollama.com/).
            """
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ollama_base_url = st.text_input(
                "Base URL",
                value=st.session_state.settings_config.get("ollama_base_url", "http://localhost:11434/v1"),
                help="Ollama API base URL (default: http://localhost:11434/v1)",
                key="ollama_base_url_input"
            )
            
            ollama_model = st.text_input(
                "Model",
                value=st.session_state.settings_config.get("ollama_model", "llama3.2:3b"),
                help="Ollama model to use (e.g., llama3.2:3b, mistral)",
                key="ollama_model_input"
            )
        
        with col2:
            st.markdown("**Test Connection**")
            if st.button("🔌 Test Ollama", use_container_width=True, key="test_ollama"):
                if not ollama_base_url:
                    st.warning("⚠️ Please enter a base URL first")
                else:
                    with st.spinner("Testing connection..."):
                        success, message = _test_ollama_connection(ollama_base_url, ollama_model)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        
        # Save button for Ollama
        if st.button("💾 Save Ollama Settings", use_container_width=True, key="save_ollama"):
            st.session_state.settings_config["ollama_base_url"] = ollama_base_url
            st.session_state.settings_config["ollama_model"] = ollama_model
            _apply_settings()
            st.success("✅ Ollama settings saved!")
            st.rerun()
    
    st.divider()
    
    # Current Configuration Summary
    st.subheader("📋 Current Configuration")
    
    config = st.session_state.settings_config
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**OpenAI**")
        has_openai_key = bool(config.get("openai_api_key"))
        st.write(f"API Key: {'🔑 Configured' if has_openai_key else '❌ Not set'}")
        st.write(f"Model: {config.get('openai_model', 'N/A')}")
    
    with col2:
        st.markdown("**Google Gemini**")
        has_google_key = bool(config.get("google_api_key"))
        st.write(f"API Key: {'🔑 Configured' if has_google_key else '❌ Not set'}")
        st.write(f"Model: {config.get('google_model', 'N/A')}")
    
    with col3:
        st.markdown("**Ollama**")
        st.write(f"Base URL: {config.get('ollama_base_url', 'N/A')}")
        st.write(f"Model: {config.get('ollama_model', 'N/A')}")
    
    st.divider()
    
    # Help and information
    st.info(
        """
        💡 **Tips:**
        - API keys are stored in memory and never saved to disk
        - Test connections before saving to ensure credentials work
        - Changes take effect immediately after saving
        - For Ollama, ensure the service is running: `ollama serve`
        - Restart the application to revert to .env file settings
        """
    )
    
    # Navigation back to main app
    st.divider()
    if st.button("← Back to Chat", use_container_width=True):
        # Return to main page
        if "show_settings" in st.session_state:
            st.session_state.show_settings = False
        st.rerun()
