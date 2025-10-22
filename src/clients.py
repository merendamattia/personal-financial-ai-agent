"""Client registry and factory for LLM clients.

This module exposes a simple registry where client factory callables can be
registered under a provider key (e.g. 'google', 'openai', 'ollama').
The `get_client` function returns a ready-to-use client instance given the
provider name and configuration.

The registry holds callables with signature: Callable[[dict], Any]
which should accept a dict of keyword configuration values and return an
instance implementing the expected datapizza client interface.
"""

import logging
import os
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _log_level, logging.INFO))

# Registry mapping provider key -> factory callable
_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register(provider: str, factory: Callable[[Dict[str, Any]], Any]) -> None:
    """Register a factory for a provider key.

    If a provider is registered more than once, the latest registration wins.

    Args:
        provider: The provider key (e.g. 'google', 'openai', 'ollama')
        factory: A callable that takes a config dict and returns a client instance
    """
    logger.debug("Registering client provider '%s'", provider)
    _REGISTRY[provider] = factory
    logger.info("Client provider '%s' registered successfully", provider)


def get_client(provider: str, config: Dict[str, Any] | None = None) -> Any:
    """Return a client instance for the given provider.

    Args:
        provider: The provider key (e.g. 'google', 'openai', 'ollama').
        config: Optional dict of configuration (api_key, model, system_prompt, etc.).

    Returns:
        A configured client instance

    Raises:
        KeyError: if the provider is unknown.
        RuntimeError: if the selected provider's factory raises an error.
    """
    logger.debug("Getting client for provider: %s", provider)

    cfg = dict(config or {})
    try:
        factory = _REGISTRY[provider]
        logger.debug("Factory found for provider: %s", provider)
    except KeyError:
        logger.error(
            "Unknown client provider: '%s'. Available: %s",
            provider,
            list(_REGISTRY.keys()),
        )
        raise KeyError(
            f"Unknown client provider: '{provider}'. Available: {list(_REGISTRY.keys())}"
        )

    try:
        logger.debug("Creating client instance for provider: %s", provider)
        client = factory(cfg)
        logger.info("Client instance created successfully for provider: %s", provider)
        return client
    except Exception as exc:
        logger.error(
            "Failed to instantiate client for provider '%s': %s",
            provider,
            str(exc),
            exc_info=True,
        )
        raise RuntimeError(
            f"Failed to instantiate client for provider '{provider}': {exc}"
        ) from exc


def list_providers() -> list:
    """Return a list of available provider keys.

    Returns:
        List of registered provider names
    """
    logger.debug("Listing available providers")
    providers = list(_REGISTRY.keys())
    logger.debug("Available providers: %s", providers)
    return providers


# Register built-in providers at import time.
def _register_builtin_providers() -> None:
    """Register all built-in client providers at import time.

    This function defines and registers factory functions for supported providers:
    - google: GoogleClient for Google Generative AI (Gemini)
    - openai: OpenAIClient for OpenAI API (GPT models)
    - ollama: OpenAILikeClient for local Ollama models

    Lazy imports are used inside factories to avoid raising ImportError at import
    time if optional packages aren't installed.
    """
    logger.debug("Starting registration of built-in providers")

    def google_factory(cfg: Dict[str, Any]) -> Any:
        """Create and return a GoogleClient instance for Gemini.

        Args:
            cfg: Configuration dict with optional 'api_key', 'model', and 'system_prompt' keys.
                 Falls back to GOOGLE_API_KEY and GOOGLE_MODEL environment variables.

        Returns:
            A configured GoogleClient instance for Google Generative AI.

        Raises:
            RuntimeError: If the datapizza.clients.google module cannot be imported.
        """
        logger.debug("Google factory called")

        try:
            logger.debug("Importing GoogleClient")
            from datapizza.clients.google import GoogleClient
        except ImportError as exc:
            logger.error("Failed to import GoogleClient: %s", str(exc))
            raise RuntimeError(
                "Google client support not installed. "
                "Please install datapizza-ai with Google support."
            ) from exc

        api_key = cfg.get("api_key") or os.getenv("GOOGLE_API_KEY")
        model = cfg.get("model") or os.getenv("GOOGLE_MODEL")
        system_prompt = cfg.get("system_prompt")

        logger.debug("Google config - model: %s, has_api_key: %s", model, bool(api_key))

        if not api_key:
            logger.error("Google API key not provided")
            raise RuntimeError(
                "Google API key not provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key in config."
            )

        logger.debug("Creating GoogleClient instance")
        return GoogleClient(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=0.5,
        )

    def openai_factory(cfg: Dict[str, Any]) -> Any:
        """Create and return an OpenAIClient instance for GPT models.

        Args:
            cfg: Configuration dict with optional 'api_key', 'model', and 'system_prompt' keys.
                 Falls back to OPENAI_API_KEY and OPENAI_MODEL environment variables.

        Returns:
            A configured OpenAIClient instance for OpenAI API.

        Raises:
            RuntimeError: If the datapizza.clients.openai module is not installed.
        """
        logger.debug("OpenAI factory called")

        try:
            logger.debug("Importing OpenAIClient")
            from datapizza.clients.openai import OpenAIClient
        except ImportError as exc:
            logger.error("Failed to import OpenAIClient: %s", str(exc))
            raise RuntimeError(
                "OpenAI client support not installed. "
                "Please install datapizza-ai with OpenAI support."
            ) from exc

        api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
        model = cfg.get("model") or os.getenv("OPENAI_MODEL")
        system_prompt = cfg.get("system_prompt")

        logger.debug("OpenAI config - model: %s, has_api_key: %s", model, bool(api_key))

        if not api_key:
            logger.error("OpenAI API key not provided")
            raise RuntimeError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        logger.debug("Creating OpenAIClient instance")
        return OpenAIClient(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=0.5,
        )

    def ollama_factory(cfg: Dict[str, Any]) -> Any:
        """Create and return an OpenAILikeClient instance for Ollama.

        Args:
            cfg: Configuration dict with optional 'model', 'base_url', and 'system_prompt' keys.
                 Falls back to OLLAMA_MODEL and OLLAMA_API_URL environment variables.
                 api_key is always empty as Ollama doesn't require authentication.

        Returns:
            A configured OpenAILikeClient instance for local Ollama models.

        Raises:
            RuntimeError: If the datapizza.clients.openai_like module is not installed.
        """
        logger.debug("Ollama factory called")

        try:
            logger.debug("Importing OpenAILikeClient")
            from datapizza.clients.openai_like import OpenAILikeClient
        except ImportError as exc:
            logger.error("Failed to import OpenAILikeClient: %s", str(exc))
            raise RuntimeError(
                "Ollama client support not installed. "
                "Please install datapizza-ai with OpenAI-like support."
            ) from exc

        model = cfg.get("model") or os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
        base_url = cfg.get("base_url") or os.getenv(
            "OLLAMA_API_URL", "http://localhost:11434/v1"
        )
        system_prompt = cfg.get("system_prompt")

        logger.debug("Ollama config - model: %s, base_url: %s", model, base_url)

        logger.debug("Creating OpenAILikeClient instance for Ollama")
        return OpenAILikeClient(
            api_key="",
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            temperature=0.5,
        )

    logger.debug("Registering Google factory")
    register("google", google_factory)
    logger.debug("Registering OpenAI factory")
    register("openai", openai_factory)
    logger.debug("Registering Ollama factory")
    register("ollama", ollama_factory)

    logger.info("All built-in providers registered successfully")


logger.debug("Calling _register_builtin_providers at import time")
_register_builtin_providers()
logger.info("Built-in provider registration completed")
