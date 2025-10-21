"""
ChatBot Agent Module.

This module defines the ChatBotAgent class that encapsulates
the datapizza Agent configuration and orchestration.
"""

import os
from pathlib import Path
from typing import Optional

from datapizza.agents import Agent
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock


class ChatBotAgent:
    """
    Financial AI ChatBot Agent.

    Encapsulates the datapizza Agent configuration and provides
    a simple interface for chat interactions.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the ChatBotAgent.

        Args:
            name: Agent name (default from env: AGENT_NAME)
            system_prompt: System prompt for the agent (default from prompts/system_prompt.md)
            api_url: Ollama API URL (default from env: OLLAMA_API_URL)
            model: Model name to use (default from env: OLLAMA_MODEL)
        """
        # Load configuration from environment
        self.name = name or os.getenv("AGENT_NAME", "PersonalFinancialAIAgent")
        self.api_url = api_url or os.getenv(
            "OLLAMA_API_URL", "http://localhost:11434/v1"
        )
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")

        # Load system prompt from file
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._load_system_prompt()

        # Initialize the datapizza client and agent
        self._client = self._create_client()
        self._agent = self._create_agent()

    def _load_system_prompt(self) -> str:
        """
        Load the system prompt from the prompts/system_prompt.md file.

        Returns:
            The system prompt text

        Raises:
            FileNotFoundError: If the prompt file is not found
        """
        # Try to find the prompt file from the current working directory
        prompt_path = Path("prompts/system_prompt.md")

        # If not found, try from the package directory
        if not prompt_path.exists():
            # Try relative to this file
            current_dir = Path(__file__).parent.parent
            prompt_path = current_dir / "prompts" / "system_prompt.md"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt file not found at {prompt_path}. "
                "Please ensure prompts/system_prompt.md exists in the project root."
            )

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _create_client(self) -> OpenAILikeClient:
        """
        Create and configure the OpenAI-like client for Ollama.

        Returns:
            Configured OpenAILikeClient instance
        """
        return OpenAILikeClient(
            base_url=self.api_url,
            model=self.model,
            api_key="",  # Ollama doesn't require an actual API key
        )

    def _create_agent(self) -> Agent:
        """
        Create and configure the datapizza Agent.

        Returns:
            Configured Agent instance
        """
        self._memory = Memory()
        return Agent(
            name=self.name,
            client=self._client,
            system_prompt=self.system_prompt,
            # tools=[
            #     calculate_monthly_budget,
            #     calculate_compound_interest,
            #     calculate_debt_payoff,
            # ],
            memory=self._memory,
        )

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            user_message: The user's input message

        Returns:
            The agent's response text

        Raises:
            Exception: If the agent fails to process the message
        """
        try:
            # Add user message to memory before processing
            self.add_message_to_memory("user", user_message)

            # Get response from agent
            response = self._agent.run(task_input=user_message)
            response_text = response.text

            # Add assistant response to memory
            self.add_message_to_memory("assistant", response_text)

            return response_text
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def stream_chat(self, user_message: str):
        """
        Stream the agent's response for a user message.

        Yields intermediate steps and the final response.

        Args:
            user_message: The user's input message

        Yields:
            Response text or step information
        """
        try:
            for step in self._agent.stream_invoke(task_input=user_message):
                yield step
        except Exception as e:
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def clear_memory(self) -> None:
        """Clear the agent's conversation memory."""
        self._memory.clear()

    def add_message_to_memory(self, role: str, content: str) -> None:
        """
        Add a message to the agent's conversation memory.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content text

        Raises:
            ValueError: If role is not "user" or "assistant"
        """
        if role not in ["user", "assistant"]:
            raise ValueError(f"Role must be 'user' or 'assistant', got '{role}'")

        # Convert role string to datapizza ROLE enum
        datapizza_role = ROLE.USER if role == "user" else ROLE.ASSISTANT

        # Use datapizza's add_turn method with TextBlock
        self._memory.add_turn(TextBlock(content=content), role=datapizza_role)

    def get_memory_context(self) -> list:
        """
        Get the current conversation memory context.

        Returns:
            List of messages in the agent's memory
        """
        return self._memory.messages

    def is_healthy(self) -> bool:
        """
        Check if the agent is healthy and can process requests.

        Returns:
            True if the agent is ready, False otherwise
        """
        try:
            # Test the client connection
            self._client.list_models()
            return True
        except Exception:
            return False

    def get_config_summary(self) -> dict:
        """
        Get a summary of the agent's configuration.

        Returns:
            Dictionary with configuration details
        """
        return {
            "name": self.name,
            "model": self.model,
            "api_url": self.api_url,
        }
