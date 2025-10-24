"""
ChatBot Agent Module.

This module defines the ChatBotAgent class that encapsulates
the datapizza Agent configuration and orchestration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from datapizza.agents import Agent
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

from .clients import get_client, list_providers

# Configure logger
logger = logging.getLogger(__name__)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _log_level, logging.INFO))


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
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the ChatBotAgent.

        Args:
            name: Agent name (default from env: AGENT_NAME)
            system_prompt: System prompt for the agent (default from prompts/system_prompt.md)
            provider: LLM provider ('ollama', 'google', 'openai') (default: 'ollama')
            api_url: API URL (for Ollama) (default from env: OLLAMA_API_URL)
            model: Model name to use (default depends on provider)
        """
        logger.debug("Initializing ChatBotAgent with provider=%s", provider)

        # Load configuration from environment
        self.name = name or os.getenv("AGENT_NAME")
        self.provider = provider or "ollama"
        logger.info("ChatBotAgent name: %s, provider: %s", self.name, self.provider)

        # Load model based on provider
        if model:
            self.model = model
        else:
            # Use provider-specific model environment variable
            if self.provider == "ollama":
                self.model = os.getenv("OLLAMA_MODEL")
            elif self.provider == "google":
                self.model = os.getenv("GOOGLE_MODEL")
            elif self.provider == "openai":
                self.model = os.getenv("OPENAI_MODEL")

        logger.debug("Model selected: %s", self.model)

        # API URL only for Ollama
        self.api_url = api_url or os.getenv("OLLAMA_API_URL")
        if self.provider == "ollama":
            logger.debug("Ollama API URL: %s", self.api_url)

        # Load system prompt from file
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self._load_system_prompt()

        logger.debug(
            "System prompt loaded, length: %d characters", len(self.system_prompt)
        )

        # Load questions from file
        self.questions = self._load_questions()
        logger.debug("Questions loaded, total: %d", len(self.questions))

        # Load prompt templates from files
        self.welcome_prompt = self._load_prompt_template("welcome")
        self.acknowledge_and_ask_prompt = self._load_prompt_template(
            "acknowledge_and_ask"
        )
        self.summary_prompt = self._load_prompt_template("summary")
        logger.debug("Prompt templates loaded successfully")

        # Initialize question tracking
        self.current_question_index = 0
        logger.debug("Question index initialized to 0")

        # Initialize the datapizza client and agent
        logger.debug("Creating client for provider: %s", self.provider)
        self._client = self._create_client()
        logger.debug("Client created successfully")

        logger.debug("Creating agent with name: %s", self.name)
        self._agent = self._create_agent()
        logger.info("ChatBotAgent initialized successfully")

    def _load_system_prompt(self) -> str:
        """
        Load the system prompt from the prompts/system_prompt.md file.

        Returns:
            The system prompt text

        Raises:
            FileNotFoundError: If the prompt file is not found
        """
        logger.debug("Loading system prompt from file")

        # Try to find the prompt file from the current working directory
        prompt_path = Path("prompts/system_prompt.md")

        # If not found, try from the package directory
        if not prompt_path.exists():
            # Try relative to this file
            current_dir = Path(__file__).parent.parent
            prompt_path = current_dir / "prompts" / "system_prompt.md"
            logger.debug(
                "Prompt file not found in current dir, trying: %s", prompt_path
            )

        if not prompt_path.exists():
            logger.error("System prompt file not found at %s", prompt_path)
            raise FileNotFoundError(
                f"System prompt file not found at {prompt_path}. "
                "Please ensure prompts/system_prompt.md exists in the project root."
            )

        logger.debug("Reading system prompt from: %s", prompt_path)
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info("System prompt loaded successfully, size: %d bytes", len(content))
        return content

    def _load_questions(self) -> list:
        """
        Load the questions from the prompts/questions.json file.

        Returns:
            List of question dictionaries with 'id' and 'text' keys

        Raises:
            FileNotFoundError: If the questions file is not found
            json.JSONDecodeError: If the JSON is invalid
        """
        logger.debug("Loading questions from JSON file")

        # Try to find the questions file from the current working directory
        questions_path = Path("prompts/questions.json")

        # If not found, try from the package directory
        if not questions_path.exists():
            # Try relative to this file
            current_dir = Path(__file__).parent.parent
            questions_path = current_dir / "prompts" / "questions.json"
            logger.debug(
                "Questions file not found in current dir, trying: %s", questions_path
            )

        if not questions_path.exists():
            logger.error("Questions file not found at %s", questions_path)
            raise FileNotFoundError(
                f"Questions file not found at {questions_path}. "
                "Please ensure prompts/questions.json exists in the project root."
            )

        logger.debug("Reading questions from: %s", questions_path)
        try:
            with open(questions_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            questions = data.get("questions", [])
            logger.info("Questions loaded successfully, total: %d", len(questions))

            # Validate that each question has 'id' and 'text'
            for question in questions:
                if "id" not in question or "text" not in question:
                    logger.warning(
                        "Question missing 'id' or 'text' field: %s", question
                    )

            return questions

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in questions file: %s", str(e))
            raise
        except Exception as e:
            logger.error("Error loading questions: %s", str(e))
            raise

    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template from the prompts directory.

        Args:
            template_name: Name of the template file (without .md extension)

        Returns:
            The prompt template text

        Raises:
            FileNotFoundError: If the prompt template file is not found
        """
        logger.debug("Loading prompt template: %s", template_name)

        # Try to find the prompt file from the current working directory
        template_path = Path(f"prompts/{template_name}.md")

        # If not found, try from the package directory
        if not template_path.exists():
            current_dir = Path(__file__).parent.parent
            template_path = current_dir / "prompts" / f"{template_name}.md"
            logger.debug(
                "Prompt template not found in current dir, trying: %s", template_path
            )

        if not template_path.exists():
            logger.error("Prompt template file not found at %s", template_path)
            raise FileNotFoundError(
                f"Prompt template file not found at {template_path}. "
                f"Please ensure prompts/{template_name}.md exists in the project root."
            )

        logger.debug("Reading prompt template from: %s", template_path)
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info("Prompt template '%s' loaded successfully", template_name)
        return content

    def _create_client(self):
        """
        Create and configure the LLM client based on the provider.

        Returns:
            Configured client instance for the selected provider
        """
        logger.debug(
            "Creating client for provider: %s with model: %s", self.provider, self.model
        )

        # Build config for the provider
        config = {
            "model": self.model,
            "system_prompt": self.system_prompt,
        }

        # Add provider-specific config
        if self.provider == "ollama":
            config["base_url"] = self.api_url
            logger.debug("Ollama config: base_url=%s", self.api_url)
        # For google and openai, API keys come from environment variables

        try:
            client = get_client(self.provider, config)
            logger.info("Client created successfully for provider: %s", self.provider)
            return client
        except Exception as e:
            logger.error(
                "Failed to create client for provider %s: %s", self.provider, str(e)
            )
            raise

    def _create_agent(self) -> Agent:
        """
        Create and configure the datapizza Agent.

        Returns:
            Configured Agent instance
        """
        logger.debug("Creating Agent with name: %s", self.name)

        self._memory = Memory()
        logger.debug("Memory initialized")

        agent = Agent(
            name=self.name,
            client=self._client,
            # tools=[],
            memory=self._memory,
            planning_interval=5,
        )

        logger.info("Agent created successfully with name: %s", self.name)
        return agent

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
        logger.debug("Processing user message: %s", user_message[:100])

        try:
            # Add user message to memory before processing
            self.add_message_to_memory("user", user_message)
            logger.debug("User message added to memory")

            # Get response from agent
            logger.debug("Running agent with task_input")
            response = self._agent.run(task_input=user_message)
            response_text = response.text
            logger.debug("Agent response received, length: %d", len(response_text))

            # Add assistant response to memory
            self.add_message_to_memory("assistant", response_text)
            logger.debug("Assistant response added to memory")

            logger.info("Chat completed successfully")
            return response_text
        except Exception as e:
            logger.error("Failed to generate response: %s", str(e), exc_info=True)
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
        logger.debug("Starting stream chat for message: %s", user_message[:100])

        try:
            for step in self._agent.stream_invoke(task_input=user_message):
                logger.debug("Stream step received")
                yield step
            logger.info("Stream chat completed successfully")
        except Exception as e:
            logger.error("Failed to stream response: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to stream response: {e}") from e

    def clear_memory(self) -> None:
        """Clear the agent's conversation memory."""
        logger.debug("Clearing agent memory")
        self._memory.clear()
        logger.info("Agent memory cleared")

    def add_message_to_memory(self, role: str, content: str) -> None:
        """
        Add a message to the agent's conversation memory.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content text

        Raises:
            ValueError: If role is not "user" or "assistant"
        """
        logger.debug(
            "Adding message to memory - role: %s, content length: %d",
            role,
            len(content),
        )

        if role not in ["user", "assistant"]:
            logger.error("Invalid role: %s", role)
            raise ValueError(f"Role must be 'user' or 'assistant', got '{role}'")

        # Convert role string to datapizza ROLE enum
        datapizza_role = ROLE.USER if role == "user" else ROLE.ASSISTANT

        # Use datapizza's add_turn method with TextBlock
        self._memory.add_turn(TextBlock(content=content), role=datapizza_role)
        logger.debug("Message added to memory successfully")

    def get_memory_context(self) -> list:
        """
        Get the current conversation memory context.

        Returns:
            List of messages in the agent's memory
        """
        logger.debug("Retrieving memory context")
        messages = self._memory.messages
        logger.debug("Retrieved %d messages from memory", len(messages))
        return messages

    def is_healthy(self) -> bool:
        """
        Check if the agent is healthy and can process requests.

        Returns:
            True if the agent is ready, False otherwise
        """
        logger.debug("Checking agent health")

        try:
            # Test the client connection
            self._client.list_models()
            logger.info("Agent health check passed")
            return True
        except Exception as e:
            logger.warning("Agent health check failed: %s", str(e))
            return False

    def get_config_summary(self) -> dict:
        """
        Get a summary of the agent's configuration.

        Returns:
            Dictionary with configuration details
        """
        logger.debug("Getting configuration summary")

        summary = {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "api_url": self.api_url if self.provider == "ollama" else "N/A",
            "available_providers": list_providers(),
        }

        logger.debug("Configuration summary: %s", summary)
        return summary

    def get_current_question(self) -> str:
        """
        Get the text of the current question to ask the user.

        Returns:
            The current question text or empty string if all questions are asked
        """
        logger.debug("Getting current question, index: %d", self.current_question_index)

        if self.current_question_index < len(self.questions):
            question_obj = self.questions[self.current_question_index]
            # Extract text from question dictionary
            question_text = (
                question_obj.get("text", "")
                if isinstance(question_obj, dict)
                else question_obj
            )
            logger.debug(
                "Current question (id=%s): %s",
                question_obj.get("id") if isinstance(question_obj, dict) else "N/A",
                question_text[:50],
            )
            return question_text
        else:
            logger.debug("All questions have been asked")
            return ""

    def advance_to_next_question(self) -> bool:
        """
        Advance to the next question.

        Returns:
            True if there are more questions, False if all questions have been asked
        """
        logger.debug(
            "Advancing to next question from index: %d", self.current_question_index
        )

        self.current_question_index += 1

        if self.current_question_index < len(self.questions):
            logger.debug("Moved to question index: %d", self.current_question_index)
            return True
        else:
            logger.info("All questions have been asked")
            return False

    def get_questions_progress(self) -> dict:
        """
        Get the progress of questions asked.

        Returns:
            Dictionary with progress information
        """
        logger.debug("Getting questions progress")

        progress = {
            "current_index": self.current_question_index,
            "total_questions": len(self.questions),
            "completed": self.current_question_index >= len(self.questions),
            "progress_percentage": int(
                (self.current_question_index / len(self.questions)) * 100
            )
            if len(self.questions) > 0
            else 0,
        }

        logger.debug("Progress: %s", progress)
        return progress

    def reset_questions(self) -> None:
        """Reset the question index to start from the beginning."""
        logger.debug("Resetting question index")
        self.current_question_index = 0
        logger.info("Question index reset to 0")

    def get_formatted_conversation_history(self) -> str:
        """
        Get the full conversation history formatted for the summary.

        Returns:
            Formatted conversation history as a string
        """
        logger.debug("Formatting conversation history")

        history = []
        messages = self.get_memory_context()

        if not messages:
            logger.debug("No messages in conversation history")
            return "No conversation history available."

        for i, message in enumerate(messages):
            # Extract content from TextBlock if needed
            if hasattr(message, "content"):
                content = message.content
            else:
                content = str(message)

            # Extract role
            if hasattr(message, "role"):
                role = str(message.role).upper()
            else:
                role = "UNKNOWN"

            # Format the message
            if "USER" in role:
                history.append(f"User: {content}")
            elif "ASSISTANT" in role:
                history.append(f"Assistant: {content}")
            else:
                history.append(f"{role}: {content}")

            logger.debug("Formatted message %d: %s...", i, content[:50])

        formatted = "\n\n".join(history)
        logger.info("Conversation history formatted, length: %d", len(formatted))
        return formatted

    def extract_financial_profile(self, conversation_summary: str):
        """
        Extract structured financial profile from conversation summary.

        Uses datapizza's structured_response to extract financial information
        from the conversation summary into a structured FinancialProfile object.

        Args:
            conversation_summary: The conversation summary text to process

        Returns:
            FinancialProfile object with extracted financial information

        Raises:
            Exception: If the extraction fails
        """
        logger.debug("Extracting financial profile from summary")

        try:
            from .financial_profile import FinancialProfile

            # Create the extraction prompt
            extraction_prompt = f"""Extract the financial profile information from the following conversation summary.
If any information is not mentioned or unclear, use reasonable default values based on context clues.

Conversation Summary:
{conversation_summary}

Please extract all available financial information and structure it according to the provided model."""

            logger.debug("Calling structured_response with FinancialProfile model")

            # Use datapizza's structured_response to extract structured data
            response = self._client.structured_response(
                input=extraction_prompt,
                output_cls=FinancialProfile,
            )

            logger.debug("Structured response received")

            # Extract the financial profile from the response
            if hasattr(response, "structured_data") and response.structured_data:
                profile = response.structured_data[0]
                logger.info("Financial profile extracted successfully")
                return profile
            else:
                logger.error("No structured data in response")
                raise ValueError("No structured data returned from extraction")

        except Exception as e:
            logger.error(
                "Failed to extract financial profile: %s", str(e), exc_info=True
            )
            raise RuntimeError(f"Failed to extract financial profile: {e}") from e
