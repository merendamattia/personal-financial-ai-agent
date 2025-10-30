"""
ChatBot Agent Module.

This module defines the ChatBotAgent class that encapsulates
the datapizza Agent configuration and orchestration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from datapizza.agents import Agent
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

from ..clients import get_client, list_providers
from ..models import FinancialProfile, Portfolio
from ..models.tools import FinancialAnalysisResponse
from ..retrieval import RAGAssetRetriever
from ..tools import analyze_financial_asset

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
        self.rag_query_builder_prompt = self._load_prompt_template("rag_query_builder")
        self.portfolio_extraction_prompt = self._load_prompt_template(
            "portfolio_extraction"
        )
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

        # Initialize RAG retriever for asset data
        logger.debug("Initializing RAG asset retriever")
        self._rag_retriever = RAGAssetRetriever()
        try:
            self._rag_retriever.build_or_load_index()
            logger.info("RAG asset retriever initialized successfully")
        except Exception as e:
            logger.warning("Failed to initialize RAG retriever: %s", str(e))
            self._rag_retriever = None

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
            current_dir = Path(__file__).parent.parent.parent
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
            current_dir = Path(__file__).parent.parent.parent
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
            current_dir = Path(__file__).parent.parent.parent
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
            tools=[analyze_financial_asset],
            memory=self._memory,
            planning_interval=5,
        )

        logger.info("Agent created successfully with name: %s", self.name)
        return agent

    def run(self, task_input: str, tool_choice: str = "auto") -> str:
        """
        Process a user message and return the agent's response.

        Args:
            task_input: The user's input message
            tool_choice: Tool selection mode ("auto", "none", "required_first", "required")

        Returns:
            The agent's response text

        Raises:
            Exception: If the agent fails to process the message
        """
        return self._agent.run(task_input=task_input, tool_choice=tool_choice)

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
            # Test the agent by running a simple task
            test_response = self._agent.run(
                task_input="Health check: respond briefly with 'OK'"
            )

            if test_response and hasattr(test_response, "text") and test_response.text:
                logger.info("Agent health check passed")
                return True
            else:
                logger.warning("Agent health check failed: Invalid response")
                return False

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

    def analyze_asset(self, ticker: str, years: int = 10) -> dict:
        """
        Analyze a financial asset and extract structured return data.

        Directly calls the analyze_financial_asset tool and parses the JSON response.
        No need for agent orchestration since the tool does the work directly.

        Args:
            ticker: The stock/ETF ticker symbol (e.g., 'SWDA', 'IX5A')
            years: Number of years to analyze (default: 10)

        Returns:
            dict: Asset analysis data as dictionary with returns, dates, and metrics

        Raises:
            Exception: If the analysis fails
        """
        logger.debug("Analyzing asset %s for %d years", ticker, years)

        try:
            # Call the tool directly - it returns JSON string with all analysis
            logger.debug("Calling analyze_financial_asset tool for %s", ticker)
            result_json = analyze_financial_asset(ticker=ticker, years=years)

            logger.debug("Tool returned JSON, parsing...")

            # Parse the JSON string returned by the tool
            analysis_dict = json.loads(result_json)

            logger.info("Asset analysis completed for %s", ticker)
            logger.debug("Analysis keys: %s", list(analysis_dict.keys()))

            return analysis_dict

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON from tool result for %s: %s", ticker, str(e)
            )
            raise RuntimeError(f"Invalid JSON from tool for {ticker}: {e}") from e
        except Exception as e:
            logger.error(
                "Failed to analyze asset %s: %s", ticker, str(e), exc_info=True
            )
            raise RuntimeError(f"Failed to analyze asset {ticker}: {e}") from e

    def generate_balanced_portfolio(self, financial_profile: dict) -> dict:
        """
        Generate a balanced investment portfolio based on the financial profile.

        Uses RAG to retrieve relevant asset information from ETF PDFs, then generates
        a customized portfolio allocation using the LLM with both the financial profile
        and asset data as context. Returns a structured Portfolio object.

        The method:
        1. Uses RAG retriever to fetch relevant ETF/asset data based on profile
        2. Augments the prompt with financial profile AND retrieved asset context
        3. LLM generates structured portfolio with real asset recommendations

        Args:
            financial_profile: Dictionary with financial profile information

        Returns:
            dict: Portfolio recommendation as dictionary with structured data

        Raises:
            ValueError: If financial_profile is None or empty
            RuntimeError: If LLM fails or RAG retriever unavailable
        """
        if not financial_profile:
            logger.error("Financial profile cannot be None or empty")
            raise ValueError("Financial profile cannot be None or empty")

        logger.info("Generating balanced portfolio with RAG context")
        logger.debug("Profile keys: %s", list(financial_profile.keys()))

        try:
            # Format profile as JSON
            profile_json = json.dumps(financial_profile, indent=2)

            # Use RAG to retrieve relevant asset information
            asset_context = ""
            if self._rag_retriever:
                logger.debug("Retrieving asset information via RAG")

                # Extract key characteristics to find relevant assets
                risk_tolerance = financial_profile.get("risk_tolerance", "Conservative")
                investment_experience = financial_profile.get(
                    "investment_experience", "Beginner"
                )
                goals = financial_profile.get("primary_goals", "Savings")
                time_horizon = financial_profile.get("long_term_goals", "None")
                geographic_allocation = financial_profile.get(
                    "geographic_allocation", "Global balanced"
                )

                # Build a semantic query using the template
                query = self.rag_query_builder_prompt.format(
                    risk_tolerance=risk_tolerance,
                    investment_experience=investment_experience,
                    time_horizon=time_horizon,
                    financial_goals=goals,
                    geographic_allocation=geographic_allocation,
                )

                logger.debug("RAG query: %s", query)

                try:
                    retrieved_assets = self._rag_retriever.retrieve(query, k=15)
                    logger.info("Retrieved %d asset documents", len(retrieved_assets))

                    # Format asset context
                    asset_texts = []
                    for asset in retrieved_assets:
                        asset_texts.append(
                            f"[Asset: {asset['id']} (relevance: {asset['score']:.2f})]\n{asset['text'][:300]}..."
                        )
                    asset_context = "\n---\n".join(asset_texts)
                    logger.debug(
                        "Asset context prepared, length: %d", len(asset_context)
                    )
                except Exception as e:
                    logger.warning("RAG retrieval failed: %s", str(e))
            else:
                logger.warning("RAG retriever not available")

            # Create extraction prompt for portfolio generation using template
            extraction_prompt = self.portfolio_extraction_prompt.format(
                client_profile=profile_json, asset_context=asset_context
            )

            logger.debug("Sending structured portfolio generation request to LLM")
            logger.debug("Prompt length: %d characters", len(extraction_prompt))

            # Use structured_response to get Portfolio object directly
            response = self._client.structured_response(
                input=extraction_prompt,
                output_cls=Portfolio,
            )

            logger.debug("Structured response received")

            # Extract the portfolio from the response
            if hasattr(response, "structured_data") and response.structured_data:
                portfolio = response.structured_data[0]
                logger.info("Portfolio generated successfully with RAG context")
                logger.info("Risk level: %s", portfolio.risk_level)

                # Convert Portfolio object to dictionary with proper enum serialization
                portfolio_dict = portfolio.model_dump(mode="json")
                logger.debug(
                    "Returning portfolio dictionary with keys: %s",
                    list(portfolio_dict.keys()),
                )

                return portfolio_dict
            else:
                logger.error("No structured data in response")
                raise ValueError(
                    "No structured data returned from portfolio generation"
                )

        except Exception as e:
            logger.error("Failed to generate portfolio: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to generate portfolio: {e}") from e
