"""
Chatbot Agent Module.

Pure chatbot agent for conversation without financial analysis tools.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ChatbotAgent(BaseAgent):
    """
    Pure Chatbot Agent for conversational interactions.

    No financial tools. Focus on engaging conversation and question gathering.
    """

    def __init__(
        self,
        name: Optional[str] = "ChatbotAgent",
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        planning_interval: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Initialize the ChatbotAgent.

        Args:
            name: Agent name
            system_prompt: System prompt for the agent
            provider: LLM provider ('ollama', 'google', 'openai')
            api_url: API URL (for Ollama)
            model: Model name to use
            planning_interval: Planning interval (default: 2)
            max_steps: Maximum steps (default: 3)
        """
        # Load questions from file before calling parent init
        self.questions = self._load_questions()

        # Initialize question tracking - specific to ChatbotAgent
        self.current_question_index = 0
        logger.debug("Question index initialized to 0")

        # Call parent init
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            provider=provider,
            api_url=api_url,
            model=model,
            planning_interval=planning_interval,
            max_steps=max_steps,
        )

        # Load prompt templates specific to chatbot
        self.welcome_prompt = self._load_prompt_template("welcome")
        self.acknowledge_and_ask_prompt = self._load_prompt_template(
            "acknowledge_and_ask"
        )
        self.summary_prompt = self._load_prompt_template("summary")

        logger.debug("Questions loaded, total: %d", len(self.questions))

    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt - loads from prompts/system_prompt.md

        Returns:
            System prompt text
        """
        return self._load_system_prompt()

    def _get_default_tools(self) -> list:
        """
        Chatbot has NO tools.

        Returns:
            Empty list - no tools for pure chatbot
        """
        logger.debug("Chatbot initialized with NO tools")
        return []

    def _get_default_planning_interval(self) -> int:
        """
        Get default planning interval for ChatbotAgent.

        Returns:
            Planning interval: 2
        """
        return 2

    def _get_default_max_steps(self) -> int:
        """
        Get default max steps for ChatbotAgent.

        Returns:
            Max steps: 3
        """
        return 3

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

        questions_path = Path("prompts/questions.json")

        if not questions_path.exists():
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

    # ==================== Question Management ====================

    def get_current_question(self) -> str:
        """
        Get the text of the current question to ask the user.

        Returns:
            The current question text or empty string if all questions are asked
        """
        logger.debug("Getting current question, index: %d", self.current_question_index)

        if self.current_question_index < len(self.questions):
            question_obj = self.questions[self.current_question_index]
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

    # ==================== Conversation Utilities ====================

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
            if hasattr(message, "content"):
                content = message.content
            else:
                content = str(message)

            if hasattr(message, "role"):
                role = str(message.role).upper()
            else:
                role = "UNKNOWN"

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
