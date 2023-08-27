# pylint: disable=too-few-public-methods

"""
This is the base module for all Chat LLM (Large Language Model) wrappers.
Each specific Chat LLM should extend this base class.
"""

from abc import ABC, abstractmethod
from llmflows.llms.message_history import MessageHistory


class BaseChatLLM(ABC):
    """
    Base class for all Chat Large Language Models. Each specific Chat LLM should extend
    this class.

    Args:
        model (str): The model name used in the LLM class.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, message_history: MessageHistory):
        """
        Generates text from the LLM.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.

        Returns:
            A string representing the generated text.
        """

    @abstractmethod
    async def generate_async(self, message_history: MessageHistory):
        """
        Generates text from the LLM asynchronously.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.

        Returns:
            A string representing the generated text.
        """
