# pylint: disable=too-few-public-methods

"""
This is the base module for all LLM (Large Language Model) wrappers.
Each specific LLM should extend this base class.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Base class for all Large Language Models (LLMs). Each specific LLM should extend
    this class.

    Args:
        model (str): The model name used in the LLM class.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str):
        """
        Generates text from the LLM.

        Args:
            prompt: A string representing the prompt to generate text from.

        Returns:
            A string representing the generated text.
        """

    @abstractmethod
    async def generate_async(self, prompt: str):
        """
        Generates text from the LLM asynchronously.

        Args:
            prompt: A string representing the prompt to generate text from.

        Returns:
            A string representing the generated text.
        """
