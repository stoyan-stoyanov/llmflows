# pylint: disable=too-few-public-methods

"""
This is the base module for all LLM (Large Language Model) wrappers.
Each specific LLM should extend this base class.
"""

from abc import ABC


class BaseLLM(ABC):
    """
    Base class for all Large Language Models (LLMs). Each specific LLM should extend 
    this class.

    Args:
        model (str): The model name used in the LLM class.
    """

    def __init__(self, model: str):
        self.model = model

    def generate(self):
        """
        Generates text from the LLM.

        Raises:
            NotImplementedError: If the LLM does not implement the generate method.
        """
        raise NotImplementedError

    async def generate_async(self):
        """
        Generates text from the LLM asynchronously.

        Raises:
            NotImplementedError: If the LLM does not implement the generate_async method.
        """
        raise NotImplementedError
