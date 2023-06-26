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
    """

    def __init__(self, model: str):
        """
        Initializes the BaseLLM with a model.

        Args:
            model (str): The model name used in the LLM class.
        """
        self.model = model
