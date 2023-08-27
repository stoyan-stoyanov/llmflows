# pylint: disable=too-few-public-methods

"""
This is the base module for all Embeddings model wrappers.
Each specific Embedding class should extend this base class.
"""

from abc import ABC, abstractmethod
from typing import Union
from llmflows.vectorstores.vector_doc import VectorDoc


class BaseEmbeddings(ABC):
    """
    Base class for all Embeddings models. Each specific Embedding class should extend
    this class.

    Args:
        model (str): The model name used to generate the embeddings.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, docs: Union[VectorDoc, list[VectorDoc]]):
        """
        Generates embeddings based on a VectorDoc or list of VectorDocs.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.

        Returns:
            A string representing the generated text.
        """

    @abstractmethod
    async def generate_async(self, docs: Union[VectorDoc, list[VectorDoc]]):
        """
        Generates embeddings asynchronously based on a VectorDoc or list of VectorDocs.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.

        Returns:
            A string representing the generated text.
        """
