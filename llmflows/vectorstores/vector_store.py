"""
A module that defines a common interface for vector store services.

This module defines an abstract base class, VectorStore, which specifies
the methods that all vector store services must implement.

Classes that represent specific vector store services (like Pinecone, for example)
should inherit from VectorStore and provide their own implementations for
each of the methods defined in VectorStore.
"""

from abc import ABC, abstractmethod
from llmflows.vectorstores.vector_doc import VectorDoc
from typing import List


class VectorStore(ABC):
    """
    Abstract base class for vector store services.

    This class defines the interface that all vector store services must provide.
    """

    def __init__(self, storage_entity: str, api_key: str, region: str):
        self.storage_entity = storage_entity
        self._api_key = api_key
        self.region = region

    @abstractmethod
    def describe(self) -> None:
        """Describe the index."""
        pass

    @abstractmethod
    def search(self, query: VectorDoc, top_k: int) -> List[dict]:
        """
        Search the index for similar vectors.

        Args:
            query (VectorDoc): The query vector to search for.
            top_k (int): The number of results to return.

        Returns:
            list[dict]: A list of dictionaries representing the search results.
        """
        pass

    @abstractmethod
    def upsert(self, docs: List[VectorDoc]) -> None:
        """Insert or update vectors in the index.

        Args:
            docs (list[VectorDoc]): VectorDoc objects to insert or update.
        """
        pass
