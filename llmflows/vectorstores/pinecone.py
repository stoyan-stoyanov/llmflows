"""
Module to interact with Pinecone, a vector database service.

This module contains a class `Pinecone` which provides several methods to
interact with the Pinecone vector database service.
"""

import pinecone # pylint: disable=import-error
from llmflows.vectorstores.vector_doc import VectorDoc


class Pinecone:
    """
    Interact with Pinecone, a vector database service.

    This class has methods to initialize the Pinecone client, describe the index,
    search the index for similar vectors, and insert or update vectors in the index.

    Args:
        index_name (str): The name of the index to use.
        api_key (str): The API key to use for authentication.
        environment (str): The environment to use, e.g. "production" or "development".
    """

    def __init__(self, index_name: str, api_key: str, environment: str):
        self.index_name = index_name
        self.api_key = api_key
        self.environment = environment
        self._init_client()

    def _init_client(self):
        pinecone.init(api_key=self.api_key, environment=self.environment)
        self.index = pinecone.Index(self.index_name)
        self.describe()

    def describe(self):
        """Describe the index."""
        print(self.index.describe_index_stats())

    def search(self, query: VectorDoc, top_k: int) -> list[dict]:
        """
        Search the index for similar vectors.

        Args:
            query (VectorDoc): The query vector to search for.
            top_k (int): The number of results to return.

        Returns:
            list[dict]: A list of dictionaries representing the search results.
        """
        query_embedding = query.embedding
        search_result = self.index.query(
            query_embedding, top_k=top_k, include_metadata=True
        )
        return search_result["matches"]

    def upsert(self, docs: list[VectorDoc]):
        """Insert or update vectors in the index.

        Args:
            docs (list[VectorDoc]): VectorDoc objects to insert or update.
        """
        to_upsert = []
        for doc in docs:
            doc_id, doc_txt, embeddings, metadata = doc.values
            if "text" not in metadata.keys():
                metadata["text"] = doc_txt
            to_upsert.append((doc_id, embeddings, metadata))
        self.index.upsert(vectors=to_upsert)
