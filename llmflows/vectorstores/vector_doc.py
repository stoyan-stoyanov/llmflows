"""
This module defines the VectorDoc class which is used to represent a document 
with an optional embedding and metadata.
"""

from typing import Union
from uuid import uuid4


class VectorDoc:
    """
    Class representing a document with an optional embedding and metadata.

    Args:
        doc (str): The document text.
        doc_id (str): Unique identifier for the document. If not provided, a 
                      new UUID will be generated.
        metadata (dict, optional): Metadata for the document.
        embedding (list, optional): Embedding for the document.

    Attributes:
        doc_id (str): The unique identifier for the document.
        doc (str): The document text.
        metadata (dict): Optional metadata for the document.
    """

    def __init__(
        self,
        doc: str,
        doc_id: Union[str, None] = None,
        metadata: Union[dict, None] = None,
        embedding: Union[list, None] = None,
    ):
        self.doc_id = doc_id if doc_id is not None else str(uuid4())
        self.doc = doc
        self.metadata = metadata if metadata is not None else {}
        self._embedding = embedding

    @property
    def embedding(self):
        """
        The embedding for the document.

        Raises:
            ValueError: If the embedding for the document has not been set.

        Returns:
            list: The embedding of the document.
        """
        if self._embedding is not None:
            return self._embedding
        raise ValueError(
            "The embedding for this VectorDoc has not been set.\nDoc Id:"
            f" {self.doc_id}\nDoc:{self.doc}"
        )

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def values(self) -> tuple[str, str, list, dict]:
        """
        Returns a tuple of the document ID, document text, embedding, and metadata.

        Raises:
            ValueError: If the embedding for the document has not been set.

        Returns:
            Tuple[str, str, list, dict]: A tuple of the document ID, document text, 
                                         embedding, and metadata.
        """
        if self.embedding is None:
            raise ValueError(
                "The embedding for this VectorDoc has not been set.\nDoc Id:"
                f" {self.doc_id}\nDoc:{self.doc}"
            )
        return self.doc_id, self.doc, self.embedding, self.metadata
