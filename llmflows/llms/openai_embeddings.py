# pylint: disable=too-few-public-methods, W0221

"""
This module helps with creating embeddings form OpenAIs API.
"""

import os
from typing import Union
import openai
from llmflows.vectorstores.vector_doc import VectorDoc
from llmflows.llms.llm_utils import call_with_retry, async_call_with_retry
from .llm import BaseLLM


class OpenAIEmbeddings(BaseLLM):
    """
    A class for interacting with the OpenAI embeddings API.

    Inherits from BaseLLM.

    Uses the specified OpenAI model and parameters for interacting with the OpenAI
    embeddings API. Adds embeddings to a single VectorDoc or a list of VectorDoc
    classes, based on the text in the VectorDoc.

    Args:
        model (str): The name of the OpenAI model to use.

    Attributes:
        _api_key (str): The API key to use for authentication.
        max_retries (int): The maximum number of retries for generating embeddings.
    """

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        max_retries: int = 3,
        api_key: Union[str, None] = None,
    ):
        super().__init__(model)
        self.max_retries = max_retries
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API Key must be provided or set in the OPENAI_API_KEY environment"
                " variable"
            )

    def generate(
        self, docs: Union[VectorDoc, list[VectorDoc]]
    ) -> Union[VectorDoc, list[VectorDoc]]:
        """
        Adds embeddings to a single or list of VectorDocs using OpenAI's service.

        Args:
            docs: A single VectorDoc or a list of VectorDocs to embed.

        Returns:
            If a single VectorDoc was passed, returns it with its embedding field
                updated. If a list of VectorDocs was passed, returns the list with the
                embedding field of each VectorDoc updated.
        """
        single_item = False

        if not isinstance(docs, list):
            docs = [docs]
            single_item = True

        texts = [doc.doc for doc in docs]

        result, _ = call_with_retry(
            api_obj=openai.Embedding,
            engine=self.model,
            input=texts,
            max_retries=self.max_retries,
        )

        for i, doc in enumerate(docs):
            doc.embedding = result["data"][i]["embedding"]

        return docs[0] if single_item else docs

    async def generate_async(
        self, docs: Union[VectorDoc, list[VectorDoc]]
    ) -> Union[VectorDoc, list[VectorDoc]]:
        """
        Async Method that adds embeddings to a single or list of VectorDocs using OpenAI's service.

        Args:
            docs: A single VectorDoc or a list of VectorDocs to embed.

        Returns:
            If a single VectorDoc was passed, returns it with its embedding field
                updated. If a list of VectorDocs was passed, returns the list with the
                embedding field of each VectorDoc updated.
        """
        single_item = False

        if not isinstance(docs, list):  # if a single item was passed
            docs = [docs]
            single_item = True

        texts = [doc.doc for doc in docs]

        result, _ = await async_call_with_retry(
            api_obj=openai.Embedding,
            engine=self.model,
            input=texts,
            max_retries=self.max_retries,
        )

        for i, doc in enumerate(docs):
            doc.embedding = result["data"][i]["embedding"]

        return docs[0] if single_item else docs
