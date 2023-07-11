# pylint: skip-file

import os
import asyncio
import unittest
from unittest.mock import MagicMock, patch
from llmflows.llms import OpenAIEmbeddings
from llmflows.vectorstores.vector_doc import VectorDoc


class TestOpenAIEmbeddings(unittest.TestCase):
    def setUp(self):
        self.llm = OpenAIEmbeddings(model="test_model", max_retries=3)

    def test_initialization(self):
        self.assertEqual(self.llm.model, "test_model")
        self.assertEqual(self.llm.max_retries, 3)

    @patch('openai.Embedding.create', autospec=True)
    def test_generate_single_doc(self, mock_openai_embedding_create):
        # Define the mock return value
        mock_output = {"data": [{"embedding": "test_embedding"}]}
        mock_openai_embedding_create.return_value = mock_output

        # Prepare a VectorDoc
        doc = VectorDoc(doc="test_doc")

        # Call the generate method
        result_doc = self.llm.generate(doc)

        # Check that openai.Embedding.create was called with the right arguments
        mock_openai_embedding_create.assert_called_once_with(
            engine="test_model",
            input=["test_doc"],
        )

        # Assert that the method returned the expected output
        self.assertEqual(result_doc.embedding, "test_embedding")

    @patch('openai.Embedding.create', autospec=True)
    def test_generate_multiple_docs(self, mock_openai_embedding_create):
        # Define the mock return value
        mock_output = {"data": [{"embedding": "test_embedding_1"}, {"embedding": "test_embedding_2"}]}
        mock_openai_embedding_create.return_value = mock_output

        # Prepare VectorDocs
        docs = [VectorDoc(doc="test_doc_1"), VectorDoc(doc="test_doc_2")]

        # Call the generate method
        result_docs = self.llm.generate(docs)

        # Check that openai.Embedding.create was called with the right arguments
        mock_openai_embedding_create.assert_called_once_with(
            engine="test_model",
            input=["test_doc_1", "test_doc_2"],
        )

        # Assert that the method returned the expected output
        self.assertEqual(result_docs[0].embedding, "test_embedding_1")
        self.assertEqual(result_docs[1].embedding, "test_embedding_2")

    @patch('openai.Embedding.create', autospec=True)
    async def test_generate_async_single_doc(self, mock_openai_embedding_create):
        # Define the mock return value
        mock_output = {"data": [{"embedding": "test_embedding"}]}
        mock_openai_embedding_create.return_value = mock_output

        # Prepare a VectorDoc
        doc = VectorDoc(doc="test_doc")

        # Call the generate_async method
        result_doc = await self.llm.generate_async(doc)

        # Check that openai.Embedding.create was called with the right arguments
        mock_openai_embedding_create.assert_called_once_with(
            engine="test_model",
            input=["test_doc"],
        )

        # Assert that the method returned the expected output
        self.assertEqual(result_doc.embedding, "test_embedding")

    @patch('openai.Embedding.create', autospec=True)
    async def test_generate_async_multiple_docs(self, mock_openai_embedding_create):
        # Define the mock return value
        mock_output = {"data": [{"embedding": "test_embedding_1"}, {"embedding": "test_embedding_2"}]}
        mock_openai_embedding_create.return_value = mock_output

        # Prepare VectorDocs
        docs = [VectorDoc(doc="test_doc_1"), VectorDoc(doc="test_doc_2")]

        # Call the generate_async method
        result_docs = await self.llm.generate_async(docs)

        # Check that openai.Embedding.create was called with the right arguments
        mock_openai_embedding_create.assert_called_once_with(
            engine="test_model",
            input=["test_doc_1", "test_doc_2"],
        )

        # Assert that the method returned the expected output
        self.assertEqual(result_docs[0].embedding, "test_embedding_1")
        self.assertEqual(result_docs[1].embedding, "test_embedding_2")
