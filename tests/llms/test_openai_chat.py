# pylint: skip-file

import os
import asyncio
import unittest
from unittest.mock import MagicMock, patch
import openai
from llmflows.llms import OpenAIChat, MessageHistory


class TestOpenAIChat(unittest.TestCase):
    def setUp(self):
        self.llm = OpenAIChat(
            model="test_model",
            temperature=0.7,
            max_tokens=250,
            max_retries=3,
            verbose=False,
            api_key="test_api_key",
        )

    def test_initialization(self):
        self.assertEqual(self.llm.model, "test_model")
        self.assertEqual(self.llm.temperature, 0.7)
        self.assertEqual(self.llm.max_tokens, 250)
        self.assertEqual(self.llm.max_retries, 3)
        self.assertEqual(self.llm.verbose, False)
        self.assertEqual(self.llm._api_key, "test_api_key")

    def tes_format_results(self):
        # Mock the model_outputs and retries arguments
        model_outputs = MagicMock()
        model_outputs.choices = [{"message": {"content": "test_text"}}]
        retries = 0

        mh = MessageHistory()

        text_result, call_data, config = self.llm.format_results(
            model_outputs, retries, message_history=mh
        )


        # Assert that the method returned the expected output
        self.assertEqual(text_result, "test_text")
        self.assertEqual(call_data["raw_outputs"], model_outputs)
        self.assertEqual(call_data["retries"], 0)
        self.assertEqual(config["messages"], [])
        self.assertEqual(config["model_name"], "test_model")
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["max_tokens"], 250)

    @patch("openai.ChatCompletion.create", autospec=True)
    def test_generate(self, mock_openai_chatcompletion_create):
        # Define the mock return value
        mock_output = MagicMock()
        mock_output.choices = [
            {"message": {"role": "assistant", "content": "test_text"}}
        ]
        mock_openai_chatcompletion_create.return_value = mock_output

        mh = MessageHistory()
        mh.add_user_message("test message")

        # Call the generate method
        text_result, call_data, config = self.llm.generate(mh)

        # Check that openai.ChatCompletion.create was called with the right arguments
        mock_openai_chatcompletion_create.assert_called_once_with(
            model="test_model", messages=mh.messages, max_tokens=250, temperature=0.7
        )

        # Assert that the method returned the expected output
        self.assertEqual(text_result, "test_text")
        self.assertEqual(call_data["raw_outputs"], mock_output)
        self.assertEqual(call_data["retries"], 0)
        self.assertEqual(config["model_name"], "test_model")
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["max_tokens"], 250)
