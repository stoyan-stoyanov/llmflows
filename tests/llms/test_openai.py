# pylint: skip-file

import os
import asyncio
import unittest
from unittest.mock import MagicMock, patch
import openai
import pytest
from llmflows.llms import OpenAI


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        self.llm = OpenAI(
            model="test_model",
            temperature=0.7,
            max_tokens=500,
            max_retries=3,
            api_key="test_api_key",
        )

    def test_initialization(self):
        self.assertEqual(self.llm.model, "test_model")
        self.assertEqual(self.llm.temperature, 0.7)
        self.assertEqual(self.llm.max_tokens, 500)
        self.assertEqual(self.llm.max_retries, 3)
        self.assertEqual(self.llm._api_key, "test_api_key")

    def test_initialization_no_api_key(self):
        with self.assertRaises(ValueError):
            OpenAI(
                model="test_model",
                temperature=0.7,
                max_tokens=500,
                max_retries=3,
                api_key=None,
            )

    @patch("openai.Completion.create", autospec=True)
    def test_generate(self, mock_openai_completion_create):
        # Define the mock return value
        mock_output = MagicMock()
        mock_output.choices = [{"text": "test_text"}]
        mock_openai_completion_create.return_value = mock_output

        # Call the generate method
        text_result, call_data, config = self.llm.generate("test_prompt")

        # Check that openai.Completion.create was called with the right arguments
        mock_openai_completion_create.assert_called_once_with(
            model="test_model", prompt="test_prompt", max_tokens=500, temperature=0.7
        )

        # Assert that the method returned the expected output
        self.assertEqual(text_result, "test_text")
        self.assertEqual(call_data["raw_outputs"], mock_output)
        self.assertEqual(call_data["retries"], 0)
        self.assertEqual(config["model_name"], "test_model")
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["max_tokens"], 500)

    def test_generate_invalid_prompt(self):
        with self.assertRaises(TypeError):
            self.llm.generate(123)

    def test_prepare_results(self):
        # Mock the model_outputs and retries arguments
        model_outputs = MagicMock()
        model_outputs.choices = [{"text": "test_text"}]
        retries = 0

        # Call the prepare_results method
        text_result, call_data, config = self.llm.prepare_results(
            model_outputs, retries
        )

        # Assert that the method returned the expected output
        self.assertEqual(text_result, "test_text")
        self.assertEqual(call_data["raw_outputs"], model_outputs)
        self.assertEqual(call_data["retries"], 0)
        self.assertEqual(config["model_name"], "test_model")
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["max_tokens"], 500)
