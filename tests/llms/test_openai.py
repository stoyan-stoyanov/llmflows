# pylint: skip-file

import unittest
from unittest.mock import MagicMock

from llmflows.llms import OpenAI


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        self.llm = OpenAI(
            model="test_model",
            temperature=0.7,
            max_tokens=500,
            max_retries=3,
            api_key="test_api_key"
        )

    def test_generate(self):
        # Mock the generate method
        self.llm.generate = MagicMock(return_value="test_output")

        # Call the generate method
        output = self.llm.generate()

        # Assert that the generate method was called and returned the expected output
        self.llm.generate.assert_called_once()
        self.assertEqual(output, "test_output")

    def test_generate_async(self):
        # Mock the generate_async method
        self.llm.generate_async = MagicMock(return_value="test_output")

        # Call the generate_async method
        output = self.llm.generate_async()

        # Assert the generate_async method was called and returned the expected output
        self.llm.generate_async.assert_called_once()
        self.assertEqual(output, "test_output")

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
