# pylint: skip-file

import unittest
from unittest.mock import patch
from llmflows.flows import FlowStep
from llmflows.prompts.prompt_template import PromptTemplate


class TestFlowStep(unittest.TestCase):
    @patch("llmflows.llms.openai.OpenAI")
    def test_generate(self, mock_openai):
        # Instantiate the mock objects
        mock_llm = mock_openai.return_value

        # Mock the generate method of the OpenAI class
        mock_llm.generate.return_value = ("mocked_text", {}, {})

        mock_prompt = PromptTemplate("test {var}")
        mock_callbacks = None
        mock_flowstep = FlowStep(
            "test", mock_llm, mock_prompt, "output_key", mock_callbacks
        )

        # Input dictionary for testing
        test_input_dict = {"var": "value"}

        # Expected output
        expected_output = (
            "mocked_text",
            {"prompt_template": "test {var}", "prompt": "test value"},
            {},
        )

        self.assertEqual(mock_flowstep.generate(test_input_dict), expected_output)
