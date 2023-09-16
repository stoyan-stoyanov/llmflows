# pylint: skip-file

import unittest
from unittest.mock import MagicMock, patch
from llmflows.llms import ClaudeChat, MessageHistory


class TestClaudeChat(unittest.TestCase):
    def setUp(self):
        self.llm = ClaudeChat(
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

    def test_format_results(self):
        model_outputs = MagicMock()
        model_outputs.completion = "test_completion"
        model_outputs.model = "test_model"
        model_outputs.stop_reason = "test_stop_reason"
        model_outputs.stop = "test_stop"
        model_outputs.log_id = "test_log_id"
        message_history = MagicMock()
        message_history.max_messages = 10
        message_history.messages = ["test_message_1", "test_message_2"]
        retries = 3
        text_result, call_data, model_config = self.llm._format_results(
            model_outputs, retries, message_history
        )
        self.assertEqual(text_result, "test_completion")
        self.assertEqual(
            call_data,
            {
                "raw_outputs": {
                    "completion": "test_completion",
                    "model": "test_model",
                    "stop_reason": "test_stop_reason",
                    "stop": "test_stop",
                    "log_id": "test_log_id",
                },
                "retries": 3,
            },
        )
        self.assertEqual(
            model_config,
            {
                "model_name": "test_model",
                "temperature": 0.7,
                "max_tokens": 250,
                "max_messages": 10,
                "messages": ["test_message_1", "test_message_2"],
            },
        )

    def test_convert_message_history(self):
        mh = MessageHistory()
        mh.add_user_message("test_message_1")
        mh.add_ai_message("test_message_2")
        mh.add_user_message("test_message_3")
        converted_message_history = self.llm._convert_message_history(mh)

        self.assertEqual(
            converted_message_history,
            (
                "\n\nHuman: test_message_1\n\nAssistant: test_message_2\n\nHuman: "
                "test_message_3\n\nAssistant:"
            ),
        )
    
    def test_convert_message_history_last_message_error(self):
        mh = MessageHistory()
        mh.add_user_message("test_message_1")
        mh.add_ai_message("test_message_2")

        # assert ValueError: Last message in message history must be from the user.
        with self.assertRaises(ValueError):
            self.llm._convert_message_history(mh)

    @patch("llmflows.llms.claude_chat.call_with_retry", autospec=True)
    def test_generate(self, mock_call_with_retry):
        # Define the mock return value
        mock_output = MagicMock()
        mock_output.completion = "test_completion"
        mock_output.model = "test_model"
        mock_output.stop_reason = "test_stop_reason"
        mock_output.stop = "test_stop"
        mock_output.log_id = "test_log_id"
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("test message")
        mh.add_ai_message("test message")
        mh.add_user_message("test message")

        # Call the generate method
        text_result, call_data, model_config = self.llm.generate(mh)

        # Assert that the method returned the expected output
        self.assertEqual(text_result, "test_completion")
        self.assertEqual(
            call_data,
            {
                "raw_outputs": {
                    "completion": "test_completion",
                    "model": "test_model",
                    "stop_reason": "test_stop_reason",
                    "stop": "test_stop",
                    "log_id": "test_log_id",
                },
                "retries": 0,
            },
        )
        self.assertEqual(
            model_config,
            {
                "model_name": "test_model",
                "temperature": 0.7,
                "max_tokens": 250,
                "max_messages": 0,
                "messages": [
                    {"role": "user", "content": "test message"},
                    {"role": "assistant", "content": "test message"},
                    {"role": "user", "content": "test message"},
                ],
            },
        )
