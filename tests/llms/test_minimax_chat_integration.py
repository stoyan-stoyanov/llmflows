# pylint: skip-file

"""
Integration tests for MiniMaxChat.

These tests verify that MiniMaxChat integrates correctly with the broader
llmflows framework (MessageHistory, FlowSteps, etc.) without making real
API calls. All external API calls are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch
from llmflows.llms import MiniMaxChat, MessageHistory


class TestMiniMaxChatIntegration(unittest.TestCase):
    """Integration tests for MiniMaxChat with the llmflows framework."""

    def setUp(self):
        self.llm = MiniMaxChat(
            api_key="test_integration_key",
            model="MiniMax-M2.7",
            temperature=0.7,
            max_tokens=500,
        )

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_multi_turn_conversation(self, mock_call_with_retry):
        """Test a full multi-turn conversation flow using MessageHistory."""
        mh = MessageHistory()
        mh.system_prompt = "You are a helpful coding assistant."

        # Turn 1
        mh.add_user_message("What is Python?")
        mock_output_1 = MagicMock()
        mock_output_1.choices = [
            {"message": {"content": "Python is a programming language."}}
        ]
        mock_call_with_retry.return_value = (mock_output_1, 0)

        response_1, _, config_1 = self.llm.generate(mh)
        mh.add_ai_message(response_1)

        self.assertEqual(response_1, "Python is a programming language.")
        self.assertEqual(len(mh.messages), 3)  # system + user + assistant

        # Turn 2
        mh.add_user_message("What are its main features?")
        mock_output_2 = MagicMock()
        mock_output_2.choices = [
            {"message": {"content": "Dynamic typing, garbage collection, and more."}}
        ]
        mock_call_with_retry.return_value = (mock_output_2, 0)

        response_2, _, config_2 = self.llm.generate(mh)
        mh.add_ai_message(response_2)

        self.assertEqual(
            response_2, "Dynamic typing, garbage collection, and more."
        )
        self.assertEqual(len(mh.messages), 5)
        self.assertEqual(config_2["model_name"], "MiniMax-M2.7")

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_message_history_max_messages(self, mock_call_with_retry):
        """Test that MessageHistory respects max_messages with MiniMaxChat."""
        mh = MessageHistory(max_messages=4)
        mh.system_prompt = "You are helpful."

        # Add enough messages to trigger trimming
        mh.add_user_message("msg 1")
        mh.add_ai_message("reply 1")
        mh.add_user_message("msg 2")

        # max_messages=4 means system + 3 messages max; adding one more trims oldest
        self.assertEqual(len(mh.messages), 4)

        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "latest reply"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        text_result, _, model_config = self.llm.generate(mh)
        self.assertEqual(text_result, "latest reply")
        self.assertEqual(model_config["max_messages"], 4)

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_model_config_completeness(self, mock_call_with_retry):
        """Test that generate returns complete model_config for tracing."""
        mh = MessageHistory()
        mh.add_user_message("test")

        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        _, call_data, model_config = self.llm.generate(mh)

        # Verify all expected keys are present for observability
        self.assertIn("model_name", model_config)
        self.assertIn("temperature", model_config)
        self.assertIn("max_tokens", model_config)
        self.assertIn("max_messages", model_config)
        self.assertIn("messages", model_config)
        self.assertIn("raw_outputs", call_data)
        self.assertIn("retries", call_data)


if __name__ == "__main__":
    unittest.main()
