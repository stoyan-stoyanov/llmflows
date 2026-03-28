# pylint: skip-file

import unittest
from unittest.mock import MagicMock, patch
from llmflows.llms import MiniMaxChat, MessageHistory


class TestMiniMaxChat(unittest.TestCase):
    def setUp(self):
        self.llm = MiniMaxChat(
            model="MiniMax-M2.7",
            temperature=0.7,
            max_tokens=250,
            max_retries=3,
            verbose=False,
            api_key="test_api_key",
        )

    def test_initialization(self):
        self.assertEqual(self.llm.model, "MiniMax-M2.7")
        self.assertEqual(self.llm.temperature, 0.7)
        self.assertEqual(self.llm.max_tokens, 250)
        self.assertEqual(self.llm.max_retries, 3)
        self.assertEqual(self.llm.verbose, False)
        self.assertEqual(self.llm._api_key, "test_api_key")

    def test_initialization_no_api_key(self):
        with self.assertRaises(ValueError):
            MiniMaxChat(api_key="")

    def test_initialization_none_api_key(self):
        with self.assertRaises(ValueError):
            MiniMaxChat(api_key=None)

    def test_default_model(self):
        llm = MiniMaxChat(api_key="test_key")
        self.assertEqual(llm.model, "MiniMax-M2.7")

    def test_temperature_clamping_zero(self):
        llm = MiniMaxChat(api_key="test_key", temperature=0.0)
        self.assertEqual(llm.temperature, 0.01)

    def test_temperature_clamping_negative(self):
        llm = MiniMaxChat(api_key="test_key", temperature=-0.5)
        self.assertEqual(llm.temperature, 0.01)

    def test_temperature_clamping_above_one(self):
        llm = MiniMaxChat(api_key="test_key", temperature=1.5)
        self.assertEqual(llm.temperature, 1.0)

    def test_temperature_clamping_valid(self):
        llm = MiniMaxChat(api_key="test_key", temperature=0.5)
        self.assertEqual(llm.temperature, 0.5)

    def test_temperature_clamping_one(self):
        llm = MiniMaxChat(api_key="test_key", temperature=1.0)
        self.assertEqual(llm.temperature, 1.0)

    def test_format_results(self):
        model_outputs = MagicMock()
        model_outputs.choices = [{"message": {"content": "test_response"}}]
        message_history = MagicMock()
        message_history.max_messages = 10
        message_history.messages = [
            {"role": "user", "content": "test_message"},
        ]
        retries = 0
        text_result, call_data, model_config = self.llm._format_results(
            model_outputs, retries, message_history
        )
        self.assertEqual(text_result, "test_response")
        self.assertEqual(
            call_data,
            {
                "raw_outputs": model_outputs,
                "retries": 0,
            },
        )
        self.assertEqual(
            model_config,
            {
                "model_name": "MiniMax-M2.7",
                "temperature": 0.7,
                "max_tokens": 250,
                "max_messages": 10,
                "messages": [{"role": "user", "content": "test_message"}],
            },
        )

    def test_format_results_with_retries(self):
        model_outputs = MagicMock()
        model_outputs.choices = [{"message": {"content": "retried_response"}}]
        message_history = MagicMock()
        message_history.max_messages = 5
        message_history.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        text_result, call_data, model_config = self.llm._format_results(
            model_outputs, 2, message_history
        )
        self.assertEqual(text_result, "retried_response")
        self.assertEqual(call_data["retries"], 2)
        self.assertEqual(model_config["max_messages"], 5)

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate(self, mock_call_with_retry):
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "test_response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("test message")

        text_result, call_data, model_config = self.llm.generate(mh)

        self.assertEqual(text_result, "test_response")
        self.assertEqual(call_data["retries"], 0)
        self.assertEqual(model_config["model_name"], "MiniMax-M2.7")
        self.assertEqual(model_config["temperature"], 0.7)
        self.assertEqual(model_config["max_tokens"], 250)

        # Verify call_with_retry was called with correct kwargs
        call_kwargs = mock_call_with_retry.call_args
        self.assertEqual(call_kwargs.kwargs["api_base"], "https://api.minimax.io/v1")
        self.assertEqual(call_kwargs.kwargs["api_key"], "test_api_key")
        self.assertEqual(call_kwargs.kwargs["model"], "MiniMax-M2.7")

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate_with_system_prompt(self, mock_call_with_retry):
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "I am a helpful assistant."}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.system_prompt = "You are a helpful assistant."
        mh.add_user_message("Hello!")

        text_result, call_data, model_config = self.llm.generate(mh)

        self.assertEqual(text_result, "I am a helpful assistant.")
        self.assertEqual(len(model_config["messages"]), 2)
        self.assertEqual(model_config["messages"][0]["role"], "system")

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate_multi_turn(self, mock_call_with_retry):
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "final_response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("first message")
        mh.add_ai_message("first reply")
        mh.add_user_message("second message")

        text_result, call_data, model_config = self.llm.generate(mh)

        self.assertEqual(text_result, "final_response")
        self.assertEqual(
            model_config["messages"],
            [
                {"role": "user", "content": "first message"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second message"},
            ],
        )

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate_with_highspeed_model(self, mock_call_with_retry):
        llm = MiniMaxChat(api_key="test_key", model="MiniMax-M2.7-highspeed")
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "fast_response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("test")

        text_result, _, model_config = llm.generate(mh)

        self.assertEqual(text_result, "fast_response")
        self.assertEqual(model_config["model_name"], "MiniMax-M2.7-highspeed")
        call_kwargs = mock_call_with_retry.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "MiniMax-M2.7-highspeed")

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate_clamped_temperature(self, mock_call_with_retry):
        llm = MiniMaxChat(api_key="test_key", temperature=0.0)
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("test")

        _, _, model_config = llm.generate(mh)

        self.assertEqual(model_config["temperature"], 0.01)
        call_kwargs = mock_call_with_retry.call_args
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.01)

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate_return_types(self, mock_call_with_retry):
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "typed_response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("test")

        result = self.llm.generate(mh)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], dict)
        self.assertIsInstance(result[2], dict)

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate_passes_max_retries(self, mock_call_with_retry):
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("test")

        self.llm.generate(mh)

        call_kwargs = mock_call_with_retry.call_args
        self.assertEqual(call_kwargs.kwargs["max_retries"], 3)

    @patch("llmflows.llms.minimax_chat.call_with_retry", autospec=True)
    def test_generate_passes_max_tokens(self, mock_call_with_retry):
        llm = MiniMaxChat(api_key="test_key", max_tokens=1000)
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"content": "response"}}]
        mock_call_with_retry.return_value = (mock_output, 0)

        mh = MessageHistory()
        mh.add_user_message("test")

        llm.generate(mh)

        call_kwargs = mock_call_with_retry.call_args
        self.assertEqual(call_kwargs.kwargs["max_tokens"], 1000)


if __name__ == "__main__":
    unittest.main()
