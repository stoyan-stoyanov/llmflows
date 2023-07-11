# pylint: skip-file

import os
import asyncio
import unittest
from unittest.mock import MagicMock, patch
import openai
from llmflows.llms import OpenAIChat

class TestOpenAIChat(unittest.TestCase):
    def setUp(self):
        self.llm = OpenAIChat(
            model="test_model",
            temperature=0.7,
            max_messages=5,
            max_tokens=250,
            max_retries=3,
            verbose=False,
            api_key="test_api_key"
        )

    def test_initialization(self):
        self.assertEqual(self.llm.messages[0], {"role": "system", "content":""})
        self.assertEqual(self.llm.model, "test_model")
        self.assertEqual(self.llm.temperature, 0.7)
        self.assertEqual(self.llm.max_messages, 5)
        self.assertEqual(self.llm.max_tokens, 250)
        self.assertEqual(self.llm.max_retries, 3)
        self.assertEqual(self.llm.verbose, False)
        self.assertEqual(self.llm._api_key, "test_api_key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_initialization_with_env_api_key(self):
        llm = OpenAIChat(
            model="test_model",
            temperature=0.7,
            max_messages=5,
            max_tokens=250,
            max_retries=3,
            verbose=False,
            api_key=None
        )
        self.assertEqual(llm._api_key, "test_api_key")
    
    @patch('openai.ChatCompletion.create', autospec=True)
    def test_generate(self, mock_openai_chatcompletion_create):
        # Define the mock return value
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"role": "assistant", "content": "test_text"}}]
        mock_openai_chatcompletion_create.return_value = mock_output

        # Call the generate method
        text_result, call_data, config = self.llm.generate()

        # Check that openai.ChatCompletion.create was called with the right arguments
        mock_openai_chatcompletion_create.assert_called_once_with(
            model="test_model",
            messages=self.llm.messages,
            max_tokens=250,
            temperature=0.7
        )

        # Assert that the method returned the expected output
        self.assertEqual(text_result, "test_text")
        self.assertEqual(call_data["raw_outputs"], mock_output)
        self.assertEqual(call_data["retries"], 0)
        self.assertEqual(config["model_name"], "test_model")
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["max_tokens"], 250)
        self.assertEqual(config["max_messages"], 5)
        self.assertEqual(config["messages"], self.llm.messages)

    @patch('openai.ChatCompletion.create', autospec=True)
    async def test_generate_async(self, mock_openai_chatcompletion_create):
        # Define the mock return value
        mock_output = MagicMock()
        mock_output.choices = [{"message": {"role": "assistant", "content": "test_text"}}]
        
        # Mock the openai.ChatCompletion.create method
        mock_openai_chatcompletion_create.return_value = mock_output

        # Call the generate_async method
        text_result, call_data, config = await self.llm.generate_async()

        # Check that openai.ChatCompletion.create was called with the right arguments
        mock_openai_chatcompletion_create.assert_called_once_with(
            model="test_model",
            messages=self.llm.messages,
            max_tokens=250,
            temperature=0.7
        )

        # Assert that the method returned the expected output
        self.assertEqual(text_result, "test_text")
        self.assertEqual(call_data["raw_outputs"], mock_output)
        self.assertEqual(call_data["retries"], 0)
        self.assertEqual(config["model_name"], "test_model")
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["max_tokens"], 250)
        self.assertEqual(config["max_messages"], 5)
        self.assertEqual(config["messages"], self.llm.messages)

    def test_add_message(self):
        self.llm.add_message("test_user_message", "user")
        self.assertEqual(len(self.llm.messages), 2)
        self.assertEqual(self.llm.messages[-1]["role"], "user")
        self.assertEqual(self.llm.messages[-1]["content"], "test_user_message")

    def test_update_system_prompt(self):
        self.llm.update_system_prompt("new_system_prompt")
        self.assertEqual(self.llm.messages[0]["content"], "new_system_prompt")

    def test_validate_message(self):
        with self.assertRaises(ValueError):
            self.llm.validate_message({"role": "user"})
    
    def test_remove_message(self):
        self.llm.add_message("test_user_message", "user")
        self.llm.remove_message()
        self.assertEqual(len(self.llm.messages), 1)
    
    def test_validate_role(self):
        self.assertEqual(self.llm.validate_role("user"), "user")
        self.assertEqual(self.llm.validate_role("system"), "system")
        self.assertEqual(self.llm.validate_role("assistant"), "assistant")
        
        with self.assertRaises(ValueError):
            self.llm.validate_role("invalid_role")

    def test_replace_message(self):
        self.llm.add_message("message_to_be_replaced", "user")
        self.llm.replace_message({"role": "user", "content": "replacement_message"})
        self.assertEqual(self.llm.messages[-2]["content"], "replacement_message")

        with self.assertRaises(ValueError):
            self.llm.replace_message({"role": "invalid_role", "content": "message"})

    def test_invalid_message_setter(self):
        # Test with a non-list
        with self.assertRaises(ValueError):
            self.llm.messages = "non_list"

        # Test with an invalid list (does not contain dict with "role" and "content")
        with self.assertRaises(ValueError):
            self.llm.messages = ["invalid_message"]

    def test_message_overflow(self):
        self.llm = OpenAIChat(
            model="test_model",
            temperature=0.7,
            max_messages=2,
            max_tokens=250,
            max_retries=3,
            verbose=False,
            api_key="test_api_key"
        )
        self.llm.add_message("first_message", "user")
        self.llm.add_message("second_message", "user")
        self.llm.add_message("third_message", "user")

        self.assertEqual(len(self.llm.messages), 2)
        self.assertEqual(self.llm.messages[0]["role"], "system")
        self.assertEqual(self.llm.messages[1]["content"], "third_message")
