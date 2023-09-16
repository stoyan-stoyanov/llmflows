# pylint: skip-file

import unittest
from llmflows.llms import MessageHistory


class TestMessageHistory(unittest.TestCase):
    def setUp(self):
        self.message_history = MessageHistory(max_messages=5)

    def test_system_prompt(self):
        self.assertEqual(self.message_history.system_prompt, "")
        self.message_history.system_prompt = "New system prompt"
        self.assertEqual(
            self.message_history.messages[0]["content"], "New system prompt"
        )

    def test_add_message(self):
        self.message_history.add_message("Test user message", "user")
        self.assertEqual(len(self.message_history.messages), 1)
        self.assertEqual(self.message_history.messages[-1]["role"], "user")
        self.assertEqual(
            self.message_history.messages[-1]["content"], "Test user message"
        )

    def test_invalid_message_setter(self):
        with self.assertRaises(ValueError):
            self.message_history.messages = "non_list"

        with self.assertRaises(ValueError):
            self.message_history.messages = ["invalid_message"]

        with self.assertRaises(ValueError):
            self.message_history.messages = [{"role": "invalid role", "content": "123"}]

    def test_validate_message(self):
        with self.assertRaises(ValueError):
            self.message_history.validate_message({"role": "user"})

    def test_remove_message(self):
        self.message_history.add_message("Test user message", "user")
        self.message_history.add_message("Test ai message", "assistant")
        self.message_history.remove_message()
        self.assertEqual(len(self.message_history.messages), 1)

    def test_validate_role(self):
        self.assertEqual(self.message_history.validate_role("user"), "user")
        self.assertEqual(self.message_history.validate_role("system"), "system")
        self.assertEqual(self.message_history.validate_role("assistant"), "assistant")

        with self.assertRaises(ValueError):
            self.message_history.validate_role("invalid_role")

    def test_replace_message(self):
        self.message_history.add_user_message("Random User Message")
        self.message_history.add_ai_message("Random AI Message To be Replaced")
        self.message_history.add_user_message("Another Random User Message")

        self.message_history.replace_message(
            new_role="user", new_message="Replacement message", idx=-2
        )
        self.assertEqual(
            self.message_history.messages[1]["content"], "Replacement message"
        )

        with self.assertRaises(ValueError):
            self.message_history.replace_message(
                new_role="invalid_role", new_message="Replacement message", idx=-2
            )

    def test_message_overflow(self):
        for i in range(6):
            self.message_history.add_message("first_message", "user")

        self.assertEqual(len(self.message_history.messages), 5)
