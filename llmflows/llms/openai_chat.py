# pylint: disable=R0913, R0902, R0801
"""
This module implements a wrapper for OpenAI chat models, using BaseLLM as a 
base class.
"""

import os
import openai
from .llm import BaseLLM
from .llm_utils import call_with_retry, async_call_with_retry


class OpenAIChat(BaseLLM):
    """
    A class for interacting with the OpenAI chat API.

    Inherits from BaseLLM.

    Uses the specified OpenAI model and parameters for interacting with the OpenAI chat
    API, and provides methods to add, remove, replace messages, update system prompts,
    and generate responses.

    Args:
        system_prompt (str): The system prompt to use for the chat model.
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature to use for text generation.
        max_messages (int): The maximum number of messages to send to the chat API.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.

    Attributes:
        temperature (float): The temperature to use for text generation.
        max_messages (int): The maximum number of messages to send to the chat API.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        messages (list[dict[str, str]]): A list of messages sent to the chat API.
        verbose (bool): Whether to print debug information.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_messages: int = 0,
        max_tokens: int = 250,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        super().__init__(model)
        self._api_key = os.environ["OPENAI_API_KEY"]
        self.temperature = temperature
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.messages = [{"role": "system", "content": system_prompt}]
        self.verbose = verbose

    @property
    def messages(self):
        """
        Returns the conversation history.
        """
        return self._messages

    @messages.setter
    def messages(self, value):
        """
        Sets the conversation history.

        Each message in the list should be a dictionary containing 
        "role" and "content" keys.

        Args:
            value (list): A list of message dictionaries.

        Raises:
            ValueError: If the provided value is not a list or if any 
            dictionary in the list is not a valid message.
        """
        if not isinstance(value, list):
            raise ValueError("messages must be a list of dicts")
        for item in value:
            self.validate_message(item)
        self._messages = value

    def add_message(self, message_str: str, role: str = "user") -> list[dict[str, str]]:
        """
        Appends a new message to the conversation history.

        Args:
            message_str (str): Content of the message.
            role (str, optional): Role in the conversation. Can be "user" or
                "assistant". Defaults to "user".

        Returns:
            Updated conversation history.
        """
        role = self.validate_role(role)

        if self.max_messages and (len(self.messages) >= self.max_messages):
            self.remove_message(idx=1)

        self.messages.append({"role": role, "content": message_str})
        return self.messages

    @staticmethod
    def validate_role(role: str) -> str:
        """
        Validates the role of a message.

        Args:
            role (str): The role of the message (either "user", "assistant", 
                or "system").

        Returns:
            str: The validated role.

        Raises:
            ValueError: If the role is "system", as this should be updated using 
                'update_system_prompt' method.
            ValueError: If the role is not "user" or "assistant".
        """
        # if role == "system":
        #     raise ValueError(
        #       "To update the system prompt use the 'update_system_prompt method"
        # )

        if role not in ["user", "system", "assistant"]:
            raise ValueError(
                "The role should be either 'system', 'user' or 'assistant'"
            )

        return role

    def validate_message(self, message: dict[str, str]) -> dict[str, str]:
        """
        Validates a message for required fields.

        Args:
            message (dict[str, str]): The message to validate.

        Returns:
            The validated message.

        Raises:
            ValueError: If the provided message does not contain the necessary fields 
                ("role" and "content").
        """
        if {"role", "content"}.issubset(message):
            self.validate_role(message["role"])
            return message
        raise ValueError("The provided message is not a valid message.")

    def replace_message(self, new_message, idx=-2):
        """
        Replaces a message in the list of messages sent to the chat API.

        Args:
            new_message: The new message to replace the old message with.
            idx (int): The index of the message to replace.
        """
        new_message = self.validate_message(new_message)
        self.messages[idx] = new_message

    def remove_message(self, idx=-1):
        """
        Removes a message from the list of messages sent to the chat API.

        Args:
            idx (int): The index of the message to remove.
        """
        self.messages.pop(idx)

    def perepare_results(self, model_outputs, retries) -> tuple[str, dict, dict]:
        """
        Prepares results after generation.

        Args:
            model_outputs: Raw output after model generation.
            retries (int): Number of retries taken for successful generation.

        Returns:
            tuple(str, dict, dict): Formatted output text, raw outputs, and model
                configuration.
        """
        text_result = model_outputs.choices[0]["message"]["content"]

        call_data = {
            "raw_outputs": model_outputs,
            "retries": retries,
        }

        model_config = {
            "model_name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_messages": self.max_messages,
            "messages": self.messages,
        }

        return text_result, call_data, model_config

    def update_system_prompt(self, new_prompt: str):
        """
        Updates the system prompt sent to the chat API.

        Args:
            new_prompt (str): The new system prompt.
        """
        self.messages[0] = {"role": "system", "content": new_prompt}

    def generate(self) -> tuple[str, dict, dict]:
        """
        Sends the messages to the OpenAI chat API and returns a chat message response.

        Returns:
            A tuple containing the generated chat message, raw output data, and model
                configuration.
        """

        completion, retries = call_with_retry(
            api_obj=openai.ChatCompletion,
            max_retries=self.max_retries,
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self.perepare_results(
            model_outputs=completion, retries=retries
        )

        self.add_message(message_str=str_message, role="assistant")

        return str_message, call_data, model_config

    async def generate_async(self) -> tuple[str, dict, dict]:
        """
        Async function that sends the messages to the OpenAI chat API and returns
        a chat message response.

        Returns:
            A tuple containing the generated chat message, raw output data, and model
                configuration.
        """

        completion, retries = await async_call_with_retry(
            api_obj=openai.ChatCompletion,
            max_retries=self.max_retries,
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self.perepare_results(
            model_outputs=completion, retries=retries
        )

        self.add_message(message_str=str_message, role="assistant")

        return str_message, call_data, model_config
