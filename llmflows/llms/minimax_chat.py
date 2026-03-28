# pylint: disable=R0913, R0902, R0801, W0221
"""
This module implements a wrapper for MiniMax chat models, using BaseChatLLM as a
base class.

MiniMax provides an OpenAI-compatible API, so this module reuses the openai library
with a custom api_base pointing to https://api.minimax.io/v1.
"""

import openai
from openai.error import (
    APIError,
    Timeout,
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
)
from llmflows.llms.chat_llm import BaseChatLLM
from llmflows.llms.llm_utils import call_with_retry, async_call_with_retry
from llmflows.llms.message_history import MessageHistory


MINIMAX_API_BASE = "https://api.minimax.io/v1"


class MiniMaxChat(BaseChatLLM):
    """
    A class for interacting with the MiniMax chat API.

    Inherits from BaseChatLLM.

    Uses the specified MiniMax model and parameters for interacting with the MiniMax
    chat API via its OpenAI-compatible endpoint, and provides methods to generate
    responses from a conversation history.

    Args:
        api_key (str): The API key to use for interacting with the MiniMax API.
        model (str): The name of the MiniMax model to use.
        temperature (float): The temperature to use for text generation.
            Must be in the range (0.0, 1.0].
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.

    Attributes:
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "MiniMax-M2.7",
        temperature: float = 0.7,
        max_tokens: int = 250,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        super().__init__(model)
        self.temperature = self._clamp_temperature(temperature)
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.verbose = verbose
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("You must provide a MiniMax API key")

    @staticmethod
    def _clamp_temperature(temperature: float) -> float:
        """Clamps temperature to MiniMax's accepted range (0.0, 1.0]."""
        if temperature <= 0.0:
            return 0.01
        if temperature > 1.0:
            return 1.0
        return temperature

    def _format_results(
        self, model_outputs, retries, message_history
    ) -> tuple[str, dict, dict]:
        """
        Prepares results after generation.

        Args:
            model_outputs: Raw output after model generation.
            retries (int): Number of retries taken for successful generation.
            message_history: The message history used for generation.

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
            "max_messages": message_history.max_messages,
            "messages": message_history.messages,
        }

        return text_result, call_data, model_config

    def generate(self, message_history: MessageHistory) -> tuple[str, dict, dict]:
        """
        Sends the messages to the MiniMax chat API and returns a chat message response.

        Args:
            message_history: A ``MessageHistory`` object representing the conversation
                history.

        Returns:
            A tuple containing the generated chat message, raw output data, and model
                configuration.
        """

        completion, retries = call_with_retry(
            func=openai.ChatCompletion.create,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
            ),
            max_retries=self.max_retries,
            api_key=self._api_key,
            api_base=MINIMAX_API_BASE,
            model=self.model,
            messages=message_history.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self._format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config

    async def generate_async(
        self, message_history: MessageHistory
    ) -> tuple[str, dict, dict]:
        """
        Async function that sends the messages to the MiniMax chat API and returns
        a chat message response.

        Args:
            message_history: A ``MessageHistory`` object representing the conversation
                history.

        Returns:
            A tuple containing the generated chat message, raw output data, and model
                configuration.
        """

        completion, retries = await async_call_with_retry(
            async_func=openai.ChatCompletion.acreate,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
            ),
            max_retries=self.max_retries,
            api_key=self._api_key,
            api_base=MINIMAX_API_BASE,
            model=self.model,
            messages=message_history.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self._format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config
