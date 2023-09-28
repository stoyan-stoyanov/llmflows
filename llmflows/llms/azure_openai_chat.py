# pylint: disable=R0913, R0902, R0801, W0221
"""
This module implements a wrapper for Azure OpenAI chat models, using BaseLLM as a 
base class.
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


class AzureOpenAIChat(BaseChatLLM):
    """
    A class for interacting with a Azure OpenAI deployment throught the OpenAI
    Chat API.

    Inherits from BaseLLM.

    Uses the specified OpenAI model and parameters for interacting with the OpenAI chat
    API, and provides methods to add, remove, replace messages, update system prompts,
    and generate responses.

    Args:
        system_prompt (str): The system prompt to use for the chat model.
        engine (str): The name of the OpenAI model to use (following Azure naming).
        azure_openai_endpoint (str): The Azure OpenAI endpoint to use.
        azure_api_version (str): The Azure OpenAI API version to use.
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        api_key (str): The API key to use for interacting with the OpenAI API.

    Attributes:
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.
    """

    def __init__(
        self,
        api_key: str,
        azure_openai_endpoint: str,
        azure_api_version: str = "2023-05-15",
        engine: str = "gpt-35-turbo",
        temperature: float = 0.7,
        max_tokens: int = 250,
        max_retries: int = 3,
    ):
        super().__init__(model=engine)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("You must provide OpenAI API key")

        self._engine = engine
        if not self._deployment_name:
            raise ValueError("You must provide Azure OpenAI deployment name")

        self._azure_openai_endpoint = azure_openai_endpoint
        if not self._azure_openai_endpoint:
            raise ValueError("You must provide Azure OpenAI endpoint")

        openai.api_key = self._api_key
        openai.api_base = self._azure_openai_endpoint
        openai.api_type = "azure"
        openai.api_version = azure_api_version

    def format_results(
        self, model_outputs, retries, message_history
    ) -> tuple[str, dict, dict]:
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
            "max_messages": message_history.max_messages,
            "messages": message_history.messages,
        }

        return text_result, call_data, model_config

    def generate(self, message_history: MessageHistory) -> tuple[str, dict, dict]:
        """
        Sends the messages to the OpenAI chat API and returns a chat message response.

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
            engine=self._engine,
            max_retries=self.max_retries,
            model=self.model,
            messages=message_history.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self.format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config

    async def generate_async(
        self, message_history: MessageHistory
    ) -> tuple[str, dict, dict]:
        """
        Async function that sends the messages to the OpenAI chat API and returns
        a chat message response.

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
            engine=self._engine,
            max_retries=self.max_retries,
            model=self.model,
            messages=message_history.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self.format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config
