# pylint: disable=too-few-public-methods, R0913, W0221

"""
This module implements a wrapper for OpenAI completion models, using BaseLLM as a 
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
from .llm import BaseLLM
from .llm_utils import call_with_retry, async_call_with_retry


class OpenAI(BaseLLM):
    """
    A class for interacting with the OpenAI completion API.

    Inherits from BaseLLM.

    Uses the specified OpenAI model and parameters for interacting with the OpenAI API.

    Args:
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        api_key (str): The API key to use for interacting with the OpenAI API.

    Attributes:
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-davinci-003",
        temperature: float = 0.7,
        max_tokens: int = 500,
        max_retries: int = 3,
    ):
        super().__init__(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("You must provide OpenAI API key")
        openai.api_key = self._api_key

    def format_results(self, model_outputs, retries) -> tuple[str, dict, dict]:
        """
        Formats results after generation.

        Args:
            model_outputs: Raw output after model generation.
            retries (int): Number of retries taken for successful generation.

        Returns:
            A tuple containing the generated text, the raw response data, and the
                model configuration.
        """
        text_result = model_outputs.choices[0]["text"]

        call_data = {
            "raw_outputs": model_outputs,
            "retries": retries,
        }

        model_config = {
            "model_name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        return text_result, call_data, model_config

    def generate(self, prompt: str) -> tuple[str, dict, dict]:
        """
        Generates text from a given prompt using OpenAI API.

        Args:
            prompt (str): Text prompt for generation.

        Returns:
            A tuple containing the generated text, the raw response data, and the
                model configuration.
        """

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        completion, retries = call_with_retry(
            func=openai.Completion.create,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
            ),
            max_retries=self.max_retries,
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self.format_results(completion, retries)

    async def generate_async(self, prompt: str) -> tuple[str, dict, dict]:
        """
        Generates text from a given prompt using OpenAI API asynchronously.

        Args:
            prompt (str): Text prompt for generation.

        Returns:
            A tuple containing the generated text, the raw response data, and the
                model configuration.
        """

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        completion, retries = await async_call_with_retry(
            async_func=openai.Completion.acreate,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
            ),
            max_retries=self.max_retries,
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self.format_results(completion, retries)
