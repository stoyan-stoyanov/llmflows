# pylint: disable=too-few-public-methods, R0913, W0221, R0801

"""
This module implements a wrapper for Azure OpenAI completion models, using BaseLLM as a 
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


class AzureOpenAI(BaseLLM):
    """
    A class for interacting with a Azure OpenAI deployment throught the OpenAI
    Completions API.

    Inherits from BaseLLM.

    Uses the specified OpenAI model and parameters for interacting with the OpenAI API.

    Args:
        deployment_name (str): The name of the Azure deployment to use.
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
    """

    def __init__(
        self,
        api_key: str,
        deployment_name: str,
        azure_openai_endpoint: str,
        azure_api_version: str = "2023-05-15",
        temperature: float = 0.7,
        max_tokens: int = 500,
        max_retries: int = 3,
    ):
        super().__init__(model=deployment_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self._api_key = api_key
        if not self._api_key:
            raise ValueError("You must provide OpenAI API key")

        self._engine = deployment_name
        if not self._engine:
            raise ValueError("You must provide Azure OpenAI deployment name")

        self._azure_openai_endpoint = azure_openai_endpoint
        if not self._azure_openai_endpoint:
            raise ValueError("You must provide Azure OpenAI endpoint")

        openai.api_key = self._api_key
        openai.api_base = self._azure_openai_endpoint
        openai.api_type = "azure"
        openai.api_version = azure_api_version

    def _format_results(self, model_outputs, retries) -> tuple[str, dict, dict]:
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
            engine=self._engine,
            max_retries=self.max_retries,
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self._format_results(completion, retries)

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
            engine=self._engine,
            max_retries=self.max_retries,
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self._format_results(completion, retries)
