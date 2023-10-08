# pylint: disable=too-few-public-methods, R0913, W0221, R0801

"""
This module implments a wrapper for the Google PaLM model API, using BaseLLM as a
base class.
"""

import google.generativeai as palm
from .llm import BaseLLM
from .llm_utils import call_with_retry


class PaLM(BaseLLM):
    """
    A class for interacting with the Google PaLM API.

    Inherits from BaseLLM.

    Uses the specified Google PaLM model and parameters for interacting with the Google
    PaLM API.

    Args:
        model (str): The name of the Google PaLM model to use.
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        api_key (str): The API key to use for interacting with the Google PaLM API.
    
    Attributes:
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-bison-001",
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
            raise ValueError("You must provide Google API key")
        palm.configure(api_key=self._api_key)

    def _format_results(self, completion, retries) -> tuple[str, dict, dict]:
        """
        Formats results after generation.

        Args:
            completion: The completion response from the API.
            retries: The number of retries.

        Returns:
            A tuple containing the generated text, the response from the API, and the
                model configuration.
        """
        text_result = completion.results

        call_data = {"raw_outputs": completion.candidates, "retries": retries}

        model_config = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        return text_result, call_data, model_config

    def generate(self, prompt: str) -> tuple[str, dict, dict]:
        """
        Generates text from the Google PaLM API.

        Args:
            prompt (str): The prompt to use for generating text.

        Returns:
            A tuple containing the generated text, the response from the API, and the
                model configuration.
        """

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        completion, retries = call_with_retry(
            func=palm.generate_text,
            exceptions_to_retry=(),
            max_retries=self.max_retries,
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self._format_results(completion, retries)

    async def generate_async(self, prompt: str) -> tuple[str, dict, dict]:
        """
        Generates text from the Google PaLM API asynchronously.

        Args:
            prompt (str): The prompt to use for generating text.

        Returns:
            A tuple containing the generated text, the response from the API, and the
                model configuration.
        """

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        completion, retries = call_with_retry(
            func=palm.generate_text,
            exceptions_to_retry=(),
            max_retries=self.max_retries,
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self._format_results(completion, retries)
