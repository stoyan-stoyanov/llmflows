# pylint: disable=R0913, R0801
"""
This module implements a wrapper for Anthropic's Claude chat models, using BaseLLM as a 
base class.
"""

from anthropic import (
    Anthropic,
    AsyncAnthropic,
    HUMAN_PROMPT,
    AI_PROMPT,
    RateLimitError,
    InternalServerError,
    APIConnectionError,
)
from llmflows.llms.chat_llm import BaseChatLLM
from llmflows.llms.llm_utils import call_with_retry, async_call_with_retry
from llmflows.llms.message_history import MessageHistory


class ClaudeChat(BaseChatLLM):
    """
    A class for interacting with the Claude API.

    Inherits from BaseLLM.

    Uses the specified Claude model and parameters for interacting with the Claude API,
    and provides methods to add, remove, replace messages, update system prompts, and
    generate responses.

    Args:
        model (str): The name of the Claude model to use.
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.
        api_key (str): The API key to use for interacting with the Claude API.

    Attributes:
        temperature (float): The temperature to use for text generation.
        max_tokens (int): The maximum number of tokens to generate.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.

    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-2",
        temperature: float = 0.7,
        max_tokens: int = 256,
        max_retries: int = 3,
        verbose: bool = False,
    ):
        super().__init__(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.verbose = verbose
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("API key must be specified.")

    def _format_results(
        self, model_outputs, retries, message_history
    ) -> tuple[str, dict, dict]:
        """
        Formats results after generation.

        Args:
            model_outputs: Raw output after model generation.
            retries (int): Number of retries taken for successful generation.

        Returns:
            tuple(str, dict, dict): Formatted output text, raw outputs, and model
                configuration.
        """
        text_result = model_outputs.completion

        call_data = {
            "raw_outputs": {
                "completion": model_outputs.completion,
                "model": model_outputs.model,
                "stop_reason": model_outputs.stop_reason,
                "stop": model_outputs.stop,
                "log_id": model_outputs.log_id,
            },
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

    def _convert_message_history(self, message_history: MessageHistory) -> str:
        """
        Converts a MessageHistory object to a Claude prompt string with the required
        format: `\n\nHuman: Why is the sky blue?\n\nAssistant:`

        Args:
            message_history (MessageHistory): The message history to convert.

        Returns:
            str: The converted message history.

        Raises:
            ValueError: If the message history is empty or the last message is not from
                the user.
        """
        if not message_history.messages:
            raise ValueError("Message history must have at least one user message.")

        if message_history.messages[-1]["role"] != "user":
            raise ValueError("Last message in message history must be from the user.")

        claude_prompt = ""

        for message in message_history.messages:
            if message["role"] == "user":
                claude_prompt += f"{HUMAN_PROMPT} {message['content']}"
            elif message["role"] == "assistant":
                claude_prompt += f"{AI_PROMPT} {message['content']}"

        claude_prompt += AI_PROMPT

        return claude_prompt

    def generate(self, message_history: MessageHistory) -> tuple[str, dict, dict]:
        """
        Generates text from the Claude API.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.

        Returns:
            A string representing the generated text.
        """
        client = Anthropic(api_key=self._api_key)
        claude_prompt = self._convert_message_history(message_history)

        completion, retries = call_with_retry(
            func=client.completions.create,
            exceptions_to_retry=(
                RateLimitError,
                InternalServerError,
                APIConnectionError,
            ),
            prompt=claude_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens_to_sample=self.max_tokens,
            max_retries=self.max_retries,
        )

        str_message, call_data, model_config = self._format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config

    async def generate_async(
        self, message_history: MessageHistory
    ) -> tuple[str, dict, dict]:
        """
        Generates text from the Claude API asynchronously.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.

        Returns:
            A string representing the generated text.
        """
        client = AsyncAnthropic(api_key=self._api_key)
        claude_prompt = self._convert_message_history(message_history)

        completion, retries = await async_call_with_retry(
            async_func=client.completions.create,
            exceptions_to_retry=(
                RateLimitError,
                InternalServerError,
                APIConnectionError,
            ),
            prompt=claude_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens_to_sample=self.max_tokens,
            max_retries=self.max_retries,
        )

        str_message, call_data, model_config = self._format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config
