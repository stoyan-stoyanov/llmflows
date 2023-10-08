# pylint: disable=R0913, R0902, R0801, W0221

"""
This module implements a wrapper for the Google PaLM chat API, using BaseChatLLM as a
base class.
"""

import google.generativeai as palm
from llmflows.llms.chat_llm import BaseChatLLM
from llmflows.llms.message_history import MessageHistory
from llmflows.llms.llm_utils import call_with_retry, async_call_with_retry


class PaLMChat(BaseChatLLM):
    """
    A class for interacting with the Google PaLM chat API.

    Inherits from BaseChatLLM.

    Uses the specified Google PaLM model and parameters for interacting with the Google
    PaLM chat API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/chat-bison-001",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        super().__init__(model)
        self.temperature = temperature
        self.max_retries = max_retries
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("You must provide Google API key")
        palm.configure(api_key=self._api_key)

    def _format_results(
            self, model_outputs, retries, message_history
    ) -> tuple[str, dict, dict]:
        """
        Formats results after generation.
        
        Args:
            model_outputs: The model outputs.
            retries: The number of retries.
            message_history: The message history.

        Returns:
            A tuple containing the generated text, the raw response data, and the model
                configuration.
        """ 
        text_result = model_outputs.last

        if text_result is None:
            raise ValueError(
                "No text result returned from PaLM API. The API might have blocked "
                f"the response.\n Filter values: {model_outputs.filters}")

        call_data = {
            "raw_outputs":{
                "model": model_outputs.model,
                "context": model_outputs.context,
                "examples": model_outputs.examples,
                "messages": model_outputs.messages,
                "filters": model_outputs.filters,
                "top_p": model_outputs.top_p,
                "top_k": model_outputs.top_k,
            },
            retries: retries
        }

        model_config = {
            "model": self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "messages": message_history.messages,
        }

        return text_result, call_data, model_config

    def _convert_message_history(self, message_history: MessageHistory) -> list[dict]:
        """
            Converts a MessageHistory object into PaLM conversation history format:

            [
                {'author': '0', 'content': 'Hello'},
                {'author': '1', 'content': 'Hi there! How can I help you today?'},
                {'author': '0', 'content': "Just chillin'"},
                {'author': '1', ....
            ]

            Args:
                message_history: The MessageHistory object.
            
            Returns:
                A list of dictionaries representing the conversation history.
        """
        if not message_history.messages:
            raise ValueError("Message history must have at least one user message.")

        history = []

        for message in message_history.messages:
            if message["role"] == "user":
                history.append({"author": "0", "content": message["content"]})
            elif message["role"] == "assistant":
                history.append({"author": "1", "content": message["content"]})

        return history


    def generate(self, message_history: MessageHistory) -> tuple[str, dict, dict]:
        """
        Generates text from the Google PaLM Chat API.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.
        
        Returns:
            A tuple containing the generated text, the raw response data, and the model
                configuration.
        """

        conversation_history = self._convert_message_history(message_history)
        
        completion, retries = call_with_retry(
            func=palm.chat,
            exceptions_to_retry=(),
            max_retries=self.max_retries,
            model=self.model,
            messages=conversation_history,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self._format_results(
            completion, retries, message_history
        )

        return str_message, call_data, model_config

    async def generate_async(self, message_history: MessageHistory) -> tuple[str, dict, dict]:
        """
        Generates text from the Google PaLM Chat API asynchronously.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.
        
        Returns:
            A tuple containing the generated text, the raw response data, and the model
                configuration.
        """

        conversation_history = self._convert_message_history(message_history)
        
        completion, retries = await async_call_with_retry(
            async_func=palm.chat_async,
            exceptions_to_retry=(),
            max_retries=self.max_retries,
            model=self.model,
            messages=conversation_history,
            temperature=self.temperature,
        )

        str_message, call_data, model_config = self._format_results(
            completion, retries, message_history
        )

        return str_message, call_data, model_config
