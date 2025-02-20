# pylint: disable=R0913, R0801
"""
This module implements a wrapper for Anthropic's Claude chat models, using BaseChatLLM 
as a base class.
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
from typing import Optional, Callable, get_type_hints
from inspect import signature


class ClaudeChat(BaseChatLLM):
    """
    A class for interacting with the Claude API.

    Inherits from BaseChatLLM.

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
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        max_retries: int = 3,
        verbose: bool = False,
        tools: Optional[list[dict]] = None,
        tools_functions: Optional[list[Callable]] = None
    ):
        super().__init__(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("API key must be specified.")
        self.client = Anthropic(api_key=self._api_key)
        self.async_client = AsyncAnthropic(api_key=self._api_key)
        self.tools = tools
        self.tools_functions = {f.__name__: f for f in tools_functions} if tools_functions else None
        self.mode_params = {}
        self._configure_optional_modes(tools)
    
    def _configure_optional_modes(self, tools) -> None:
        """
        Configures optional modes for the Claude API.

        Args:
            tools (list[dict]): The tools to use for function calling.
        """

        if tools:
            self._validate_tools_setup()
            self.mode = "tools"
            self.mode_params = {"tools": self.tools}
        else:
            self.mode = None
    
    def _validate_tools_setup(self) -> None:
        """
        Validates tools setup and checks if:
        - Functions defined in the schema exist in tools_functions
        - Required attributes in the schema exist as parameters in tools_functions
        - All attributes in the schema have types
        - Attributes in tools_functions lack types

        Raises:
            ValueError: If any of the checks fail.
        """
        errors = []
        warnings = []

        if self.tools_functions is None:
            raise ValueError(
                "If you want to use function (tools) calling, you must provide a "
                "list of python functions with the tools_functions argument"
            )


        for tool in self.tools:
            tool_name = tool["name"]

            if tool_name not in self.tools_functions:
                errors.append(f"Tool {tool_name} not found in tools_functions")
                continue

            func = self.tools_functions[tool_name]
            schema = tool["input_schema"]
            properties = schema["properties"]

            func_params = signature(func).parameters
            func_types = get_type_hints(func)

            required_params = schema.get("required", [])
            for param_name in required_params:
                if param_name not in func_params:
                    errors.append(f"Required parameter {param_name} missing in function {tool_name}")

            for param_name, param_info in properties.items():
                if "type" not in param_info:
                    errors.append(f"Missing type definition for parameter {param_name} in {tool_name} schema")

            for param_name in func_params:
                if param_name not in func_types:
                    warnings.append(f"Missing type hint for parameter {param_name} in {tool_name}")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"- {warning}")

        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"- {error}")
            raise ValueError(f"Tools setup is not valid. Please check the errors and warnings above.")


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

        text_result = model_outputs.content[0].text

        tool_calling = None

        print("model_outputs: ", model_outputs)
        # Check if any of the dicts in model_outputs.content have a "tool_use" value in the "type" key
        for content in model_outputs.content:
            tool_calls = []
            if content.type == "tool_use":
                tool_calls.append({
                    "function_name": content.name,
                    "function_arguments": content.input,
                    "tool_call_id": content.id
                })

        if tool_calls:
            tool_calling = {
                "tool_calls": tool_calls
            }

        call_data = {
            "raw_outputs": model_outputs,
            "retries": retries,
            "tool_calling": tool_calling,
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
        Gets the system prompt from message history if it's available and removes it
        from the message history.

        Args:
            message_history (MessageHistory): The message history to convert.

        Returns:
            list: The converted message history.
        """
        print("----------ORIGINAL MESSAGES---------------------")
        print(message_history.messages)
        print("------------------------------------------------")
        system = ""

        if not message_history.messages:
            raise ValueError("Message history must have at least one user message.")

        # if message_history.messages[-1]["role"] != "user":
        #     raise ValueError("Last message in message history must be from the user.")
        
        if message_history.messages[0]["role"] == "system":
            system = message_history.messages[0]["content"]
            message_history.messages = message_history.messages[1:]

        converted_messages = []
        for original_msg in message_history.messages:
            msg = original_msg.copy()

            if "tool_calls" in msg:
                msg["content"] = [
                    {
                        "type": "text",
                        "text": msg["content"]
                    }
                ]
                for tool_call in msg["tool_calls"]:
                    msg["content"].append({
                        "type": "tool_use",
                        "id": tool_call["tool_call_id"],
                        "name": tool_call["function_name"],
                        "input": tool_call["function_arguments"],
                    })

                del msg["tool_calls"]
            
            content = msg.get("content")
            if isinstance(content, list):
                converted_list = []
                for item in content:
                    if item.get("type") == "image":
                        converted_list.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item["mime_type"],
                                    "data": item.get("base64")
                                }
                            })
                    else:
                        converted_list.append(item)
                msg["content"] = converted_list

            if msg.get("role") == "tool_result":
                msg["role"] = "user"
                msg["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": msg.get("content")
                    }
                ]
                del msg["tool_call_id"]

            converted_messages.append(msg)

        print("--------------------------------")
        print("converted messages: ", converted_messages)
        print("system: ", system)
        print("--------------------------------")
        
        return system, converted_messages


    def generate(self, message_history: MessageHistory) -> tuple[str, dict, dict]:
        """
        Generates text from the Claude API.

        Args:
            message_history: A `MessageHistory` object representing the conversation
                history.

        Returns:
            A tuple containing the generated text, the raw response data, and the model
                configuration.
        """
        system, messages = self._convert_message_history(message_history)
        completion, retries = call_with_retry(
            func=self.client.messages.create,
            exceptions_to_retry=(
                RateLimitError,
                InternalServerError,
                APIConnectionError,
            ),
            system=system,
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            **self.mode_params,
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
            A tuple containing the generated text, the raw response data, and the model
                configuration.
        """
        completion, retries = await async_call_with_retry(
            async_func=self.async_client.messages.create,
            exceptions_to_retry=(
                RateLimitError,
                InternalServerError,
                APIConnectionError,
            ),
            messages=message_history.messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            **self.mode_params
        )

        str_message, call_data, model_config = self._format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config
