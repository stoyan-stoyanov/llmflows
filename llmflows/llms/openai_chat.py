# pylint: disable=R0913, R0902, R0801, W0221
"""
This module implements a wrapper for OpenAI chat models, using BaseChatLLM as a 
base class.
"""

from openai import OpenAI as OpenAIClient
from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import (
    APIError,
    Timeout,
    RateLimitError,
    APIConnectionError,
)
from llmflows.llms.chat_llm import BaseChatLLM
from llmflows.llms.llm_utils import call_with_retry, async_call_with_retry
from llmflows.llms.message_history import MessageHistory
from typing import Optional, Callable, get_type_hints
from inspect import signature

class OpenAIChat(BaseChatLLM):
    """
    A class for interacting with the OpenAI chat API.

    Inherits from BaseChatLLM.

    Uses the specified OpenAI model and parameters for interacting with the OpenAI chat
    API, and provides methods to add, remove, replace messages, update system prompts,
    and generate responses.

    Args:
        api_key (str): The API key to use for interacting with the OpenAI API.
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature to use for text generation.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.
        tools (list[dict]): The tools to use for function calling.
        tools_functions (list[Callable]): The functions to use for function calling.
        response_format (dict): The response format to use for structured outputs.
        prediction (dict): The prediction to use for predicted outputs.

    Attributes:
        temperature (float): The temperature to use for text generation.
        max_retries (int): The maximum number of retries for generating tokens.
        verbose (bool): Whether to print debug information.
        tools (list[dict]): The tools definitions to use for function calling.
        tools_functions (dict[str, Callable]): The functions to use for function calling.
        response_format (dict): The response format to use for structured outputs.
        prediction (dict): The prediction to use for predicted outputs.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_retries: int = 3,
        verbose: bool = False,
        tools: Optional[list[dict]] = None,
        tools_functions: Optional[list[Callable]] = None,
        response_format: Optional[dict] = None,
        prediction: Optional[dict] = None,
    ):
        super().__init__(model)
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        self._api_key = api_key
        if not self._api_key:
            raise ValueError("You must provide OpenAI API key")
        self.client = OpenAIClient(api_key=self._api_key)
        self.async_client = AsyncOpenAIClient(api_key=self._api_key)
        self.tools = tools
        self.tools_functions = {f.__name__: f for f in tools_functions} if tools_functions else None
        self.response_format = response_format
        self.prediction = prediction
        self.mode_params = {}
        self._configure_optional_modes(tools, response_format, prediction)

    def _configure_optional_modes(self, tools, response_format, prediction) -> None:
        """
        Check if more than one of the optional modes is used(Function calling, 
        structured outputs, or predicted outputs). Select the correct client function 
        and API arguments to invoke the selected mode. If more than one mode is used, 
        raise an error.

        Args:
            tools (list[dict]): The tools to use for function calling.
            response_format (dict): The response format to use for structured outputs.
            prediction (dict): The prediction to use for predicted outputs.

        Raises:
            ValueError: If more than one of the optional modes is used.
        """
        modes = {name: value for name, value in locals().items() 
                if value is not None and name != 'self'}

        if len(modes) > 1:
            raise ValueError(
                "You can not use more than one of the following modes "
                "at the same time:\n\t- function calling\n\t- structured outputs\n\t" 
                "- predicted outputs"
            )

        self.mode = next(iter(modes), None)

        # TODO: refactor when OpenAI moves all modes out of beta
        if self.mode == "tools":
            self._validate_tools_setup()
            self.client_func = self.client.chat.completions.create
            self.mode_params = {"tools": self.tools}
        elif self.mode == "response_format":
            self.client_func = self.client.beta.chat.completions.parse
            self.mode_params = {"response_format": self.response_format}
        elif self.mode == "prediction":
            self.client_func = self.client.beta.chat.completions.parse
            self.mode_params = {"prediction": self.prediction}
        else:
            self.client_func = self.client.chat.completions.create

    def _validate_tools_setup(self) -> None:
        """
        Validates tools setup and checks if:
        - Functions defined in the schema exist in tools_functions
        - Required attributes in the schema exist as parameters in tools_functions
        - All attributes in the schema have types
        - Attributes in tools_functions lack types

        Args:
            tools: List of tool definitions in JSON schema format
            tools_functions: Dictionary mapping tool names to the function 
                implementations
        
        Returns:
            None
        """
        errors = []
        warnings = []

        if self.tools_functions is None:
            raise ValueError(
                "If you want to use function (tools) calling, you must provide a "
                "list of python functions with the tools_functions argument"
            )

        for tool in self.tools:
            tool_name = tool["function"]["name"]

            if tool_name not in self.tools_functions:
                errors.append(f"Tool {tool_name} not found in tools_functions")
                continue

            func = self.tools_functions[tool_name]
            schema = tool["function"]["parameters"]
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
        Prepares results after generation.

        Args:
            model_outputs: Raw output after model generation.
            retries (int): Number of retries taken for successful generation.

        Returns:
            tuple(str, dict, dict): Formatted output text, raw outputs, and model
                configuration.
        """

        message = model_outputs.choices[0].message

        text_result = message.content

        tool_calling = None

        """
        tool calling format:
        [
            {
                "function_name": tool_call.function.name,
                "function_arguments": tool_call.function.arguments,
                "tool_call_id": tool_call.id
            },
            ...
        ]
        """

        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append({
                    "function_name": tool_call.function.name,
                    "function_arguments": tool_call.function.arguments,
                    "tool_call_id": tool_call.id
                })

            tool_calling = {
                "tool_calls": tool_calls,
            }

        call_data = {
            "raw_outputs": model_outputs,
            "retries": retries,
            "tool_calling": tool_calling,
        }

        model_config = {
            "model_name": self.model,
            "temperature": self.temperature,
            "max_messages": message_history.max_messages,
            "messages": message_history.messages,
        }

        return text_result, call_data, model_config

    def _convert_message_history(self, message_history: MessageHistory) -> str:
        """
        Converts the message history to a format that can be used by the OpenAI API.
        Replaces the 'tool_result' role with 'tool' role.
        Replaces the 'content' with None if the message includes tool calls.
        Replaces the 'tool_calls' with the correct format.
        Replaces the 'content' with the correct format if the message includes images.

        For more information about the OpenAI API message format, see:
        - https://platform.openai.com/docs/guides/text-generation
        - https://platform.openai.com/docs/guides/vision
        - https://platform.openai.com/docs/guides/function-calling

        Args:
            message_history (MessageHistory): The message history to convert.

        Returns:
            list: The converted message history.
        """
        print("----------ORIGINAL MESSAGES---------------------")
        print(message_history.messages)
        print("------------------------------------------------")
        converted_messages = []

        for original_msg in message_history.messages:
            msg = original_msg.copy()

            if msg.get("role") == "tool_result":
                msg["role"] = "tool"

            content = msg.get("content")
            if isinstance(content, list):
                converted_list = []
                for item in content:
                    if item.get("type") == "image":
                        converted_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{item['mime_type']};base64,{item['base64']}"}
                        })
                    else:
                        converted_list.append(item)
                msg["content"] = converted_list

            if "tool_calls" in msg:
                tool_calls = []
                for call in msg["tool_calls"]:
                    tool_calls.append({
                        "id": call["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": call["function_name"],
                            "arguments": call["function_arguments"]
                        },
                    })

                msg["content"] = None
                msg["tool_calls"] = tool_calls

            converted_messages.append(msg)
        
        print("--------------------------------")
        print(converted_messages)
        print("--------------------------------")

        return converted_messages


    def generate(self, message_history: MessageHistory) -> tuple[str, dict, dict]:
        """
        Sends the messages to the OpenAI chat API and returns a chat message response.

        Returns:
            A tuple containing the generated chat message, raw output data, and model
                configuration.
        """

        completion, retries = call_with_retry(
            func=self.client_func,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
            ),
            max_retries=self.max_retries,
            model=self.model,
            messages=self._convert_message_history(message_history),
            temperature=self.temperature,
            **self.mode_params
        )

        str_message, call_data, model_config = self._format_results(
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
            async_func=self.async_client.chat.completions.create,
            exceptions_to_retry=(
                APIError,
                Timeout,
                RateLimitError,
                APIConnectionError,
            ),
            max_retries=self.max_retries,
            model=self.model,
            messages=message_history.messages,
            temperature=self.temperature,
            **self.mode_params
        )

        str_message, call_data, model_config = self._format_results(
            model_outputs=completion, retries=retries, message_history=message_history
        )

        return str_message, call_data, model_config




# for m in message_history.messages:
        #     new_msg = {}
        #     for key, value in m.items():
        #         if key == 'role':
        #             if value == 'tool_result':
        #                 new_msg[key] = 'tool'
        #             else:
        #                 new_msg[key] = value
        #         elif key == 'tool_calls':
        #             tool_calls = []
        #             for tool_call in value:
        #                 tool_calls.append({
        #                     "id": tool_call["tool_call_id"],
        #                     "type": "function",
        #                     "function": {
        #                         "name": tool_call["function_name"],
        #                         "arguments": tool_call["function_arguments"],
        #                     },
        #                 })
        #             new_msg["content"] = None
        #             new_msg["tool_calls"] = tool_calls
        #         elif key == 'content':
        #             if type(value) == list:
        #                 content_list = []
        #                 for item in value:
        #                     if item["type"] == "image":
        #                         content_list.append({
        #                             "type": "image_url",
        #                             "image_url": {"url": f"data:image/jpeg;base64,{item['base64']}"}
        #                         })
        #                     else:
        #                         content_list.append(item)
        #                 new_msg["content"] = content_list
        #             else:
        #                 new_msg["content"] = value
        #         else:
        #             new_msg[key] = value
        #     messages.append(new_msg)