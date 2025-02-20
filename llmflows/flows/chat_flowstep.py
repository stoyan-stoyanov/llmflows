# pylint: disable=R0913
"""
This file contains the ChatFlowStep class, which represents a step in a Flow that is 
using a chat LLM.
"""

from typing import Any, Union, Optional
from llmflows.llms import MessageHistory
from llmflows.llms.chat_llm import BaseChatLLM
from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.callbacks.base_callback import BaseCallback
from llmflows.flows.flowstep import BaseFlowStep
from llmflows.llms.inputs import Image
import json


class ChatFlowStep(BaseFlowStep):
    """
    Represents a step in a Flow that is utilizing a chat Language Learning Model (LLM).

    A ChatFlowStep calls a language model using a system prompt and a message prompt,
    records the run time, and optionally invokes callback functions on the
    results. The message prompt is generated using a PromptTemplate, which can be
    configured to use variables from other flow steps. The message prompt is appended
    to the message history of the ChatLLM before calling the language model.

    If function calling is enabled and the LLM decides to call a tool, the tool
    results are added to the message history and the LLM is called again with the
    updated message history to provide a final response.

    Args:
        name (str): The name of the flow step.
        llm (OpenAIChat): The language model to be used in the flow step.
        output_key (str): The key for the output of the flow step.
        message_history (Union[MessageHistory, None]): predefined conversation history
        message_prompt_template (PromptTemplate): Prompt template for the message used
            with the language model. Gets appended to the message history.
        callbacks (Union[list[AsyncBaseCallback], None]): Callbacks to be invoked
            within the flowstep

    Attributes:
        llm (BaseChatLLM): The language model to be used in the flow step.
        message_history (MessageHistory): Message history for the ChatLLM if not 
            passed, an empty Message history is created.
        message_prompt_template (PromptTemplate): Prompt template for the message used
            with the language model.
        required_keys (set[str]): The keys required for the flow step to run.
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatLLM,
        output_key: str,
        message_prompt_template: PromptTemplate,
        images: Optional[list[Image]] = None,
        run_tools: Optional[bool] = False,
        message_history: Union[MessageHistory, None] = None,
        callbacks: Union[list[BaseCallback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.message_history = message_history if message_history else MessageHistory()
        self.message_prompt_template = message_prompt_template
        self.run_tools = run_tools
        self.required_keys = self.message_prompt_template.variables
        self.images = images

    def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        message = self.message_prompt_template.get_prompt(**inputs)
        self.message_history.add_user_message(message, self.images)

        text_result, call_data, model_config = self.llm.generate(self.message_history)

        if call_data["tool_calling"]:
            # Add the assistant message that decided to call tools
            self.message_history.add_ai_message(
                message=text_result, 
                tool_calls=call_data["tool_calling"]["tool_calls"])
            tool_results = []

            # Call the tools and add the results to the message history
            
            for tool_call in call_data["tool_calling"]["tool_calls"]:
                if isinstance(tool_call["function_arguments"], str):
                    function_result = self.llm.tools_functions[tool_call["function_name"]](**json.loads(tool_call["function_arguments"]))
                else:
                    function_result = self.llm.tools_functions[tool_call["function_name"]](**tool_call["function_arguments"])
                tool_results.append(function_result)
                if self.run_tools:
                    self.message_history.add_tool_result_message(function_result, tool_call["tool_call_id"])

            # Call the LLM again with the tool results for a final response or return the tool results
            if self.run_tools:
                text_result, call_data, model_config = self.llm.generate(self.message_history)
            else:
                text_result = json.dumps(tool_results)

        # TODO: We have 2 calls but only one call_data. We need to make it a list of dicts
        self.message_history.add_ai_message(message=text_result)

        call_data["message_prompt_template"] = (
            self.message_prompt_template.prompt
            if self.message_prompt_template
            else None
        )
        call_data["message_prompt"] = message
        call_data["message_history"] = self.message_history.messages

        return text_result, call_data, model_config
