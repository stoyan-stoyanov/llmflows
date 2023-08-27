# pylint: disable=R0913
"""
This file contains the ChatFlowStep class, which represents a step in a Flow that is 
using a chat LLM.
"""

from typing import Any, Union
from llmflows.llms import MessageHistory
from llmflows.llms.chat_llm import BaseChatLLM
from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.callbacks.base_callback import BaseCallback
from llmflows.flows.flowstep import BaseFlowStep


class ChatFlowStep(BaseFlowStep):
    """
    Represents a step in a Flow that is utilizing a chat Language Learning Model (LLM).

    A ChatFlowStep calls a language model using a system prompt and a message prompt,
    records the run time, and optionally invokes callback functions on the
    results.

    Args:
        name (str): The name of the flow step.
        llm (OpenAIChat): The language model to be used in the flow step.
        output_key (str): The key for the output of the flow step.
        message_history (Union[MessageHistory, None]): predefined conversation history
        message_key (str): Key used to extract message from inputs.
        message_prompt_template (PromptTemplate): Prompt template for the message used
            with the language model.
        callbacks (Union[list[AsyncBaseCallback], None]): Callbacks to be invoked
            within the flowstep

    Attributes:
        llm (OpenAIChat): The language model to be used in the flow step.
        message_key (str): Key specifying which input should be used for a message.
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
        message_key: str,
        message_history: Union[MessageHistory, None] = None,
        message_prompt_template: Union[PromptTemplate, None] = None,
        callbacks: Union[list[BaseCallback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.message_key = message_key
        self.message_history = message_history if message_history else MessageHistory()
        self.message_prompt_template = message_prompt_template
        self.required_keys = self._add_required_keys()
        self._validate_message_key()

    def _add_required_keys(self):
        required_keys = {}
        if self.message_prompt_template:
            required_keys = {self.message_key}.union(
                self.message_prompt_template.variables,
            )

        return required_keys

    def _validate_message_key(self):
        if self.message_prompt_template:
            if self.message_key not in self.message_prompt_template.variables:
                raise ValueError(
                    "You've provided a message prompt template that doesn't contain "
                    "the message key variable."
                )

    def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        if self.message_prompt_template:
            message = self.message_prompt_template.get_prompt(**inputs)
        else:
            message = inputs[self.message_key]

        self.message_history.add_user_message(message)
        text_result, call_data, model_config = self.llm.generate(self.message_history)

        call_data["message_prompt_template"] = (
            self.message_prompt_template.prompt
            if self.message_prompt_template
            else None
        )
        call_data["message_prompt"] = message
        call_data["message_history"] = self.message_history.messages
        return text_result, call_data, model_config
