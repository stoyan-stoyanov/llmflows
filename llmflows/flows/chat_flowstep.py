"""
This file contains the ChatFlowStep class, which represents a step in a Flow that is 
using a chat LLM.
"""

import logging
from typing import Any, Union
from llmflows.llms.llm import BaseLLM
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
        llm (BaseLLM): The language model to be used in the flow step.
        output_key (str): The key for the output of the flow step.
        system_prompt_template (PromptTemplate): Prompt tempalte for the system prompt
            to be used with the language model.
        message_key (str): Key used to extract message from inputs.
        message_prompt_template (PromptTemplate): Prompt template for the message used
            with the language model.
        callbacks (Union[list[AsyncBaseCallback], None]): Callbacks to be invoked
            within the flowstep

    Attributes:
        llm (BaseLLM): The language model to be used in the flow step.
        message_key (str): Key used to extract message from inputs.
        system_prompt_template (PromptTemplate): Prompt tempalte for the system prompt
            to be used with the language model.
        message_prompt_template (PromptTemplate): Prompt template for the message used
            with the language model.
        required_keys (set[str]): The keys required for the flow step to run.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        output_key: str,
        system_prompt_template: PromptTemplate,
        message_key: str,
        message_prompt_template: Union[PromptTemplate, None] = None,
        callbacks: Union[list[BaseCallback], None] = None
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.message_key = message_key
        self.system_prompt_template = system_prompt_template
        self.message_prompt_template = message_prompt_template
        self.required_keys = self._add_required_keys()
        self._validate_message_key()

    def _add_required_keys(self):
        if self.message_prompt_template:
            required_keys = {self.message_key}.union(
                self.system_prompt_template.variables,
                self.message_prompt_template.variables,
            )
        else:
            required_keys = {self.message_key}.union(
                self.system_prompt_template.variables
            )

        return required_keys

    def _validate_message_key(self):
        if self.message_key in self.system_prompt_template.variables:
            logging.warning(
                "The message_key matches a variable in the system"
                " prompt.\nmessage_key: %s\nsystem_prompt_template"
                " variables: %s. Ignore this warning"
                " if you intended to include the message in the system prompt.",
                self.message_key,
                self.system_prompt_template.variables
            )

        if self.message_prompt_template:
            if self.message_key not in self.message_prompt_template.variables:
                raise ValueError(
                    "You've provided a message prompt template that doesn't contain "
                    "the message key variable."
                )

    def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        system_prompt = self.system_prompt_template.get_prompt(**inputs)
        self.llm.update_system_prompt(system_prompt)
        if self.message_prompt_template:
            message = self.message_prompt_template.get_prompt(**inputs)
        else:
            message = inputs[self.message_key]
        self.llm.add_message(message)
        results = self.llm.generate()

        results[1]["system_prompt_template"] = self.system_prompt_template.prompt
        results[1]["system_prompt"] = system_prompt
        results[1]["message_prompt_template"] = self.message_prompt_template.prompt
        results[1]["message_prompt"] = message
        results[1]["message_history"] = self.llm.messages
        return results
