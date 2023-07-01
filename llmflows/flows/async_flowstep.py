# pylint: disable=R0801, R0913

"""
This module the AsyncFlowStep class, which can execute a language model, record
execution times, and optionally invoke callbacks on the results. The async 
implementation allows async flowsteps to be executed in parallel if multiple flowsteps
have all the required inputs available.
"""

import logging
from typing import Any, Union
from llmflows.llms.llm import BaseLLM
from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.callbacks.callback import Callback
from llmflows.flows.async_base_flowstep import AsyncBaseFlowStep


class AsyncFlowStep(AsyncBaseFlowStep):
    """
    Represents a specific async step in an async Flow.

    An AsyncFlowStep executes a language model using a prompt template, records the 
    execution time, and optionally invokes callback functions on the results.
    Async Flowsteps can be executed in parallel in an AsyncFlow if all the required 
    inputs are available.

    Attributes:
        name (str): The name of the flow step.
        output_key (str): The key for the output of the flow step.
        llm: The language model to be used in the flow step.
        prompt_template (PromptTemplate): Template for the prompt to be used with the 
            language model.
        callbacks (list[Callback]): Optional functions to be invoked with the results.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        prompt_template: PromptTemplate,
        output_key: str,
        callbacks: Union[list[Callback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.prompt_template = prompt_template
        self.required_keys = prompt_template.variables

    async def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        prompt = self.prompt_template.get_prompt(**inputs)
        return await self.llm.generate_async(prompt)


class AsyncChatFlowStep(AsyncBaseFlowStep):
    """
    Represents an async step in a Flow that is utilizing a chat LLM.

    An AsyncChatFlowStep executes a language model using two prompt templates, namely 
    a system prompt and a message prompt, records the execution time, and optionally 
    invokes callback functions on the results.

    Attributes:
        name (str): The name of the flow step.
        output_key (str): The key for the output of the flow step.
        llm: The language model to be used in the flow step.
        system_prompt_template (PromptTemplate): Template for the system prompt to be 
            used with the language model.
        message_prompt_template (PromptTemplate): Template for the message prompt to 
            be used with the language model.
        message_key (str): Key used to extract message from inputs.
        callbacks (list[Callback]): Optional functions to be invoked with the results.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt_template: PromptTemplate,
        message_key: str,
        output_key: str,
        message_prompt_template: PromptTemplate = None,
        callbacks: list[Callback] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.message_key = message_key
        self.system_prompt_template = system_prompt_template
        self.message_prompt_template = message_prompt_template
        self.callbacks = callbacks if callbacks else []
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

    async def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        system_prompt = self.system_prompt_template.get_prompt(**inputs)
        self.llm.update_system_prompt(system_prompt)

        if self.message_prompt_template:
            message = self.message_prompt_template.get_prompt(**inputs)
        else:
            message = inputs[self.message_key]
        self.llm.add_message(message)

        return await self.llm.generate_async()
