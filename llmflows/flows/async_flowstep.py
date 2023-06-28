# pylint: disable=R0801, R0913

"""
This module contains the AsyncBaseFlowStep and AsyncFlowStep classes, which represent 
individual steps in an AsyncFlow. Each FlowStep can execute a language model, record 
execution times, and optionally invoke callbacks on the results. The async 
implementation allows async flowsteps to be executed in parallel if multiple flowsteps
have all the required inputs available.
"""

import logging
import time
import datetime
from abc import ABC, abstractmethod
from typing import Any
from llmflows.llms.llm import BaseLLM
from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.callbacks.callback import Callback


class AsyncBaseFlowStep:
    """
    Base class for flowsteps in a flow.

    Attributes:
        name (str): The name of the flow step.
        output_key (str): The key for the output of the flow step.
        next_steps (list[BaseFlowStep]): The subsequent steps this step connects to.
        parents (list[BaseFlowStep]): The preceding steps that connect to this step.
        callbacks(list[Callback]): Optional functions to be invoked with the results.
    """

    def __init__(self, name: str, output_key: str, callbacks: list[Callback]):
        self.name = name
        self.output_key = output_key
        self.next_steps: list[AsyncBaseFlowStep] = []
        self.parents: list[AsyncBaseFlowStep] = []
        self.callbacks = [] if callbacks is None else callbacks

    def connect(self, *steps: "AsyncBaseFlowStep") -> None:
        """
        Connects this flow step to one or more subsequent flow steps.

        Args:
            *steps (AsyncBaseFlowStep): Async flow steps to connect to this step.

        Raises:
            ValueError: If connected flow steps have duplicate output keys.
        """
        self._check_unique_keys(*steps)
        self.next_steps.extend(steps)
        for step in steps:
            step.parents.append(self)

    def _check_unique_keys(self, *steps: "AsyncBaseFlowStep") -> None:
        """
        Ensures unique output keys among connected steps.

        Args:
            *steps (AsyncBaseFlowStep): Flow steps to be connected to this step.

        Raises:
            ValueError: If connected flow steps have the same output key.
        """
        output_keys = [step.output_key for step in steps]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("All connected flowsteps must have unique output keys.")


    @abstractmethod
    async def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        """
        Executes the language model with the provided inputs and returns result,
        call data and model configuration.

        Args:
            inputs (dict[str, Any]): The inputs to the flow step.

        Returns:
            tuple: result, call data and model configuration.
        """
        pass

    async def execute(self, inputs: dict[str, str], verbose: bool = False) -> dict[str, str]:
        """
        Executes the flow step with the provided inputs and returns a dictionary with
        execution details.

        This includes the start and end times, the prompts and the input to the
        language model, the output from the language model, details about the model
        configuration and the result of the step. Callback functions can be executed
        with the result as well.

        Args:
            inputs (dict[str, str]): The inputs to the flow step.
            verbose (bool, optional): If true, the output of the step
                and callback executions are printed.

        Returns:
            dict[str, str]: A dictionary with various execution details and results.
        """
        execution_info = {}
        start_time = datetime.datetime.now().isoformat()
        start_perf_time = time.perf_counter()
        execution_info["start_time"] = start_time
        execution_info["prompt_inputs"] = inputs

        for callback in self.callbacks:
            callback.on_start(inputs)

        result, call_data, model_config = await self.generate(inputs)
        execution_info["llm_output"] = result
        execution_info["call_data"] = call_data
        execution_info["model_config"] = model_config

        for callback in self.callbacks:
            callback.on_results(result)

        if verbose:
            print(f"{self.name}:\n{result}\n")

        end_time = datetime.datetime.now().isoformat()
        end_perf_time = time.perf_counter()

        execution_info["end_time"] = end_time
        execution_info["execution_time"] = end_perf_time - start_perf_time
        execution_info["result"] = {self.output_key: result}

        for callback in self.callbacks:
            callback.on_end(execution_info)

        return execution_info


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
        callbacks: list[Callback] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.prompt_template = prompt_template
        self.required_keys = prompt_template.variables
        self.callbacks = callbacks if callbacks else []

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
