# pylint: disable=R0801, R0913

"""
This module contains the BaseFlowStep used as a base class by all non-async
FlowStep classes. Each FlowStep can execute a task, record
execution times, and optionally invoke callbacks on the results.
"""

import time
import datetime
from abc import ABC, abstractmethod
from typing import Any
from llmflows.callbacks.callback import Callback


class BaseFlowStep(ABC):
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
        self.next_steps: list[BaseFlowStep] = []
        self.parents: list[BaseFlowStep] = []
        self.callbacks = [] if callbacks is None else callbacks

    def connect(self, *steps: "BaseFlowStep") -> None:
        """
        Connects this flow step to one or more subsequent flow steps.

        Args:
            *steps (BaseFlowStep): Flow steps to connect to this step.

        Raises:
            ValueError: If connected flow steps have the same output key.
        """
        self._check_unique_keys(*steps)
        self.next_steps.extend(steps)
        for step in steps:
            step.parents.append(self)

    def _check_unique_keys(self, *steps: "BaseFlowStep") -> None:
        """
        Checks that all connected flow steps have unique output keys.

        Args:
            *steps (BaseFlowStep): Flow steps to be connected to this step.

        Raises:
            ValueError: If connected flow steps have the same output key.
        """
        output_keys = [step.output_key for step in steps]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("All connected flowsteps must have unique output keys.")

    @abstractmethod
    def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        """
        Executes the language model with the provided inputs and returns result,
        call data and model configuration.

        Args:
            inputs (dict[str, Any]): The inputs to the flow step.

        Returns:
            tuple: result, call data and model configuration.
        """
        pass

    def execute(self, inputs: dict[str, str], verbose: bool = False) -> dict[str, str]:
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

        result, call_data, model_config = self.generate(inputs)
        execution_info["generated"] = result
        execution_info["call_data"] = call_data
        execution_info["config"] = model_config

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
