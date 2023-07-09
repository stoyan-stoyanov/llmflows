# pylint: disable=R0801

"""
LLMFlow module for the Flow class used for defining and executing flows, which are 
digraphs of steps. Each step is represented by a `FlowStep` instance.
"""

from llmflows.flows.flowstep import FlowStep
from llmflows.flows.base_flow import BaseFlow


class Flow(BaseFlow):
    """
    Base class for all flows. Each flow is a digraph of steps, represented by FlowStep
    instances.

    Args:
        first_step (FlowStep): The first step of the flow.

    Attributes:
        _first_step (FlowStep): The first step in the flow.
        results (dict): Stores the results of the executed flow steps.
        completed_steps (set): Keeps track of the steps that have been executed.
    """

    def __init__(self, first_step: FlowStep):
        super().__init__(first_step)
        self.results = {}
        self.completed_steps = set()

    def start(self, verbose=False, **inputs) -> dict:
        """
        Executes the flow with the provided inputs.

        Args:
            verbose (bool): Specifies if the flow step should print their output.
            **inputs (dict): The inputs to the flow.

        Returns:
            A dictionary of the results from each flow step.

        Raises:
            ValueError: If any required inputs are missing.
        """
        self._check_all_input_keys_available(inputs)
        self._run_step(self._first_step, inputs, verbose)
        return self.results

    def _run_step(self, step, inputs, verbose):
        """
        Executes the given step and its next steps in a DFS-like manner.

        Args:
            step (FlowStep): The step to run.
            inputs (dict): The inputs to the step.
            verbose (bool): Specifies if the flow step should print its output.

        Returns:
            Any: The output of the step.
        """

        if not step or any(parent.output_key not in inputs for parent in step.parents):
            return

        required_inputs = {
            key: inputs[key] for key in step.required_keys
        }

        if step not in self.completed_steps:
            flow_data = step.run(required_inputs, verbose)
            self.completed_steps.add(step)

            if flow_data:
                self.results[step.name] = flow_data
                inputs.update(flow_data["result"])

        for next_step in step.next_steps:
            self._run_step(next_step, inputs, verbose)
