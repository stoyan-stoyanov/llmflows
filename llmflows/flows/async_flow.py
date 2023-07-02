# pylint: disable=R0801

"""
LLMFlow module for the AsyncFlow class used for defining and executing flows, which are 
digraphs of steps. The async implementation of this class allows for the parallel
execution of async flowsteps that have all their required inputs available.
"""

import asyncio
from llmflows.flows.async_flowstep import AsyncFlowStep
from llmflows.flows.async_base_flow import AsyncBaseFlow


class AsyncFlow(AsyncBaseFlow):
    """
    Async implementation of BaseFlow that executes a series of FlowSteps in a Directed
    Acyclic Graph (DAG) structure.

    Args:
        first_step (AsyncFlowStep): The first step of the flow.

    Attributes:
        _first_step (AsyncFlowStep): The first step in the flow.
        results (dict): Stores the results of the executed flow steps.
        executed_steps (set): Keeps track of the steps that have been executed.
    """

    def __init__(self, first_step: AsyncFlowStep):
        super().__init__(first_step)
        self.results = {}
        self.executed_steps = set()

    async def execute(self, verbose=False, **inputs) -> dict:
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
        await self._execute_step(self._first_step, inputs, verbose)
        return self.results

    async def _execute_step(self, step, inputs, verbose):
        """
        Executes the given step and its next steps in a DFS-like manner.

        Args:
            step (AsyncFlowStep): The step to execute.
            inputs (dict): The inputs to the step.
            verbose (bool): Specifies if the flow step should print its output.

        Returns:
            Any: The output of the step.
        """
        if not step or any(parent.output_key not in inputs for parent in step.parents):
            return

        required_inputs = {key: inputs[key] for key in step.required_keys}

        if step not in self.executed_steps:
            flow_data = await step.execute(required_inputs, verbose)
            self.executed_steps.add(step)

            if flow_data:
                self.results[step.name] = flow_data
                inputs.update(flow_data["result"])

        tasks = [
            self._execute_step(next_step, inputs, verbose)
            for next_step in step.next_steps
        ]
        await asyncio.gather(*tasks)
