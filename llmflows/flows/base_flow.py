"""
This module provides the BaseFlow class which is used as a base class by all non-async
flow classes.
"""

from llmflows.flows.base_flowstep import BaseFlowStep


class BaseFlow:
    """
    Base class for all flows. Each flow is a sequence of steps, represented by FlowStep
    instances.

    Args:
        first_step (FlowStep): The first step of the flow.

    Attributes:
        steps (list): All steps in the flow.
        output_keys (set): Set of output keys for all steps in the flow.
        input_keys (set): Set of input keys for all steps in the flow.
        names (set): Set of names of all steps in the flow.
    """

    def __init__(self, first_step: BaseFlowStep):
        self._first_step = first_step
        self.steps = self._get_all_steps()
        self.output_keys = set()
        self.input_keys = set()
        self.names = set()
        self._check_unique_attributes()

    def set_first_step(self, step):
        """
        Sets the initial step of the flow.

        Args:
            step (FlowStep): The initial step for the flow.
        """
        self._first_step = step

    def _get_all_steps(self):
        """
        Traverses the flow to get all steps.

        Returns:
            list: A list containing all steps in the flow.
        """
        queue = [self._first_step]
        visited_steps = set()
        all_steps = []

        while queue:
            current_step = queue.pop(0)

            if current_step in visited_steps or current_step is None:
                continue

            visited_steps.add(current_step)
            all_steps.append(current_step)
            queue.extend(current_step.next_steps)

        return all_steps

    def _check_unique_attributes(self):
        """
        Checks that all flow steps have unique output keys and names.

        Raises:
            ValueError: If any flow steps have the same output key or name.
        """
        for step in self.steps:
            if step.output_key in self.output_keys:
                raise ValueError(
                    f"The output key '{step.output_key}' has already been used"
                    " in another FlowStep."
                )

            if step.name in self.names:
                raise ValueError(
                    f"The name '{step.name}' has already been used"
                    " for another FlowStep."
                )

            self.output_keys.add(step.output_key)
            self.names.add(step.name)

            flowstep_class = step.__class__.__name__
            if flowstep_class == "FlowStep":
                self.input_keys.update(step.prompt_template.variables)
            elif flowstep_class == "ChatFlowStep":
                self.input_keys.update(
                    step.system_prompt_template.variables,
                    step.message_prompt_template.variables,
                    {step.message_key},
                )
            elif flowstep_class == "FunctionalFlowStep":
                self.input_keys.update(step.required_keys)

    def _check_all_input_keys_available(self, user_inputs):
        """
        Checks that all required input keys are available.

        Args:
            user_inputs (dict): The set of input keys provided by the user.

        Raises:
            ValueError: If not all required input keys are covered.
        """

        print(self.output_keys.union(user_inputs.keys()))
        print(self.input_keys)
        if not self.input_keys.issubset(self.output_keys.union(user_inputs.keys())):
            missing_inputs = self.input_keys.difference(
                self.output_keys.union(user_inputs.keys())
            )
            raise ValueError(f"Some flowsteps have missing inputs: {missing_inputs}")

    def start(self, **inputs):
        """
        Placeholder method for starting the flow.

        Args:
            inputs (dict): Inputs to the flow.

        Raises:
            NotImplementedError: If not implemented in a subclass of BaseFlow.
        """
        raise NotImplementedError
