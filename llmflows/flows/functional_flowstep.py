"""
LLMFlow module for the `FunctionalFlowstep` class that can be used to run a given
function within a flow.
"""

import inspect
from typing import Callable, Any, Union
from llmflows.flows.flowstep import BaseFlowStep
from llmflows.callbacks.base_callback import BaseCallback


class FunctionalFlowStep(BaseFlowStep):
    """
    Represents a functional flow step that runs a function. The function must take
        a dictionary of strings as input and return a string(like regular flow steps)

    Args:
        name (str): The name of the flow step.
        fn (Callable): The function to run.
        required_keys (list[str]): A list of required keys.
        output_key (str): The key to use for the output.
        callbacks (list[Callback], optional): List of callback instances. Defaults to
            None.

    Attributes:
        required_keys (set[str]): The keys required for the flow step to run.
        fn (Callable[[dict[str, str]], str]): The function to be run.
    """

    def __init__(
        self,
        name: str,
        flowstep_fn: Callable[..., str],
        output_key: str,
        callbacks: Union[list[BaseCallback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.flowstep_fn = flowstep_fn
        self.required_keys = inspect.getfullargspec(self.flowstep_fn).args

    def generate(
        self, inputs: dict[str, Any]
    ) -> tuple[Any, Union[dict, None], Union[dict, None]]:
        """
        Executes the function with the provided inputs.

        Args:
            inputs (dict[str, Any]): Input parameters as a dictionary.

        Returns:
            The result of the function call, followed by two None values (for call
                data and config, which are not applicable in this case).
        """
        # Get the argument names of fn
        fn_args = inspect.getfullargspec(self.flowstep_fn).args

        # Create a new dictionary from inputs containing only the keys that fn requires
        filtered_inputs = {
            key: value for key, value in inputs.items() if key in fn_args
        }

        return self.flowstep_fn(**filtered_inputs), None, None
