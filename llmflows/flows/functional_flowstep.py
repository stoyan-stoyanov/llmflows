"""
LLMFlow module for the `FunctionalFlowstep` class that can be used to execute a given
function within a flow.
"""

from llmflows.flows.flowstep import BaseFlowStep
from llmflows.callbacks.base_callback import BaseCallback
from typing import Callable, Any, Union


class FunctionalFlowStep(BaseFlowStep):
    """
    Represents a functional flow step that executes a function.

    Args:
        name (str): The name of the flow step.
        fn (Callable): The function to execute.
        required_keys (list[str]): A list of required keys.
        output_key (str): The key to use for the output.
        callbacks (list[Callback], optional): List of callback instances. Defaults to 
            None.
    """
    def __init__(
        self,
        name: str,
        fn: Callable[[dict[str, str]], str],
        required_keys: list[str],
        output_key: str,
        callbacks: Union[list[BaseCallback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.required_keys = set(required_keys)
        self.fn = fn

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
        return self.fn(inputs), None, None
