from llmflows.flows.flowstep import BaseFlowStep
from llmflows.callbacks.callback import Callback
from typing import Callable, Any, Union


class FunctionalFlowstep(BaseFlowStep):
    def __init__(
        self,
        name: str,
        fn: Callable,
        required_keys: list[str],
        output_key: str,
        callbacks: list[Callback] = None
    ):
        super().__init__(name, output_key, callbacks)
        self.required_keys = set(required_keys)
        self.fn = fn

    def generate(self, inputs: dict[str, Any]) -> tuple[Any, Union[dict, None], Union[dict, None]]:
        return self.fn(inputs), None, None
