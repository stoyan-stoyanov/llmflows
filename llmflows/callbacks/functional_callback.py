"""
This module provides FunctionalCallback class.

The FunctionalCallback class allows users to provide specific functions to be
executed at each stage of a FlowStep execution, without the need for subclassing.
"""

from typing import Callable, Optional, Any
from llmflows.callbacks.callback import BaseCallback

class FunctionalCallback(BaseCallback):
    """
    Represents a callback to be invoked at different stages of a FlowStep execution,
    with a specific function provided for each stage.

    Args:
        on_start_fn (Optional[Callable[[dict[str, Any]], None]]): The function to be
            invoked at the start stage.
        on_results_fn (Optional[Callable[[dict[str, Any]], None]]): The function to be
            invoked at the results stage.
        on_end_fn (Optional[Callable[[dict[str, Any]], None]]): The function to be
            invoked at the end stage.
        on_error_fn (Optional[Callable[[Exception], None]]): The function to be
            invoked in case of error.
    """
    def __init__(
        self,
        on_start_fn: Optional[Callable[[dict[str, Any]], None]] = None,
        on_results_fn: Optional[Callable[[dict[str, Any]], None]] = None,
        on_end_fn: Optional[Callable[[dict[str, Any]], None]] = None,
        on_error_fn: Optional[Callable[[Exception], None]] = None
    ):
        self.on_start_fn = on_start_fn
        self.on_results_fn = on_results_fn
        self.on_end_fn = on_end_fn
        self.on_error_fn = on_error_fn

    def on_start(self, inputs: dict[str, Any]):
        if self.on_start_fn is not None:
            self.on_start_fn(inputs)

    def on_results(self, results: dict[str, Any]):
        if self.on_results_fn is not None:
            self.on_results_fn(results)

    def on_end(self, execution_info: dict[str, Any]):
        if self.on_end_fn is not None:
            self.on_end_fn(execution_info)

    def on_error(self, error: Exception):
        if self.on_error_fn is not None:
            self.on_error_fn(error)
