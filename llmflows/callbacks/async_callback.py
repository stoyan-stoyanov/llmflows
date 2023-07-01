"""
This module provides the AsyncCallback and AsyncFunctionalCallback classes.

The AsyncCallback class is designed to be subclassed by users who want to create their
own custom callbacks for different stages of a FlowStep execution.

The AsyncFunctionalCallback class allows users to provide specific functions to be
executed at each stage of a FlowStep execution, without the need for subclassing.
"""

from typing import Callable, Awaitable, Optional, Any


class AsyncCallback:
    """
    Represents a callback to be invoked at different stages of a FlowStep execution.

    The user can subclass this and override the methods corresponding to
    the stages where they want their callback logic to be executed.
    """
    async def on_start(self, inputs: dict[str, Any]):
        """
        Method invoked at the start of the FlowStep's execution. Can be overridden
        for custom logic.

        Args:
            inputs (dict[str, Any]): Inputs provided to the FlowStep at the start.
        """

    async def on_results(self, results: dict[str, Any]):
        """
        Method invoked when the FlowStep produces results. Can be overridden for
        custom logic.

        Args:
            results (dict[str, Any]): Results produced by the FlowStep's execution.
        """

    async def on_end(self, execution_info: dict[str, Any]):
        """
        Method invoked at the end of the FlowStep's execution. Can be overridden for
        custom logic.

        Args:
            execution_info (dict[str, Any]): Information about the FlowStep's execution.
        """

    async def on_error(self, error: Exception):
        """
        Method invoked when an error occurs during the FlowStep's execution. Can be
        overridden for custom error handling.

        Args:
            error (Exception): The error that occurred during execution.
        """


class AsyncFunctionalCallback(AsyncCallback):
    """
    Represents a callback to be invoked at different stages of a FlowStep execution,
    with a specific asynchronous function provided for each stage.

    Args:
        on_start_fn (Optional[Callable[[dict[str, Any]], Awaitable[None]]]): The 
            asynchronous function to be invoked at the start stage.
        on_results_fn (Optional[Callable[[dict[str, Any]], Awaitable[None]]]): The 
            asynchronous function to be invoked at the results stage.
        on_end_fn (Optional[Callable[[dict[str, Any]], Awaitable[None]]]): The 
            asynchronous function to be invoked at the end stage.
        on_error_fn (Optional[Callable[[Exception], Awaitable[None]]]): The 
            asynchronous function to be invoked in case of error.
    """
    def __init__(
        self,
        on_start_fn: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
        on_results_fn: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
        on_end_fn: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
        on_error_fn: Optional[Callable[[Exception], Awaitable[None]]] = None
    ):
        self.on_start_fn = on_start_fn
        self.on_results_fn = on_results_fn
        self.on_end_fn = on_end_fn
        self.on_error_fn = on_error_fn

    async def on_start(self, inputs: dict[str, Any]):
        if self.on_start_fn:
            await self.on_start_fn(inputs)

    async def on_results(self, results: dict[str, Any]):
        if self.on_results_fn:
            await self.on_results_fn(results)

    async def on_end(self, execution_info: dict[str, Any]):
        if self.on_end_fn:
            await self.on_end_fn(execution_info)

    async def on_error(self, error: Exception):
        if self.on_error_fn:
            await self.on_error_fn(error)
