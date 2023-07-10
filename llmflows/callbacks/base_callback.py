"""
This module provides the BaseCallback class.

The Callback class is designed to be subclassed by users who want to create their
own custom callbacks for different stages of a FlowStep execution.
"""

from typing import Any


class BaseCallback:
    """
    Represents a callback to be invoked at different stages of a FlowStep execution.

    The user can subclass this and override the methods corresponding to
    the stages where they want their callback logic to be executed.
    """
    def on_start(self, inputs: dict[str, Any]):
        """
        Method invoked at the start of the FlowStep's execution. Can be overridden
        for custom logic.

        Args:
            inputs (dict[str, Any]): Inputs provided to the FlowStep at the start.
        """

    def on_results(self, results: dict[str, Any]):
        """
        Method invoked when the FlowStep produces results. Can be overridden for
        custom logic.

        Args:
            results (dict[str, Any]): Results produced by the FlowStep's execution.
        """

    def on_end(self, execution_info: dict[str, Any]):
        """
        Method invoked at the end of the FlowStep's execution. Can be overridden for
        custom logic.

        Args:
            execution_info (dict[str, Any]): Information about the FlowStep's execution.
        """

    def on_error(self, error: Exception):
        """
        Method invoked when an error occurs during the FlowStep's execution. Can be 
        overridden for custom error handling.

        Args:
            error (Exception): The error that occurred during execution.
        """
