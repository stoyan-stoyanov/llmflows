"""
This module defines the PromptTemplate class which is used for generating 
prompts with variable values.
"""

# pylint: disable=too-few-public-methods

from string import Formatter


class PromptTemplate:
    """
    A class for generating prompts with variables.

    Args:
        prompt (str): The prompt string with variables.

    Attributes:
        prompt (str): The prompt string with variables.
        variables (List[str]): A list of variable names in the prompt string.
    """

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.variables = {
            fn for _, fn, _, _ in Formatter().parse(self.prompt) if fn is not None
        }

    def get_prompt(self, **kwargs: str) -> str:
        """
        Returns the prompt string with variables replaced by the provided values.

        Args:
            **kwargs: A dictionary of variable names and their values.

        Returns:
            The prompt string with variables replaced by the provided values.

        Raises:
            ValueError: If the provided variables do not match the defined ones.
        """
        if not self.variables:
            return self.prompt

        if set(kwargs.keys()) != self.variables:
            raise ValueError("The provided variables do not match the defined ones.")

        return self.prompt.format(**kwargs)
