# pylint: disable=R0801, R0913

"""
This module the AsyncFlowStep class, which can execute a language model, record
execution times, and optionally invoke callbacks on the results. The async 
implementation allows async flowsteps to be executed in parallel if multiple flowsteps
have all the required inputs available.
"""

from typing import Any, Union
from llmflows.llms.llm import BaseLLM
from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.callbacks.async_base_callback import AsyncBaseCallback
from llmflows.flows.async_base_flowstep import AsyncBaseFlowStep


class AsyncFlowStep(AsyncBaseFlowStep):
    """
    Represents a specific async step in an async Flow.

    An AsyncFlowStep executes a language model using a prompt template, records the 
    execution time, and optionally invokes callback functions on the results.
    Async Flowsteps can be executed in parallel in an AsyncFlow if all the required 
    inputs are available.

    Args:
        name (str): The name of the flow step.
        llm (BaseLLM): The language model to be used in the flow step.
        prompt_template (PromptTemplate): Template for the prompt to be used with the 
            language model.
        callbacks Union[list[AsyncBaseCallback], None]: Callbacks to be invoked during 
            the flowstep execution.

    Attributes:
        llm (BaseLLM): The language model to be used in the flow step.
        prompt_template (PromptTemplate): Template for the prompt to be used with the 
            language model.
        required_keys (set[str]): The keys required for the flow step to execute.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        prompt_template: PromptTemplate,
        output_key: str,
        callbacks: Union[list[AsyncBaseCallback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.prompt_template = prompt_template
        self.required_keys = prompt_template.variables

    async def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        prompt = self.prompt_template.get_prompt(**inputs)
        return await self.llm.generate_async(prompt)
