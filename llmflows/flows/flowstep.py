# pylint: disable=R0801, R0913

"""
This module the FlowStep class, which can execute a language model, record
execution times, and optionally invoke callbacks on the results.
"""

from typing import Any, Union
from llmflows.llms.llm import BaseLLM
from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.callbacks.base_callback import BaseCallback
from llmflows.flows.base_flowstep import BaseFlowStep


class FlowStep(BaseFlowStep):
    """
    Represents a specific step in a Flow.

    A FlowStep executes a language model using a prompt template, records the execution
    time, and optionally invokes callback functions on the results.

    Attributes:
        name (str): The name of the flow step.
        output_key (str): The key for the output of the flow step.
        llm: The language model to be used in the flow step.
        prompt_template (PromptTemplate): Template for the prompt to be used with the 
            language model.
        callbacks (list[Callable]): Optional functions to be invoked with the results.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        prompt_template: PromptTemplate,
        output_key: str,
        callbacks:  Union[list[BaseCallback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.llm = llm
        self.prompt_template = prompt_template
        self.required_keys = prompt_template.variables

    def generate(self, inputs: dict[str, Any]) -> tuple[Any, Any, Any]:
        prompt = self.prompt_template.get_prompt(**inputs)
        return self.llm.generate(prompt)
