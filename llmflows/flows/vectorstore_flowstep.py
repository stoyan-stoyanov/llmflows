"""
This module provides the `VectorStoreFlowStep` class which extends the `BaseFlowStep` 
class.

This class represents a flow step that uses its prompt to search for a vector store.
Each instance of this class will be initialized with a specific vector store, 
embeddings model, prompt template, and other attributes.
"""

from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.llms.llm import BaseLLM
from llmflows.flows.flowstep import BaseFlowStep
from llmflows.callbacks.callback import Callback
from llmflows.vectorstores.vector_store import VectorStore
from llmflows.vectorstores.vector_doc import VectorDoc
from typing import Any, Union


class VectorStoreFlowStep(BaseFlowStep):
    """
    Represents a flowstep that uses a prompt to search for a vector store.

    The VectorStoreFlowstep uses the prompt template and the inputs to create a prompt,
    then uses the embeddings model to embed the prompt, and finally uses the vector
    store to search for similar vectors.

    If the `append_top_k` attribute is set to True, the top_k results will be appended
    in the final result

    Attributes:
        name (str): The name of the flow step.
        vector_store (VectorStore): The vector store instance to use.
        embeddings_model (BaseLLM): The embeddings model instance to use.
        prompt_template (PromptTemplate): The prompt template to use.
        required_keys (list[str]): A list of required keys.
        output_key (str): The key to use for the output.
        top_k (int, optional): The number of top results to return. Defaults to 1.
        append_top_k (bool, optional): Whether to append top_k results. Defaults to
            False.
        callbacks (list[Callback], optional): List of callback instances. Defaults to
            None.
    """

    def __init__(
        self,
        name: str,
        vector_store: VectorStore,
        embeddings_model: BaseLLM,
        prompt_template: PromptTemplate,
        required_keys: list[str],
        output_key: str,
        top_k: int = 1,
        append_top_k: bool = False,
        callbacks: Union[list[Callback], None] = None,
    ):
        super().__init__(name, output_key, callbacks)
        self.embeddings_model = embeddings_model
        self.prompt_template = prompt_template
        self.required_keys = set(required_keys)
        self.vector_store = vector_store
        self.top_k = top_k
        self.append_top_k = append_top_k

    def generate(
        self, inputs: dict[str, Any]
    ) -> tuple[Any, Union[dict, None], Union[dict, None]]:
        question = VectorDoc(doc=self.prompt_template.get_prompt(**inputs))
        embedded_question = self.embeddings_model.generate(question)
        search_results, call_data, config = self.vector_store.search(
            embedded_question, top_k=2
        )

        result = search_results[0]["metadata"]["text"]

        if self.append_top_k:
            result = ""
            for i in range(self.top_k):
                result += search_results[i + 1]["metadata"]["text"] + "\n"

        return result, call_data, config
