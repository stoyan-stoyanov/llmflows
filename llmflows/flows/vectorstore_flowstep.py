from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.llms.llm import BaseLLM
from llmflows.flows.flowstep import BaseFlowStep
from llmflows.callbacks.callback import Callback
from llmflows.vectorstores.vector_store import VectorStore
from llmflows.vectorstores.vector_doc import VectorDoc
from typing import Callable, Any, Union


class VectorStoreFlowStep(BaseFlowStep):
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
        callbacks: list[Callback] = None
    ):
        super().__init__(name, output_key, callbacks)
        self.embeddings_model = embeddings_model
        self.prompt_template = prompt_template
        self.required_keys = set(required_keys)
        self.vector_store = vector_store
        self.top_k = top_k
        self.apend_top_k = append_top_k

    def generate(self, inputs: dict[str, Any]) -> tuple[Any, Union[dict, None], Union[dict, None]]:

        """TODO: do we add prompts templates?
         Or do we just do whatever is the output of the parent flowstep?
         Then we need to ensure only a single flowstep is connected
         Or do we do search key like we did with a message key?

         """

        question = VectorDoc(doc="How was dark energy discovered?")
        embedded_question = self.embeddings_model.generate(question)
        search_results, call_data, config = self.vector_store.search(embedded_question, top_k=2)

        result = search_results[0]["metadata"]["text"]

        if self.apend_top_k:
            result = ''
            for i in range(self.top_k):
                result += (search_results[i+1]["metadata"]["text"] + "\n")

        return result, call_data, config
