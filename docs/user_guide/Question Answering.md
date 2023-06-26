## TL;DR

```python
from llmflows.llms.openai_embeddings import OpenAIEmbeddings
from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate
from llmflows.vectorstores.vector_doc import VectorDoc
from llmflows.vectorstores.pinecone import Pinecone
import os

"""
Before starting this tutorial go and create an index in Pinecone with dimension of 1536 
(the default dimension or openai's embeddings)
"""

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "<YOUR-API-KEY>")

# Create embeddings LLM
embeddings_llm = OpenAIEmbeddings()

# initialize Pinecone
vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free",
)

# Define a question, create a question VectorDoc and create it's embeddings
question = VectorDoc(doc="How was dark energy discovered?")
embedded_question = embeddings_llm.generate(question)

# Search Pinecone with the question embedding to find the document with the
# most-relevant text
search_result = vector_db.search(embedded_question, top_k=2)
context = search_result[0]["metadata"]["text"]

# Provide the most-relevant document text to a llm and use the text as a context
# to generate the final answer
llm = OpenAI()
prompt_template = PromptTemplate(
    prompt=(
        "Answer the question based on the"
        " context.\nContext:\n{context}\nQuestion:\n{question}"
    )
)

llm_prompt = prompt_template.get_prompt(question=question.doc, context=context)
print(llm_prompt)

final_answer = llm.generate(llm_prompt)
print("Final answer:", final_answer)

```

## Guide
Not implemented