## TL;DR

```python
import os
from llmflows.flows import Flow, FlowStep, VectorStoreFlowStep
from llmflows.llms import OpenAI, OpenAIEmbeddings
from llmflows.prompts import PromptTemplate
from llmflows.vectorstores import Pinecone


vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key=os.environ.get("PINECONE_API_KEY", "<YOUR-API-KEY>"),
    environment="us-west4-gcp-free",
)

# Create prompt templates
question_template = PromptTemplate("Ask a random question about {topic}")
vs_template = PromptTemplate("I have the following question: {question}")
response_template = PromptTemplate(
    "Answer the question based on the context.\n"
    "Context:\n"
    "{context}\n"
    "Question:\n"
    "{question}"
)
eli5_template = PromptTemplate(
    "paraphrase the following text in an ELI5 style:\n{response}"
)

# Create flowsteps
q_flowstep = FlowStep(
    name="Question Flowstep",
    llm=OpenAI(),
    prompt_template=question_template,
    output_key="question",
)

vs_flowstep = VectorStoreFlowStep(
    name="Vectorstore Flowstep",
    embeddings_model=OpenAIEmbeddings(),
    vector_store=vector_db,
    prompt_template=vs_template,
    output_key="context",
)

answer_flowstep = FlowStep(
    name="Response Flowstep",
    llm=OpenAI(),
    prompt_template=response_template,
    output_key="response",
)

eli5_flowstep = FlowStep(
    name="ELI5 Flowstep",
    llm=OpenAI(),
    prompt_template=eli5_template,
    output_key="eli5_response",
)

# Connect flowsteps
q_flowstep.connect(vs_flowstep, answer_flowstep)
vs_flowstep.connect(answer_flowstep)
answer_flowstep.connect(eli5_flowstep)

# Create and run Flow
qa_flow = Flow(q_flowstep)
results = qa_flow.execute(topic="wormholes", verbose=True)
print(results)

```

## Guide
![Screenshot](assets/LLMFlows_VectorStore_Flowstep.png)

Not implemented