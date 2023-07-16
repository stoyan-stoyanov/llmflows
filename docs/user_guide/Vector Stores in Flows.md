## TL;DR

```python
import os
from llmflows.flows import Flow, FlowStep, VectorStoreFlowStep
from llmflows.llms import OpenAI, OpenAIEmbeddings
from llmflows.prompts import PromptTemplate
from llmflows.vectorstores import Pinecone

vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key="<pinecone-api-key>",
    environment="us-west4-gcp-free",
)

openai_api_key = "<openai-api-key>"

openai_llm = OpenAI(api_key=openai_api_key)
openai_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

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
    llm=openai_llm,
    prompt_template=question_template,
    output_key="question",
)

vs_flowstep = VectorStoreFlowStep(
    name="Vectorstore Flowstep",
    embeddings_model=openai_embeddings,
    vector_store=vector_db,
    prompt_template=vs_template,
    output_key="context",
)

answer_flowstep = FlowStep(
    name="Response Flowstep",
    llm=openai_llm,
    prompt_template=response_template,
    output_key="response",
)

eli5_flowstep = FlowStep(
    name="ELI5 Flowstep",
    llm=openai_llm,
    prompt_template=eli5_template,
    output_key="eli5_response",
)

# Connect flowsteps
q_flowstep.connect(vs_flowstep, answer_flowstep)
vs_flowstep.connect(answer_flowstep)
answer_flowstep.connect(eli5_flowstep)

# Create and start the Flow
qa_flow = Flow(q_flowstep)
results = qa_flow.start(topic="wormholes", verbose=True)
print(results)

```
***
## Guide
!!! warning

    Before starting this guide, you have to complete the 
    [Vector Stores](Vector Databases.md) guide and upsert the VectorDocs based on the 
    Wikipedia texts into Pinecone.

In the previous two guides, we saw how to create vector embeddings for text documents, 
upload them to a vector database, and then use them for question answering.
In this guide, we will create a question-answering flow by utilizing the 
`VectorStoreFlowstep` class. We will try to achieve the same results as in the 
previous guide, but we will try to add one more step at the end to paraphrase the 
complex physics response in an ELI5 style:

![Screenshot](assets/LLMFlows_VectorStore_Flowstep.png)

Let's utilize the good old pattern for creating flows:

1. Decide how the flow should look like
2. Create the prompt templates
3. Create flow steps
4. Connect flow steps
5. Start the flow

Here are the prompt templates that we are going to need:

```python
from llmflows.prompts import PromptTemplate

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
```

Now let's create the flow steps based on the figure above. To add a vector store, we 
can utilize the `VectorStoreFlowStep` class which requires an embedding model, a vector 
database client, and a prompt template. 

The `VectorStoreFlowStep` constructs a prompt based on the prompt template and its 
inputs, creates embeddings of the prompt template, searches the vector database with the embeddings, and returns the result as a variable that other flow steps can consume.

```python
from llmflows.flows import Flow, FlowStep, VectorStoreFlowStep
from llmflows.llms import OpenAI, OpenAIEmbeddings

vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key="<pinecone-api-key>",
    environment="us-west4-gcp-free",
)

openai_api_key = "<openai-api-key>"

openai_llm = OpenAI(api_key=openai_api_key)
openai_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

q_flowstep = FlowStep(
    name="Question Flowstep",
    llm=openai_llm,
    prompt_template=question_template,
    output_key="question",
)

vs_flowstep = VectorStoreFlowStep(
    name="Vectorstore Flowstep",
    embeddings_model=openai_embeddings,
    vector_store=vector_db,
    prompt_template=vs_template,
    output_key="context",
)

answer_flowstep = FlowStep(
    name="Response Flowstep",
    llm=openai_llm,
    prompt_template=response_template,
    output_key="response",
)

eli5_flowstep = FlowStep(
    name="ELI5 Flowstep",
    llm=openai_llm,
    prompt_template=eli5_template,
    output_key="eli5_response",
)
```
Now we can connect the flow steps:
```python
q_flowstep.connect(vs_flowstep, answer_flowstep)
vs_flowstep.connect(answer_flowstep)
answer_flowstep.connect(eli5_flowstep)
```

And finally we can create the flow and start it:
```
qa_flow = Flow(q_flowstep)
results = qa_flow.start(topic="wormholes", verbose=True)
print(results)
```

In our next section we will show how you can use custom functions in flows. 

***
[:material-arrow-left: Previous: Question Answering](Question Answering.md){ .md-button }
[Next: Functional FlowSteps :material-arrow-right:](Functional FlowSteps.md){ .md-button }