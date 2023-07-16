# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create a complex flow with
multiple flow steps using chat LLMs together with vector stores.

The script uses the OpenAI and OpenAIChat classes together with several prompt templates
to generate a question related to the information stored in the vector store.
The question is then passed to the vector store to retrieve the most similar context. 
After the context is retrieved, it is passed to the next flowstep together with the 
initial question. The next flowstep then generates an answer to the question based on
the context. The answer is then passed to the next flowstep which paraphrases the
answer in an ELI5 style.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""

import os
from llmflows.flows import Flow, FlowStep, VectorStoreFlowStep
from llmflows.llms import OpenAI, OpenAIEmbeddings
from llmflows.prompts import PromptTemplate
from llmflows.vectorstores import Pinecone

openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

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
    "paraphrase the following text in an ELI5 style:\n{answer}"
)

# Create flowsteps
q_flowstep = FlowStep(
    name="Question Flowstep",
    llm=OpenAI(api_key=openai_api_key),
    prompt_template=question_template,
    output_key="question",
)

vs_flowstep = VectorStoreFlowStep(
    name="Vectorstore Flowstep",
    embeddings_model=OpenAIEmbeddings(api_key=openai_api_key),
    vector_store=vector_db,
    prompt_template=vs_template,
    output_key="context",
)

answer_flowstep = FlowStep(
    name="Response Flowstep",
    llm=OpenAI(api_key=openai_api_key),
    prompt_template=response_template,
    output_key="answer",
)

eli5_flowstep = FlowStep(
    name="ELI5 Flowstep",
    llm=OpenAI(api_key=openai_api_key),
    prompt_template=eli5_template,
    output_key="eli5_answer",
)

# Connect flowsteps
q_flowstep.connect(vs_flowstep, answer_flowstep)
vs_flowstep.connect(answer_flowstep)
answer_flowstep.connect(eli5_flowstep)

# Create and run the Flow
qa_flow = Flow(q_flowstep)
results = qa_flow.start(topic="wormholes", verbose=True)
print(results)
