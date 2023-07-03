# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create a complex data
processing pipeline using multiple flow steps.

The script creates an OpenAI language model (LLM) and several prompt templates, and
uses them to define four flow steps: one for generating a movie title, one for
generating a song title for the movie, one for generating two main characters for
the movie, and one for generating lyrics for a song based on the movie title and
main characters. The script then connects the flow steps together to create a data
processing pipeline.

Example:
    $ python 8_complex_flows.py
    {
        "movie_title": "The Last Unicorn",
        "song_title": "The Last Unicorn",
        "main_characters": "Amalthea and Schmendrick",
        "lyrics": "In a world of darkness and despair, two heroes rise to fight the
            evil that threatens to destroy them..."
    }

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""

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
    "paraphrase the following text in an ELI5 style:\n{answer}"
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
    output_key="answer",
)

eli5_flowstep = FlowStep(
    name="ELI5 Flowstep",
    llm=OpenAI(),
    prompt_template=eli5_template,
    output_key="eli5_answer",
)

# Connect flowsteps
q_flowstep.connect(vs_flowstep, answer_flowstep)
vs_flowstep.connect(answer_flowstep)
answer_flowstep.connect(eli5_flowstep)

# Create and run the Flow
qa_flow = Flow(q_flowstep)
results = qa_flow.execute(topic="wormholes", verbose=True)
print(results)
