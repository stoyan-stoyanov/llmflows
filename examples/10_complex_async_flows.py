# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create an async flow
multiple async flow steps.

The script creates an OpenAI language model (LLM) and several prompt templates, and 
uses them to define four async flow steps: one for generating a movie title, one for 
generating a song title for the movie, one for generating two main characters for 
the movie, and one for generating lyrics for a song based on the movie title and 
main characters. The script then connects the flow steps together to create a flow 
which runs flowsteps that have all required inputs in parallel.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""
import os
from llmflows.flows import AsyncFlow, AsyncFlowStep
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate
import asyncio

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

# Create LLM
open_ai_llm = OpenAI(api_key=open_ai_key)

# Create prompt templates
title_template = PromptTemplate("What is a good title of a movie about {topic}?")

song_template = PromptTemplate(
    "What is a good song title and a chorus of a soundtrack for a movie called"
    " {movie_title}?"
)

characters_template = PromptTemplate(
    "What are two main characters for a movie called {movie_title}?"
)

lyrics_template = PromptTemplate(
    "Write the lyrics for a movie song based on the provided title and chorus:\n"
    "{title_and_chorus}\n"
    "The main characters in tihe movie are:\n"
    "{main_characters}"
)

# Create flowsteps
flowstep1 = AsyncFlowStep(
    name="Flowstep 1",
    llm=open_ai_llm,
    prompt_template=title_template,
    output_key="movie_title",
)

flowstep2 = AsyncFlowStep(
    name="Flowstep 2",
    llm=open_ai_llm,
    prompt_template=song_template,
    output_key="title_and_chorus",
)

flowstep3 = AsyncFlowStep(
    name="Flowstep 3",
    llm=open_ai_llm,
    prompt_template=characters_template,
    output_key="main_characters",
)

flowstep4 = AsyncFlowStep(
    name="Flowstep 4",
    llm=open_ai_llm,
    prompt_template=lyrics_template,
    output_key="song_lyrics",
)

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)


# Create and run Flow
# Define async function to run the flow
async def run_flow():
    # Create and run Flow
    soundtrack_flow = AsyncFlow(flowstep1)
    result = await soundtrack_flow.start(topic="friendship", verbose=True)
    print(result)


# Run the flow in an event loop
asyncio.run(run_flow())
