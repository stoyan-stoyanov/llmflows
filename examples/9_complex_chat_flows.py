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

from llmflows.flows import Flow, FlowStep, ChatFlowStep
from llmflows.llms import OpenAI, OpenAIChat
from llmflows.prompts import PromptTemplate

import json

# Create LLM
open_ai_llm = OpenAI()
chat_llm = OpenAIChat()

# Create prompt templates
title_template = PromptTemplate("What is a good title of a movie about {topic}?")
song_template = PromptTemplate(
    "What is a good song title of a soundtrack for a movie called {movie_title}?"
)
characters_template = PromptTemplate(
    "What are two main characters for a movie called {movie_title}?"
)
lyrics_template = PromptTemplate(
    "Write lyrics of a movie song called {song_title}. The main characters are"
    " {main_characters}"
)

critic_system_template = PromptTemplate("You are a music critic and write short reviews of song lyrics")
critic_message_template = PromptTemplate("Hey, what is your opinion on the following song: {song_lyrics}")

# Create flowsteps
flowstep1 = FlowStep(
    name="Flowstep 1",
    llm=OpenAI(),
    prompt_template=title_template,
    output_key="movie_title",
)

flowstep2 = FlowStep(
    name="Flowstep 2",
    llm=OpenAI(),
    prompt_template=song_template,
    output_key="song_title",
)

flowstep3 = FlowStep(
    name="Flowstep 3",
    llm=OpenAI(),
    prompt_template=characters_template,
    output_key="main_characters",
)

flowstep4 = FlowStep(
    name="Flowstep 4",
    llm=OpenAI(),
    prompt_template=lyrics_template,
    output_key="song_lyrics",
)

critics = []

for i in range(2):
    critics.append(
        ChatFlowStep(
            name=f"Critic Flowstep {i}",
            llm=OpenAIChat(),
            system_prompt_template=critic_system_template,
            message_prompt_template=critic_message_template,
            message_key="song_lyrics",
            output_key=f"song_review_{i}"
        )
    )

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)
flowstep4.connect(*critics)

# Create and run Flow
soundtrack_flow = Flow(flowstep1)
results = soundtrack_flow.execute(topic="friendship", verbose=True)
print(json.dumps(results, indent=4))
