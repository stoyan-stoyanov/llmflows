# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create a more complex flow
using multiple interdependent chat flow steps.

The script creates an ChatOpenAI LLM class and several prompt templates, and 
uses them to define four flow steps: one for generating a movie title, one for 
generating a song title for the movie, one for generating two main characters for 
the movie, and one for generating lyrics for a song based on the movie title and 
main characters. The script then connects the flow steps together to create a data 
processing pipeline.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""
import os
import json
from llmflows.flows import Flow, FlowStep, ChatFlowStep
from llmflows.llms import OpenAI, OpenAIChat, MessageHistory
from llmflows.prompts import PromptTemplate

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

# Create LLM
open_ai_llm = OpenAI(api_key=open_ai_key)

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

# Create flowsteps
flowstep1 = FlowStep(
    name="Flowstep 1",
    llm=open_ai_llm,
    prompt_template=title_template,
    output_key="movie_title",
)

flowstep2 = FlowStep(
    name="Flowstep 2",
    llm=open_ai_llm,
    prompt_template=song_template,
    output_key="song_title",
)

flowstep3 = FlowStep(
    name="Flowstep 3",
    llm=open_ai_llm,
    prompt_template=characters_template,
    output_key="main_characters",
)

flowstep4 = FlowStep(
    name="Flowstep 4",
    llm=open_ai_llm,
    prompt_template=lyrics_template,
    output_key="song_lyrics",
)

critics = []
critic_system_prompt = "You are a music critic who writes short reviews of song lyrics"
critic_message_template = PromptTemplate(
    "Hey, what is your opinion on the following song: {song_lyrics}"
)

for i in range(3):
    message_history = MessageHistory()
    message_history.system_prompt = critic_system_prompt

    critics.append(
        ChatFlowStep(
            name=f"Critic Flowstep {i+1}",
            llm=OpenAIChat(api_key=open_ai_key),
            message_history=message_history,
            message_prompt_template=critic_message_template,
            message_key="song_lyrics",
            output_key=f"song_review_{i+1}",
        )
    )

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)
flowstep4.connect(*critics)

# Create and run Flow
soundtrack_flow = Flow(flowstep1)
results = soundtrack_flow.start(topic="friendship", verbose=True)
print(json.dumps(results, indent=4))
