# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create an async flow
with multiple async flow steps using chat LLMs.

The script uses the OpenAI and OpenAIChat classes together with several prompt 
templates to define four async flow steps: one for generating a movie title, one for 
generating a song title for the movie, one for generating two main characters for 
the movie, and one for generating lyrics for a song based on the movie title and 
main characters. In addition to this there are ten more critic flowsteps that write 
a review for the generated lyrics. The script connects the flow steps together to 
create a flow which runs flowsteps that have all required inputs in parallel.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the OpenAI API.
"""
import os
from llmflows.flows import AsyncFlow, AsyncFlowStep, AsyncChatFlowStep
from llmflows.llms import OpenAI, OpenAIChat, MessageHistory
from llmflows.prompts import PromptTemplate
import asyncio
import json

openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")
openai_llm = OpenAI(api_key=openai_api_key)

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
flowstep1 = AsyncFlowStep(
    name="Flowstep 1",
    llm=openai_llm,
    prompt_template=title_template,
    output_key="movie_title",
)

flowstep2 = AsyncFlowStep(
    name="Flowstep 2",
    llm=openai_llm,
    prompt_template=song_template,
    output_key="song_title",
)

flowstep3 = AsyncFlowStep(
    name="Flowstep 3",
    llm=openai_llm,
    prompt_template=characters_template,
    output_key="main_characters",
)

flowstep4 = AsyncFlowStep(
    name="Flowstep 4",
    llm=openai_llm,
    prompt_template=lyrics_template,
    output_key="song_lyrics",
)

critics = []
critic_system_prompt = "You are a music critic who writes short reviews of song lyrics"
critic_message_template = PromptTemplate(
    "Hey, what is your opinion on the following song: {song_lyrics}"
)

for i in range(10):
    message_history = MessageHistory()
    message_history.system_prompt = critic_system_prompt

    critics.append(
        AsyncChatFlowStep(
            name=f"Critic Flowstep {i}",
            llm=OpenAIChat(api_key=openai_api_key),
            message_history=message_history,
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
async def run_flow():
    # Create and run Flow
    soundtrack_flow = AsyncFlow(flowstep1)
    result = await soundtrack_flow.start(topic="friendship", verbose=True)
    print(json.dumps(result, indent=4))

# Run the flow in an event loop
asyncio.run(run_flow())
