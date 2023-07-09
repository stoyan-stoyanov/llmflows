## TL;DR

```python

from llmflows.flows import AsyncFlow, AsyncFlowStep, AsyncChatFlowStep
from llmflows.llms import OpenAI, OpenAIChat
from llmflows.prompts import PromptTemplate
import asyncio
import json

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
critic_system_template = PromptTemplate(
    "You are a music critic and write short reviews of song lyrics"
)
critic_message_template = PromptTemplate(
    "Hey, what is your opinion on the following song: {song_lyrics}
)

# Create flowsteps
flowstep1 = AsyncFlowStep(
    name="Flowstep 1",
    llm=OpenAI(),
    prompt_template=title_template,
    output_key="movie_title",
)

flowstep2 = AsyncFlowStep(
    name="Flowstep 2",
    llm=OpenAI(),
    prompt_template=song_template,
    output_key="song_title",
)

flowstep3 = AsyncFlowStep(
    name="Flowstep 3",
    llm=OpenAI(),
    prompt_template=characters_template,
    output_key="main_characters",
)

flowstep4 = AsyncFlowStep(
    name="Flowstep 4",
    llm=OpenAI(),
    prompt_template=lyrics_template,
    output_key="song_lyrics",
)

critics = []

for i in range(5):
    critics.append(
        AsyncChatFlowStep(
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


# Create and start Flow
async def start_flow():
    soundtrack_flow = AsyncFlow(flowstep1)
    result = await soundtrack_flow.start(topic="friendship", verbose=True)
    print(json.dumps(result, indent=4))


# Run the flow in an event loop
asyncio.run(start_flow())

```
***
## Guide

In the previous guide, we tried to create a more complex flow and rely on the `Flow` 
class to figure out the dependencies, run the flow steps in the correct order, and 
track information related to each flowstep in the flow. One thing we haven't touched 
upon so far is running flow steps in parallel. 

Representing a flow as a DAG creates opportunities to run multiple flow steps 
simultaneously as long as all their inputs are available. 

This guide will introduce another flavor of flows and flow steps - the `AsyncFlow` and 
`AsyncFlowstep` classes. We will also teach the `AsyncChatFlowstep` class that allows 
us to use chat LLMs in flows.

Let's reproduce the example from the previous page and slightly tweak it. 

!!! note

    If you haven't checked the previous example, we highly recommend giving it a try
    before continuing this example.

In the previous example, we imagined we wanted to build an app that could generate a 
movie title, a movie song title based on the movie title, write a summary for the two 
main characters of the movie and finally create song lyrics based on the movie title, 
song title, and the two characters.

If we take a look closely, the "Song Title Flowstep" and "Characters Flowstep" can be 
run in parallel since they only rely on the output of the "Movie Title Flowstep."

Let's further extend the example by adding song critics at the end of the flow. After 
we generate the song lyrics, we would like five critics to write an opinion on it. 
This is another opportunity for running flow steps in parallel. 

At the end of the previous guide, we also discussed a recipe for structuring our code, 
so let's try to follow it.  Here is a reminder:

1. Decide how the flow should look like
2. Create the prompt templates
3. Create flow steps
4. Connect flow steps
5. Start the flow

Since we already discussed how the flow will look, we can start with creating our 
prompt templates:

```python
from llmflows.prompts import PromptTemplate

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
critic_system_template = PromptTemplate(
    "You are a music critic and write short reviews of song lyrics"
)
critic_message_template = PromptTemplate(
    "Hey, what is your opinion on the following song: {song_lyrics}
)
```
The next step is to create our flowsteps:
```python
from llmflows.flows import AsyncFlow, AsyncFlowStep, AsyncChatFlowStep
from llmflows.llms import OpenAI, OpenAIChat

# Create flowsteps
flowstep1 = AsyncFlowStep(
    name="Movie Title Flowstep",
    llm=OpenAI(),
    prompt_template=title_template,
    output_key="movie_title",
)

flowstep2 = AsyncFlowStep(
    name="Song Title Flowstep",
    llm=OpenAI(),
    prompt_template=song_template,
    output_key="song_title",
)

flowstep3 = AsyncFlowStep(
    name="Characters Flowstep",
    llm=OpenAI(),
    prompt_template=characters_template,
    output_key="main_characters",
)

flowstep4 = AsyncFlowStep(
    name="Lyrics Flowstep",
    llm=OpenAI(),
    prompt_template=lyrics_template,
    output_key="song_lyrics",
)

critics = []

for i in range(5):
    critics.append(
        AsyncChatFlowStep(
            name=f"Critic Flowstep {i}",
            llm=OpenAIChat(),
            system_prompt_template=critic_system_template,
            message_prompt_template=critic_message_template,
            message_key="song_lyrics",
            output_key=f"song_review_{i}"
        )
    )
```

Note that after creating the four flowsteps from our previous example we added a list 
of five "Critic Flowsteps". 

Now let's connect all the flowsteps:
```python
# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)
flowstep4.connect(*critics)
```

Finally, we can define and start the whole flow. Since we use the async flavor of the 
flow and flowstep classes, we will have to use `asyncio` to be able to run the flow in 
an event loop:

```python
import asyncio
import json


# Create the AsyncFlow
async def start_flow():
    soundtrack_flow = AsyncFlow(flowstep1)
    result = await soundtrack_flow.start(topic="friendship", verbose=True)
    print(json.dumps(result, indent=4))


# Run the flow in an event loop
asyncio.run(start_flow())
```

There it is! When using the async classes, LLMFlows can run the flow steps that already 
have all their required inputs in parallel and save significant runtime.

The following guide will look into vector databases and how to use them to build LLM 
applications.

***
[:material-arrow-left: Previous: LLM Flows](LLM Flows.md){ .md-button }
[Next: Vector Stores :material-arrow-right:](Vector Stores.md){ .md-button }