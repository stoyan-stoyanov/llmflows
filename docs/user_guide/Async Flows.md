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


# Create and run Flow
async def run_flow():
    soundtrack_flow = AsyncFlow(flowstep1)
    result = await soundtrack_flow.execute(topic="friendship", verbose=True)
    print(json.dumps(result, indent=4))

# Run the flow in an event loop
asyncio.run(run_flow())

```
***
## Guide

In the previous guide we tried to create a more complex flow and rely on the `Flow` class to figure out the dependencies, execute the flowsteps in the right order, and track information related to the execution of each flowsteps. One think we haven't touched upon so far is parallel execution. 
Obviously with a DAG-like flow we will have cases where multiple flowsteps should be able to run in parallel. 

In this guide we will introduce another flavor of flows and flowsteps - the `AsyncFlow`, and `AsyncFlowstep` classes. We will also introduce the `AsyncChatFlowstep` class that allows us to use chat LLMs in flows.

Let's try to reproduce the example from the previous page but also tweak it a bit. 

!!! note

    If you haven't checked the previous example we highly recommend giving it a try
    before continuing this example.

In the previous example we imagined we want to build an app that can generate a movie title, a movie song title based on the movie title, write a summary for the two main characters of the movie and finally create song lyrics based on movie title, song title, and the two characters.

If we take a look closely the "Song Title Flowstep", and "Characters Flowstep" can be executed in parallel since they only rely on the output of the "Movie Title Flowstep"

Let's further extend the example by adding song critics at the end of the flow. After the song lyrics are generated we would like to have five critics write an opinion on the lyrics. Obviously, this is another opportunity for concurrent execution. 

At the end of the previous guide we also discussed a recipe for structuring our code so let's try to follow it. 

Since we already discussed how the flow will look like let's start with creating our prompt templates:

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

Note that after creating the four flowsteps from our previous example we create a list of five "Critic Flowsteps". 

Now let's connect all the flowsteps together:
```python
# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)
flowstep4.connect(*critics)
```

Finally we can define and execute the whole flow. Since we use the async flavour of the flow and flowstep classes and they are built with async functins we will have to use `asyncio` to be able to run the flow in an event loop:

```python
import asyncio
import json

# Create the AsyncFlow
async def run_flow():
    soundtrack_flow = AsyncFlow(flowstep1)
    result = await soundtrack_flow.execute(topic="friendship", verbose=True)
    print(json.dumps(result, indent=4))

# Run the flow in an event loop
asyncio.run(run_flow())
```

There it is! When using the async classes, LLMFlows can run the flowsteps that already have all their required inputs available in parallel and therefore save significant amount of runtime.

Thank you for reading this far! This was our final guide on flows and flowsteps. We hope that you enjoyed reading the guide so far, and hopefully the information was easily digestable.

In the next guide we are going to look into vector dabases, and how we can use them to build useful applications.

***
[:material-arrow-left: Previous: Complex Flows](Complex Flows.md){ .md-button }
[Next: Vector Databases :material-arrow-right:](Vector Databases.md){ .md-button }