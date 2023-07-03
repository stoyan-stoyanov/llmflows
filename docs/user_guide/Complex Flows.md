## TL;DR

```python
from llmflows.flows import Flow, FlowStep
from llmflows.llms import OpenAI
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

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)

# Create and run Flow
soundtrack_flow = Flow(flowstep1)
results = soundtrack_flow.execute(topic="friendship", verbose=True)
print(results)

```

## Guide
![Screenshot](assets/complex_flow.png)
Not implemented