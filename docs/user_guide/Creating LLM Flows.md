## TL;DR

```python
from llmflows.flows.flow import Flow
from llmflows.flows.flowstep import FlowStep
from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate


# Create prompt templates
title_prompt_template = PromptTemplate(
    prompt="What is a good title of a song about {topic}"
)
lyrics_prompt_template = PromptTemplate(
    prompt="Write me the lyrics for a song with a title {song_title}"
)
heavy_metal_prompt_template = PromptTemplate(
    prompt="paraphrase the following lyrics in the heavy metal style: {lyrics}"
)

# Create flowsteps
title_flowstep = FlowStep(
    name="Title Flowstep",
    llm=OpenAI(),
    prompt_template=title_prompt_template,
    output_key="song_title",
)

lyrics_flowstep = FlowStep(
    name="Lyrics Flowstep",
    llm=OpenAI(),
    prompt_template=lyrics_prompt_template,
    output_key="lyrics",
)

heavy_metal_flowstep = FlowStep(
    name="Heavy Metal Flowstep",
    llm=OpenAI(),
    prompt_template=heavy_metal_prompt_template,
    output_key="heavy_metal_lyrics",
)

# Connect flowsteps
title_flowstep.connect(lyrics_flowstep)
lyrics_flowstep.connect(heavy_metal_flowstep)

# Create and run Flow
songwriting_flow = Flow(title_flowstep)
songwriting_flow.execute(topic="love")  # provide initial data for the flow

```

## Guide
Not implemented