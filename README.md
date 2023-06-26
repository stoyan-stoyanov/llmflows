# LLMFlows - Simple, Explicit and Transparent LLM Apps

<p align="center">
  <img style="width: 80%" src="docs/llmflows_last_logo.png" />
</p>

[![Twitter](https://img.shields.io/twitter/follow/LLMFlows?style=social)](https://twitter.com/LLMFlows)
![Pylint workflow](https://github.com/stoyan-stoyanov/llmflow/actions/workflows/pylint.yml/badge.svg)
![License](https://img.shields.io/github/license/stoyan-stoyanov/llmflow)
![PyPi](https://img.shields.io/pypi/v/llmflows)
![Stars](https://img.shields.io/github/stars/stoyan-stoyanov/llmflow?style=social)
![Release date](https://img.shields.io/github/release-date/stoyan-stoyanov/llmflow?style=social)

## About
LLMFlows is a simple and lightweight framework for building LLM-powered applications.

## Installation
You can quickly install LLMFlows with pip
```
pip install llmflows
```

## Philosophy

### Simple
We want to build a framework with a minimal set of classes that allows users to build powerful LLM-powered apps without compromising on capabilities.

### Explicit
We want to enable users to easily create complex flows of LLMs interacting with each other that have explicit and obvious structure.

### Transparent
Flows are traceable, executions are easily logged and none of the classes provided in LLMFlows have hidden prompts. Default prompts introduce unexpected behavior and behavior drift when models are updated. 

## Documentation
The full documentation for LLMFlows can be found here.

## Usage
Here is a minimal example of an LLM with a PromptTemplate:

```python

from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate

prompt_template = PromptTemplate(
   prompt="Generate a title for a 90s hip-hop song about {topic}."
)
llm_prompt = prompt_template.get_prompt(topic="friendship")

llm = OpenAI()
song_title = llm.generate(llm_prompt)

```

While this is a good example on how easy it is to use LLMFlows, real-world applications are more complex and have dependencies between prompts, and LLMs outputs. Let's take a look at such example. 
![Complex flow](docs/complex_flow.png)
With LLMFlows it's quite easy to reproduce this flow by utilizing the Flow and Flowstep classes. LLMFlows will figure out the dependencies and make sure each flowstep is executed only when the flowsteps it depends on are complete:

```python
from llmflows.flows.flow import Flow
from llmflows.flows.flowstep import FlowStep
from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate

# Create LLM
open_ai_llm = OpenAI()

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

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)

# Create and run Flow
soundtrack_flow = Flow(flowstep1)
soundtrack_flow.execute(topic="friendship")

```
In fact, LLMFlows provides async, and threaded classes so any complex DAG can be executed in parallel.
For more examples such as how to create question answering apps and web applications with Flask and FastAPI check our documentation.

## FAQ

### How is this different than langchain?
Langchain is a great library and LLMFlows has been been certainly inspired by it. However, our philosophy is a bit different. Langchian has a "chain for everything" philosophy and provides many classes that come with multiple LLM calls, logic, and built-in default prompts. While this is great for beginners and default use-cases, we feel this can be a bit overwhelming if users want to do anything "out of the ordinary". In contrast, we are focusing on providing as few building blocks as possible and having an easy to understand API while matching (and in some cases exceeding) the capabilities of langchain.

### You only have OpenAI wrappers but I want to use AcmeLLM?
We decided to release the library initially supporting only OpenAI LLMs but we have a roadmap and we will slowly add new wrappers around the most popular models. If you are willing to spend some time we are looking for contributors and maintainers

### You only support Pinecone and Redis vector DBs do you have plans to extend the list?
Yes! We will also add Redis, Elastic Search and other popular solutions over time. If you want to help us out check out our contribution section.

### Why can't I find any info related to document loaders?
For the time being we have decided not to implement document loaders for few reasons:
1. There are plenty of capable libraries like Llama-index and langchain that have tons of loaders.
2. We think it is awkward to mix document loaders together with LLM and prompt management libraries since usually document loading happens in separate pipelines and are not part of the LLM-powered app.
3. Real-life documents are messy. In our experience, no matter how many loaders are out there they will never cover all the specific use-cases.

While we are not going to invest time into document loaders we might decide to change direction if we get significant interest and contributors.

### What about agents?
We believe agents are the future of LLM-powered apps and we have a few basic examples in the repo. However, we are working on a agent-focused library built on top of llmflow.

## License
LLMFlows is covered by the MIT license. For more information check `LICENCE.md`.

## Contributing
Thank you for spending the time to read our README! If you like what you saw and are considering contributing please check CONTRIBUTING.md

## Links
Twitter

Documentation
