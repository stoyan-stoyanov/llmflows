# pylint: skip-file

"""
This script demonstrates how to create a flow pipeline using the llmflows
package.

The script creates three OpenAI language models (LLMs) and three prompt templates, 
and uses them to define three flow steps: one for generating a song title, one for 
generating lyrics for that title, and one for generating a heavy metal paraphrase of 
the lyrics. The script then connects the flow steps together to create a flow which is 
used to run all flowsteps and return the results with useful runtime data.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""
import os
from llmflows.flows import Flow, FlowStep
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

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
    llm=OpenAI(api_key=open_ai_key),
    prompt_template=title_prompt_template,
    output_key="song_title",
)

lyrics_flowstep = FlowStep(
    name="Lyrics Flowstep",
    llm=OpenAI(api_key=open_ai_key),
    prompt_template=lyrics_prompt_template,
    output_key="lyrics",
)

heavy_metal_flowstep = FlowStep(
    name="Heavy Metal Flowstep",
    llm=OpenAI(api_key=open_ai_key),
    prompt_template=heavy_metal_prompt_template,
    output_key="heavy_metal_lyrics",
)

# Connect flowsteps
title_flowstep.connect(lyrics_flowstep)
lyrics_flowstep.connect(heavy_metal_flowstep)

# Create the flow
songwriting_flow = Flow(title_flowstep)

# provide initial inputs for the flow and start it
result = songwriting_flow.start(topic="love", verbose=True)  
print(result)
