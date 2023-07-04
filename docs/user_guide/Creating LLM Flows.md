## TL;DR

```python
from llmflows.flows import Flow, FlowStep
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

# Create prompt templates
title_template = PromptTemplate("What is a good title of a song about {topic}")
lyrics_template = PromptTemplate("Write me the lyrics for a song with a title {song_title}")
heavy_metal_template = PromptTemplate("paraphrase the following lyrics: {lyrics}")

# Create flowsteps
title_flowstep = FlowStep(
    name="Title Flowstep",
    llm=OpenAI(),
    prompt_template=title_template,
    output_key="song_title",
)

lyrics_flowstep = FlowStep(
    name="Lyrics Flowstep",
    llm=OpenAI(),
    prompt_template=lyrics_template,
    output_key="lyrics",
)

heavy_metal_flowstep = FlowStep(
    name="Heavy Metal Flowstep",
    llm=OpenAI(),
    prompt_template=heavy_metal_template,
    output_key="heavy_metal_lyrics",
)

# Connect flowsteps
title_flowstep.connect(lyrics_flowstep)
lyrics_flowstep.connect(heavy_metal_flowstep)

# Create and run Flow
songwriting_flow = Flow(title_flowstep)
result = songwriting_flow.execute(topic="love")  # provide initial inputs for the flow
print(result)

```
***
## Guide
In the previous guides we went through the `LLM` and `PromptTemplate` abstractions and we introduced two common patterns when building LLM-powered apps. 
In the first pattern we used prompt templates to create dynamic prompts and in the second pattern we used the output of an LLM call as an input to another LLM.

In this guide we are going to introduce two new main abstractions of LLMFlows - Flowsteps and Flows.

!!! info
    The `Flow` and `FlowStep` classes can be imported from `llmflows.flows`

Flows and FlowSteps are the bread and butter of LLMFlows. They are simple but powerful abstractions 
that allow for the creation of arbitrary DAG-like connections between LLMs while managing dependencies, order of execution and prompt variables.

Let's try to reproduce the previous example with using Flows and Flowsteps. As a start let's define the same templates that we are already familiar with:
```python
from llmflows.prompts import PromptTemplate

title_template = PromptTemplate("What is a good title of a song about {topic}")
lyrics_template = PromptTemplate("Write me the lyrics for a song with a title {song_title}")
heavy_metal_template = PromptTemplate("paraphrase the following lyrics: {lyrics}")
```

Once we have the prompt templates, we can start defining the flowsteps:
```python
from llmflows.flows import Flow, FlowStep

title_flowstep = FlowStep(
    name="Title Flowstep",
    llm=OpenAI(),
    prompt_template=title_template,
    output_key="song_title",
)

lyrics_flowstep = FlowStep(
    name="Lyrics Flowstep",
    llm=OpenAI(),
    prompt_template=lyrics_template,
    output_key="lyrics",
)

heavy_metal_flowstep = FlowStep(
    name="Heavy Metal Flowstep",
    llm=OpenAI(),
    prompt_template=heavy_metal_template,
    output_key="heavy_metal_lyrics",
)
```
In order to create a flowstep we have to provide the required parameters for the `FlowStep` class which include:

- name (must be unique)
- the LLM to be used within the flow
- the prompt template to be used when calling the LLM
- output_key (must be unique) which is treated as a prompt variable for other flowsteps

!!! question
    
    Q: What if I don't want to provide a prompt template? In many cases I can simply use a string instead.

    A: Makes sense! In this scenario, feel free to create a prompt template without any variables.


Once we have the FlowStep definitions we can connect the flowsteps in the order we want
```python
title_flowstep.connect(lyrics_flowstep)
lyrics_flowstep.connect(heavy_metal_flowstep)
```

Finally we can create the flow and run it. To create the `Flow` object we need to provide the first `FlowStep` and to run it 
we have to use the `execute()` method and provide any required initial inputs.

```python
songwriting_flow = Flow(title_flowstep)
result = songwriting_flow.execute(topic="love", verbose=True)  # provide initial inputs for the flow
```

This is it!

Although, this might initially seem like a lot of extra abstractions to achieve the same functionality as in the previous examples, if we start inspecting the results
we will start seeing some advantages of using Flows and FlowSteps.

After running all FlowSteps, the Flow will return detailed results for the execution of each individual FlowStep:
```python
print(result)
```

```python
{
    "Title Flowstep": {...},
    "Lyrics Flowstep": {...},
    "Heavy Metal Flowstep": {...}
}
```

Let's take a look at what happend when executing the "Title Flowstep":
```python
print(result["Title Flowstep"])
```
```json
{
   "start_time":"2023-07-03T15:23:47.490368",
   "prompt_inputs":{
      "topic":"love"
   },
   "generated":"\n\n\"Love Is All Around Us\"",
   "call_data":{
      "raw_outputs":{
         "<OpenAIObject text_completion id=cmpl-7YMFPac1MKUje0jIyk4adkYssk4rQ at 0x107946f90> JSON":{
            "choices":[
               {
                  "finish_reason":"stop",
                  "index":0,
                  "logprobs":null,
                  "text":"\n\n\"Love Is All Around Us\""
               }
            ],
            "created":1688423027,
            "id":"cmpl-7YMFPac1MKUje0jIyk4adkYssk4rQ",
            "model":"text-davinci-003",
            "object":"text_completion",
            "usage":{
               "completion_tokens":9,
               "prompt_tokens":10,
               "total_tokens":19
            }
         }
      },
      "retries":0,
      "prompt_template":"What is a good title of a song about {topic}",
      "prompt":"What is a good title of a song about love"
   },
   "config":{
      "model_name":"text-davinci-003",
      "temperature":0.7,
      "max_tokens":500
   },
   "end_time":"2023-07-03T15:23:48.845386",
   "execution_time":1.355005416,
   "result":{
      "song_title":"\n\n\"Love Is All Around Us\""
   }
}

```
There is a lot to unpack here but after executing the flow we have a complete visibility of what happened at each flowstep.
By having this information we can answer questions such as:

- when was a particular flowstep executed?
- how much time it took?
- what were the input variables?
- what was the prompt template?
- how did the prompt look like?
- what was the exact configuration of the model?
- how many times did we retry the request?
- what was the raw data the API returned?
- how many tokens were used?
- what was the final result?

All of this ties to our philosphy of **"Simple, Explicit, and Transparent LLM apps"**. This information allows developers to have complete 
visibility and be able to easily log, debug and maintain LLM apps.

This, however, is not the only value that LLMFlows can provide. This example is great for the purposes of this guide but real-life applications are usually more complex.
Next, we will go deeper into more complex applications where Flows and FlowSteps really start to shine due to features like figuring out variable dependencies, and concurrent(async) execution.
***
[:material-arrow-left: Previous: Combining LLMs](Combining LLMs.md){ .md-button }
[Next: Complex Flows :material-arrow-right:](Complex Flows.md){ .md-button }