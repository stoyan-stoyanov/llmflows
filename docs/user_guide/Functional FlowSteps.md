## TL;DR

```python

from llmflows.flows import Flow, ChatFlowStep, FunctionalFlowStep
from llmflows.llms import OpenAIChat
from llmflows.prompts import PromptTemplate


def capitalize_first_letters(lyrics: str) -> str:
    """Capitalize the first letter of each word in a string."""
    return lyrics.title()


# Create flowsteps
title_flowstep = ChatFlowStep(
    name="Title Flowstep",
    llm=OpenAIChat(),
    system_prompt_template=PromptTemplate("You write song titles"),
    message_prompt_template=PromptTemplate("Write a good song title about {topic}?"),
    message_key="topic",
    output_key="song_title",
)

lyrics_flowstep = ChatFlowStep(
    name="Lyrics Flowstep",
    llm=OpenAIChat(),
    system_prompt_template=PromptTemplate("You write song lyrics"),
    message_prompt_template=PromptTemplate(
        "Write the lyrics of a song titled {song_title}"
    ),
    message_key="song_title",
    output_key="lyrics",
)

capitalizer_flowstep = FunctionalFlowStep(
    name="Capitalizer Flowstep",
    flowstep_fn=capitalize_first_letters,
    output_key="capitalized_lyrics",
)

# Connect flowsteps
title_flowstep.connect(lyrics_flowstep)
lyrics_flowstep.connect(capitalizer_flowstep)

# Create and run Flow
songwriting_flow = Flow(title_flowstep)
result, _, _ = songwriting_flow.start(topic="love", verbose=True)
print(result)
```
## Guide

In the previous guide, we looked into Vector Stores and how to use them in flows with the help of the `VectorStoreFlowstep`.

In this guide, we will introduce another type of flow step - the `FunctionalFlowStep` class. 

Sometimes we don't need to call an LLM to manipulate text. In some cases, we need a simple manipulation - maybe utilizing some built-in Python function or a regular expression. For situations like this LLMFlows provides the `FunctionalFlowStep` class. 

Let's build something similar to one of the early examples where we created a song title and its lyrics based on a topic. 

We will use ChatFlowstep and FunctionalFlowStep in our flow, so let's import it.

```python
from llmflows.flows import Flow, ChatFlowStep, FunctionalFlowStep
from llmflows.llms import OpenAIChat
from llmflows.prompts import PromptTemplate
```

When creating functional flow steps, we need to define the function that the flowstep will run. 
Let's say we want to capitalize each word in our song lyrics for some reason. To do so, we can create the following function:

```python
def capitalize_first_letters(lyrics: str) -> str:
    """Capitalize the first letter of each word in a string."""
    return lyrics.title()

```

Now that we have the function we will need for our `FunctionalFlowStep`, let's create the actual flow steps.

```python
# Create flowsteps
title_flowstep = ChatFlowStep(
    name="Title Flowstep",
    llm=OpenAIChat(),
    system_prompt_template=PromptTemplate("You write song titles"),
    message_prompt_template=PromptTemplate("Write a good song title about {topic}?"),
    message_key="topic",
    output_key="song_title",
)

lyrics_flowstep = ChatFlowStep(
    name="Lyrics Flowstep",
    llm=OpenAIChat(),
    system_prompt_template=PromptTemplate("You write song lyrics"),
    message_prompt_template=PromptTemplate(
        "Write the lyrics of a song titled {song_title}"
    ),
    message_key="song_title",
    output_key="lyrics",
)

capitalizer_flowstep = FunctionalFlowStep(
    name="Capitalizer Flowstep",
    flowstep_fn=capitalize_first_letters,
    output_key="capitalized_lyrics",
)
```


We only need to create the capitalizer flowstep to pass the function we defined earlier. 

LLMFlows will determine the required inputs to the function and ensure they exist in the flow as user-provided variables or output keys from other flow steps. If the function has missing input variables LLMFlows will raise a `ValueError` just like other flow steps.

!!!info

    Functions used in `FunctionalFlowStep` should only receive and return string variables.
    


Finally, let's initialize and start the flow.

```python
# Create and run Flow
songwriting_flow = Flow(title_flowstep)
result, _, _ = songwriting_flow.start(topic="love", verbose=True)
print(result)
```
Let's check the results:

```commandline
(Verse 1)
In The Stillness Of The Night,
Underneath The Moon'S Soft Light,
I Feel Your Presence By My Side,
A Love So Pure, It'S Hard To Hide.

(Pre-Chorus)
Our Hearts Beat As One,
Our Souls Forever Intertwined,
In This Dance We'Ve Just Begun,
Our Love, Eternal, Never Confined.

(Chorus)
Eternal Embrace, Our Love Will Never Fade,
Through The Storms And The Darkest Days,
In Your Arms, I Find My Solace,
Forever Entwined In This Eternal Embrace.
```

The result looks exactly as we want it. The output of the Lyrics flowstep was passed to the capitalizer flowstep, and our `capitalize_first_letters` function did its job!

In the following guide, we will learn how to use callback functions in flow steps.

***
[:material-arrow-left: Previous: Vector Stores in Flows](Vector Stores in Flows.md){ .md-button }
[Next: Callbacks :material-arrow-right:](Callbacks.md){ .md-button }