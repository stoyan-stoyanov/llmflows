## TL;DR

```python
import logging
from llmflows.flows import Flow, ChatFlowStep, FunctionalFlowStep
from llmflows.callbacks import FunctionalCallback
from llmflows.llms import OpenAIChat, MessageHistory
from llmflows.prompts import PromptTemplate


logging.getLogger().setLevel(logging.INFO)

openai_api_key = "<your-api-key>"

# Create functional flow step function 
def capitalize_first_letters(lyrics: str) -> str:
    """Capitalize the first letter of each word in a string."""
    return lyrics.title()


# Create functional callback functions
def logging_on_start(inputs: dict[str, str]):
    """Log the inputs at the start of a flowstep."""
    logging.info(f"Inputs: {inputs}") 


def logging_on_results(results: dict[str, str]):
    """Log the results at the results stage of a flowstep."""
    logging.info(f"Results: {results}")


# Create functional callback
logging_callback = FunctionalCallback(
    on_start_fn=logging_on_start,
    on_results_fn=logging_on_results
)

title_message_history = MessageHistory()
title_message_history.system_prompt = "You write song titles"

lyrics_message_history = MessageHistory()
title_message_history.system_prompt = "You write song lyrics"

# Create flowsteps
title_flowstep = ChatFlowStep(
    name="Title Flowstep",
    llm=OpenAIChat(api_key=openai_api_key),
    message_history=title_message_history,
    message_prompt_template=PromptTemplate("Write a good song title about {topic}?"),
    message_key="topic",
    output_key="song_title",
)

lyrics_flowstep = ChatFlowStep(
    name="Lyrics Flowstep",
    llm=OpenAIChat(api_key=openai_api_key),
    message_history=lyrics_message_history,
    message_prompt_template=PromptTemplate(
        "Write the lyrics of a song titled {song_title}"
    ),
    callbacks=[logging_callback],
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

***
## Guide
In the last guide, we saw how we could use the `FunctionalFlowStep` class within flows 
when we want to have functions that manipulate strings that don't rely on LLM calls. 

This guide introduces the `FunctionalCallback` class, which we can use to run callback 
functions at different stages of any flow step. 

!!! question

    **Q: What is the difference between `FuncionalFlowStep` and `FunctionalCallback`?**

    **A:** The `FunctionalFlowStep` class is a fully-fledged flow step used to 
    manipulate strings within flows. It gets input variables, generates an output based 
    on its `flowstep_function`, and has an `output_key` like any other flow step. On 
    the other hand, the `FunctionalCallback` class is a callback that can be passed to 
    any flow step and runs on certain events. Unlike the `FunctionalFlowStep`, the 
    `FunctionalCallback` class can't manipulate data within a `Flow`. It's primary 
    usage is for integrations with 3rd party solutions for tracing and logging.

!!! info

    You can import Callback classes from `llmflows.callbacks`

To understand better the difference between `FunctionalFlowStep` and 
`FunctionalCallback` let's take the same example from the previous guide where we had 
two LLMs generate a song title, song lyrics, and a functional flow step that 
capitalized the lyrics at the end. 

Let's define all the required functions.

```python
import logging
from llmflows.flows import Flow, ChatFlowStep, FunctionalFlowStep
from llmflows.callbacks import FunctionalCallback
from llmflows.llms import OpenAIChat, MessageHistory
from llmflows.prompts import PromptTemplate


logging.getLogger().setLevel(logging.INFO)

# Create functional flow step function 
def capitalize_first_letters(lyrics: str) -> str:
    """Capitalize the first letter of each word in a string."""
    return lyrics.title()


# Create functional callback functions
def logging_on_start(inputs: dict[str, str]):
    """Log the inputs at the start of a flowstep."""
    logging.info(f"Inputs: {inputs}") 


def logging_on_results(results: dict[str, str]):
    """Log the results at the results stage of a flowstep."""
    logging.info(f"Results: {results}")


# Create functional callback
logging_callback = FunctionalCallback(
    on_start_fn=logging_on_start,
    on_results_fn=logging_on_results
)
```

Like in the previous example, the `capitalize_first_letters()` function capitalizes the 
final lyrics' first letter. We also define two other functions that we pass to our 
`FunctionalCallback` class. 

!!! info

    Every callback class has four main methods that run at different stages of the flow 
    step":

    1. `on_start_fn` runs at the beginning of the flow. 
    2. `on_result_fn` that runs after the flow step results are computed. 
    3. `on_end_fn` runs right before the flow step ends.
    4. `on_error_fn` runs if there is any error within the flow step.

Now that we have the required functions, we can define the flow steps. Let's also pass 
the `FunctionalCallback` to the "Lyrics Flowstep". Now the two functions we defined 
above will run when at the start of the "Lyrics Flowstep" and when we get the results 
from the LLM. 

```python
title_message_history = MessageHistory()
title_message_history.system_prompt = "You write song titles"

lyrics_message_history = MessageHistory()
title_message_history.system_prompt = "You write song lyrics"

# Create flowsteps

openai_api_key = "<your-api-key>"

title_flowstep = ChatFlowStep(
    name="Title Flowstep",
    llm=OpenAIChat(api_key=openai_api_key),
    message_history=title_message_history,
    message_prompt_template=PromptTemplate("Write a good song title about {topic}?"),
    message_key="topic",
    output_key="song_title",
)

lyrics_flowstep = ChatFlowStep(
    name="Lyrics Flowstep",
    llm=OpenAIChat(api_key=openai_api_key),
    message_history=lyrics_message_history,
    message_prompt_template=PromptTemplate(
        "Write the lyrics of a song titled {song_title}"
    ),
    callbacks=[logging_callback],
    message_key="song_title",
    output_key="lyrics",
)

capitalizer_flowstep = FunctionalFlowStep(
    name="Capitalizer Flowstep",
    flowstep_fn=capitalize_first_letters,
    output_key="capitalized_lyrics",
)
```

Finally, let's connect the flow steps and start the flow:

```python
# Connect flowsteps
title_flowstep.connect(lyrics_flowstep)
lyrics_flowstep.connect(capitalizer_flowstep)

# Create and run Flow
songwriting_flow = Flow(title_flowstep)
result, _, _ = songwriting_flow.start(topic="love", verbose=True)
print(result)
```
Now when we run the flow we get the following outputs from the logger at the different 
stages of the "Lyrics Flowstep":

```commandline
INFO:root:Inputs: {'song_title': '"Eternal Flame of Love"'}

INFO:root:Results: 
(Verse 1)
In the darkness of the night, I saw a glimmering light
A spark that ignited a fire within
A love so pure, so divine, like a symphony in rhyme
It's a flame that will never dim

(Pre-Chorus)
In this world that's so uncertain
We found a love that's undying
Through the storms, we'll keep on burning
Our eternal flame of love, we're defying

(Chorus)
Oh, our love's an eternal flame
Burning bright, never the same
Through the years, it will remain
Guiding us, forever, in its embrace

(Verse 2)
With every passing day, our love continues to sway
Like a dance, we're spinning in perfect harmony
Through the highs and the lows, our love only grows
A bond that's unbreakable, for all eternity
```

***
[:material-arrow-left: Previous: Functional FlowSteps](Functional FlowSteps.md){ .md-button }
[Next: LLMFlows with FastAPI :material-arrow-right:](LLMFlows with FastAPI.md){ .md-button }