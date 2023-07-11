## TL;DR

`flows.py`
```python
from llmflows.flows import AsyncFlow, AsyncFlowStep, AsyncChatFlowStep
from llmflows.llms import OpenAI, OpenAIChat
from llmflows.prompts import PromptTemplate


# Create flowsteps
flowstep1 = AsyncFlowStep(
    name="Flowstep 1",
    llm=OpenAI(),
    prompt_template=PromptTemplate("What is a good title of a movie about {topic}?"),
    output_key="movie_title",
)

flowstep2 = AsyncFlowStep(
    name="Flowstep 2",
    llm=OpenAI(),
    prompt_template=PromptTemplate(
        "What is a good song title of a soundtrack for a movie called {movie_title}?"
    ),
    output_key="song_title",
)

flowstep3 = AsyncFlowStep(
    name="Flowstep 3",
    llm=OpenAI(),
    prompt_template=PromptTemplate(
        "What are two main characters for a movie called {movie_title}?"
    ),
    output_key="main_characters",
)

flowstep4 = AsyncFlowStep(
    name="Flowstep 4",
    llm=OpenAI(),
    prompt_template=PromptTemplate(
        "Write lyrics of a movie song called {song_title}. The main characters are"
        " {main_characters}"
    ),
    output_key="song_lyrics",
)

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)

soundtrack_flow = AsyncFlow(flowstep1)
```

`app.py`

```python
from fastapi import FastAPI
import uvicorn
from flows import soundtrack_flow

app = FastAPI()

@app.get("/generate_lyrics/")
async def generate_lyrics(movie_topic: str):
    print(movie_topic)
    return await soundtrack_flow.start(topic=movie_topic, verbose=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

## Guide
In this guide we are going to see how we can create a simple fastaAPI app and use LLMFlows to build a LLM-powered web app.

Let's start by creating a simple fastAPI app. To start we need to install `fastapi` and `uvicorn`.

```
pip install fastapi uvicorn
```

let's create our `app.py`:

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/generate_lyrics/")
async def generate_lyrics(movie_topic: str):
    return {"movie_topic"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Let's run the app and see what happens:

```commandline
python3 app.py
```

```commandline
INFO:     Started server process [31938]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Looking good! Our app is running. 
Let's check what happens when we open it in the browser:

![Screenshot](assets/fastapi_guide_1.png)

So far, so good! We made a simple fastAPI app with just a few lines of code and is able to return our query parameter. 
Now let's add LLMFlow. 

Let's create a `flow.py` file in the same directory. FastAPI works great with async functions so let's reuse our async flow example from our guide.

```python
from llmflows.flows import AsyncFlow, AsyncFlowStep, AsyncChatFlowStep
from llmflows.llms import OpenAI, OpenAIChat
from llmflows.prompts import PromptTemplate


# Create flowsteps
flowstep1 = AsyncFlowStep(
    name="Flowstep 1",
    llm=OpenAI(),
    prompt_template=PromptTemplate("What is a good title of a movie about {topic}?"),
    output_key="movie_title",
)

flowstep2 = AsyncFlowStep(
    name="Flowstep 2",
    llm=OpenAI(),
    prompt_template=PromptTemplate(
        "What is a good song title of a soundtrack for a movie called {movie_title}?"
    ),
    output_key="song_title",
)

flowstep3 = AsyncFlowStep(
    name="Flowstep 3",
    llm=OpenAI(),
    prompt_template=PromptTemplate(
        "What are two main characters for a movie called {movie_title}?"
    ),
    output_key="main_characters",
)

flowstep4 = AsyncFlowStep(
    name="Flowstep 4",
    llm=OpenAI(),
    prompt_template=PromptTemplate(
        "Write lyrics of a movie song called {song_title}. The main characters are"
        " {main_characters}"
    ),
    output_key="song_lyrics",
)

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)

soundtrack_flow = AsyncFlow(flowstep1)
```

Now we can import the flow in `app.py`. Here is how the final code looks like:

```python
from fastapi import FastAPI
import uvicorn
from flows import soundtrack_flow

app = FastAPI()

@app.get("/generate_lyrics/")
async def generate_lyrics(movie_topic: str):
    return await soundtrack_flow.start(topic=movie_topic, verbose=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
Now when we pass the `movie_topic` query paramter in the url, it will be used as an input to our `soundtrack_flow`. 
Let's run it again and see what happens. 

```commandline
python3 app.py
```

***
[:material-arrow-left: Previous: Callbacks](Callbacks.md){ .md-button }
[Next: Conclusion :material-arrow-right:](Conclusion.md){ .md-button }