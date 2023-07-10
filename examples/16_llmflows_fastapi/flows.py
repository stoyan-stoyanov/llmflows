# pylint: skip-file

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

critics = []

for i in range(10):
    critics.append(
        AsyncChatFlowStep(
            name=f"Critic Flowstep {i}",
            llm=OpenAIChat(),
            system_prompt_template=PromptTemplate(
                "You are a music critic and write short reviews of song lyrics"
            ),
            message_prompt_template=PromptTemplate(
                "Hey, what is your opinion on the following song: {song_lyrics}"
            ),
            message_key="song_lyrics",
            output_key=f"song_review_{i}",
        )
    )

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)
flowstep4.connect(*critics)

soundtrack_flow = AsyncFlow(flowstep1)
