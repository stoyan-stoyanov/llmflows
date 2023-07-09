# pylint: skip-file

from llmflows.flows import Flow, ChatFlowStep, FunctionalFlowStep
from llmflows.llms import OpenAIChat
from llmflows.prompts import PromptTemplate

# Create prompt templates
title_prompt_template = PromptTemplate(
    prompt="What is a good title of a song about {topic}"
)
lyrics_prompt_template = PromptTemplate(
    prompt="Write me the lyrics for a song with a title {song_title}"
)


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
