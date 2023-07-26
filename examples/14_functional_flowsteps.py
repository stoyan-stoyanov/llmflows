# pylint: skip-file
import os
from llmflows.flows import Flow, ChatFlowStep, FunctionalFlowStep
from llmflows.llms import OpenAIChat, MessageHistory
from llmflows.prompts import PromptTemplate

openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

def capitalize_first_letters(lyrics: str) -> str:
    """Capitalize the first letter of each word in a string."""
    return lyrics.title()

title_message_history = MessageHistory()
title_message_history.system_prompt = "You write song titles"

lyrics_message_history = MessageHistory()
title_message_history.system_prompt = "You write song lyrics"

# Create flowsteps
title_flowstep = ChatFlowStep(
    name="Title Flowstep",
    llm=OpenAIChat(api_key=openai_api_key),
    message_history=title_message_history,
    message_prompt_template=PromptTemplate("Write a good song title about {topic}"),
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
result = songwriting_flow.start(topic="love", verbose=True)
print(result)
