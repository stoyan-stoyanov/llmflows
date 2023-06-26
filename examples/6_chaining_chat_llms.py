# pylint: skip-file

"""
This script demonstrates how to chain multiple OpenAIChatLLMs together to generate 
song titles, lyrics, and heavy metal paraphrases.

The script uses three OpenAIChatLLMs, each with a different system prompt and prompt 
template, to generate a song title, lyrics for that title, and a heavy metal 
paraphrase of the lyrics. The script demonstrates how to use the PromptTemplate class 
to generate prompts with placeholders for variables, and how to chain the output of 
one LLM to the input of another.

Example:
    $ python 6_chaining_chat_llms.py
    "The Power of Friendship"
    "Verse 1: When I'm feeling down, you're always there to lift me up..."
    "Verse 1: When the darkness comes, I'll stand my ground and fight..."

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI
    API key with access to the GPT-3 API.
"""

from llmflows.llms import OpenAIChat
from llmflows.prompts import PromptTemplate

title_llm = OpenAIChat(
    system_prompt="You are a useful AI that can come up with song titles"
)
writer_llm = OpenAIChat(
    system_prompt="You are a useful AI that can write song lyrics"
)
heavy_metal_llm = OpenAIChat(
    system_prompt="You are an AI that can paraphrase song lyrics in a heavy metal style"
)

title_prompt_template = PromptTemplate(
    prompt="What is a good title of a song about {topic}"
)
lyrics_prompt_template = PromptTemplate(
    prompt="Write me the lyrics for a song with a title {song_title}"
)
heavy_metal_prompt_template = PromptTemplate(
    prompt="paraphrase the following lyrics: {lyrics}"
)

title_prompt = title_prompt_template.get_prompt(topic="friendship")
title_llm.add_message(title_prompt)
song_title = title_llm.generate()

print(song_title)

lyrics_prompt = lyrics_prompt_template.get_prompt(song_title=song_title)
writer_llm.add_message(lyrics_prompt)
song_lyrics = writer_llm.generate()

print(song_lyrics)

heavy_metal_prompt = heavy_metal_prompt_template.get_prompt(lyrics=song_lyrics)
heavy_metal_llm.add_message(heavy_metal_prompt)
heavy_metal_lyrics = heavy_metal_llm.generate()

print(heavy_metal_lyrics)
