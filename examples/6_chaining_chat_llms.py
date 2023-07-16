# pylint: skip-file

"""
This script demonstrates how to use the output of a ChatLLM as an input to another one
to generate song titles, lyrics, and paraphrase its.

The script makes three calls to LLMs, each with a different prompt template, to generate 
a song title, lyrics for that title, and a heavy metal paraphrase of the lyrics. 

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI
    API key with access to the GPT-3 API.
"""
import os
from llmflows.llms import OpenAIChat, MessageHistory
from llmflows.prompts import PromptTemplate

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

llm = OpenAIChat(api_key=open_ai_key)

title_msg_history = MessageHistory()
title_msg_history.system_prompt = (
    "You are a useful AI that can come up with song titles"
)

writer_msg_history = MessageHistory()
writer_msg_history.system_prompt = "You are a useful AI that can write song lyrics"

heavy_metal_msg_history = MessageHistory()
heavy_metal_msg_history.system_prompt = (
    "You are an AI that can paraphrase song lyrics in a heavy metal style"
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

title_msg_history.add_user_message(title_prompt_template.get_prompt(topic="friendship"))
song_title, _, _ = llm.generate(title_msg_history)
print(song_title)

writer_msg_history.add_user_message(
    lyrics_prompt_template.get_prompt(song_title=song_title)
)
song_lyrics, _, _ = llm.generate(writer_msg_history)
print(song_lyrics)

heavy_metal_msg_history.add_user_message(
    heavy_metal_prompt_template.get_prompt(lyrics=song_lyrics)
)
heavy_metal_lyrics, _, _ = llm.generate(heavy_metal_msg_history)
print(heavy_metal_lyrics)
