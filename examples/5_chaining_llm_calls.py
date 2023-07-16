# pylint: skip-file

"""
This script demonstrates how to use the output of a given LLM as an input to another one
to generate song titles, lyrics, and paraphrase its.

The script makes three calls to LLMs, each with a different prompt template, to generate 
a song title, lyrics for that title, and a heavy metal paraphrase of the lyrics. 

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI
    API key with access to the GPT-3 API.
"""
import os
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

llm = OpenAI(api_key=open_ai_key)

title_prompt_template = PromptTemplate(
    prompt="What is a good title of a song about {topic}"
)
lyrics_prompt_template = PromptTemplate(
    prompt="Generate lyrics for a song with a title {song_title}"
)
heavy_metal_prompt_template = PromptTemplate(
    prompt="paraphrase the following lyrics to heavy metal style: {lyrics}"
)

title_prompt = title_prompt_template.get_prompt(topic="friendship")
song_title, _, _ = llm.generate(title_prompt)
print("Song title:\n", song_title)

lyrics_prompt = lyrics_prompt_template.get_prompt(song_title=song_title)
song_lyrics, _, _ = llm.generate(lyrics_prompt)
print("Song Lyrics:\n", song_lyrics)

heavy_metal_prompt = heavy_metal_prompt_template.get_prompt(lyrics=song_lyrics)
heavy_metal_lyrics, _, _ = llm.generate(heavy_metal_prompt)
print("Heavy Metal Lyrics:\n", heavy_metal_lyrics)
