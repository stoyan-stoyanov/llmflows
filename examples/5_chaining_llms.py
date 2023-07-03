# pylint: skip-file

"""
This script demonstrates how to chain multiple OpenAI language models (LLMs) together 
to generate song titles, lyrics, and heavy metal paraphrases.

The script uses three OpenAI LLMs, each with a different prompt template, to generate 
a song title, lyrics for that title, and a heavy metal paraphrase of the lyrics. 
The script demonstrates how to use the PromptTemplate class to generate prompts 
with placeholders for variables, and how to chain the output of one LLM to the input 
of another.

Example:
    $ python 5_chaining_llms.py
    "The Power of Friendship"
    "Verse 1: When I'm feeling down, you're always there to lift me up..."
    "Verse 1: When the darkness comes, I'll stand my ground and fight..."

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI
    API key with access to the GPT-3 API.
"""

from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

title_llm = OpenAI()
writer_llm = OpenAI()
heavy_metal_llm = OpenAI()

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
song_title, _, _ = title_llm.generate(title_prompt)
print("Song title:\n", song_title)

lyrics_prompt = lyrics_prompt_template.get_prompt(song_title=song_title)
song_lyrics, _, _ = writer_llm.generate(lyrics_prompt)
print("Song Lyrics:\n", song_lyrics)

heavy_metal_prompt = heavy_metal_prompt_template.get_prompt(lyrics=song_lyrics)
heavy_metal_lyrics, _, _ = heavy_metal_llm.generate(heavy_metal_prompt)
print("Heavy Metal Lyrics:\n", heavy_metal_lyrics)
