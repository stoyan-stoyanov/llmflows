## TL;DR

```python
from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate

title_llm = OpenAI()
writer_llm = OpenAI()
heavy_metal_llm = OpenAI()

title_prompt_template = PromptTemplate(
    prompt="What is a good title of a song about {topic}"
)
lyrics_prompt_template = PromptTemplate(
    prompt="Write me the lyrics for a song with a title {song_title}"
)
heavy_metal_prompt_template = PromptTemplate(
    prompt="paraphrase the following lyrics in a heavy metal style: {lyrics}"
)

title_prompt = title_prompt_template.get_prompt(topic="friendship")
song_title = title_llm.generate(title_prompt)
print(song_title)

lyrics_prompt = lyrics_prompt_template.get_prompt(song_title=song_title)
song_lyrics = writer_llm.generate(lyrics_prompt)
print(song_lyrics)

heavy_metal_prompt = heavy_metal_prompt_template.get_prompt(lyrics=song_lyrics)
heavy_metal_lyrics = heavy_metal_llm.generate(heavy_metal_prompt)
print(heavy_metal_lyrics)

```

## Guide
Not implemented