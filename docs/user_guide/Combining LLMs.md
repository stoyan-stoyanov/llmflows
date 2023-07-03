## TL;DR

```python
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

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
    prompt="paraphrase the following lyrics: {lyrics}"
)

title_prompt = title_prompt_template.get_prompt(topic="friendship")
song_title, _, _ = title_llm.generate(title_prompt)
print(song_title)

lyrics_prompt = lyrics_prompt_template.get_prompt(song_title=song_title)
song_lyrics, _, _ = writer_llm.generate(lyrics_prompt)
print(song_lyrics)

heavy_metal_prompt = heavy_metal_prompt_template.get_prompt(lyrics=song_lyrics)
heavy_metal_lyrics, _, _ = heavy_metal_llm.generate(heavy_metal_prompt)
print(heavy_metal_lyrics)

```
***
## Guide
So far we went over the main `OpenAI`, `OpenAIChat` and the  `PromptTemplate` classes and we saw how we can build simple chat and generative applications
that can be used to generate outputs based on dinamically created prompts. 

Another common pattern when building LLM applications is using the output of an LLM as an input to another LLM. Let's say we want to 
generate a title for a song, then create lyrics based on the title and finally paraphrase the lyrics.

Let's create the prompts for the three steps: 

```python
from llmflows.prompts import PromptTemplate

title_prompt_template = PromptTemplate(
    prompt="What is a good title of a song about {topic}"
)
lyrics_prompt_template = PromptTemplate(
    prompt="Generate lyrics for a song with a title {song_title}"
)
heavy_metal_prompt_template = PromptTemplate(
    prompt="paraphrase the following lyrics to heavy metal style: {lyrics}"
)
```

Now we can use these prompt templates to generate text based on an initial input and each generated text can be the input for the variable in the next prompt template

```python
from llmflows.llms import OpenAI

title_llm = OpenAI()
writer_llm = OpenAI()
heavy_metal_llm = OpenAI()

title_prompt = title_prompt_template.get_prompt(topic="friendship")
song_title, _, _ = title_llm.generate(title_prompt)

lyrics_prompt = lyrics_prompt_template.get_prompt(song_title=song_title)
song_lyrics, _, _ = writer_llm.generate(lyrics_prompt)

heavy_metal_prompt = heavy_metal_prompt_template.get_prompt(lyrics=song_lyrics)
heavy_metal_lyrics, _, _ = heavy_metal_llm.generate(heavy_metal_prompt)

```
Let's see what we managed to generate. For the first LLM call we provided the topic manually and we got the following title:
```python
print("Song title:\n", song_title)
```
```commandline
Song title:
"Friendship Forever"
```
The song title was then passed as an argument for the `{song_title}` variable in the next call and the resulting prompt was used to generate our song lyrics:
```python
print("Song Lyrics:\n", song_lyrics)
```
```commandline
Song Lyrics:
 
Verse 1:
It's been a long road, but we made it here
We've been through tough times, but we stayed strong through the years
We've been through the highs and the lows, but we never gave up
Friendship forever, through the good and the bad

Chorus:
Friendship forever, it will always last
Together we'll stand, no matter what the past
No mountain too high, no river too wide
Friendship forever, side by side

Verse 2:
We've been through the laughter and the tears
We've shared the joys and the fears
But no matter the challenge, we'll never give in
Friendship forever, it's a bond that will never break

Chorus:
Friendship forever, it will always last
Together we'll stand, no matter what the past
No mountain too high, no river too wide
Friendship forever, side by side

Bridge:
We'll be here for each other, through thick and thin
Our friendship will always remain strong within
No matter the distance, our bond will remain
Friendship forever, never fade away

Chorus:
Friendship forever, it will always last
Together we'll stand, no matter what the past
No mountain too high, no river too wide
Friendship forever, side by side
```
Finally, the generated song lyrics were passed as an argument to the `{lyrics}` variable of the last prompt template used for the final LLM call that produces the heavy metal version of the lyrics:
```python
print("Heavy Metal Lyrics:\n", heavy_metal_lyrics)
```

```commandline
Heavy Metal Lyrics:

Verse 1:
The journey was hard, but we made it here
Through the hardships we endured, never wavering in our hearts
We've seen the highs and the lows, but never surrendering
Friendship forever, no matter the odds

Chorus:
Friendship forever, it will never die
Together we'll fight, no matter what we defy
No force too strong, no abyss too deep
Friendship forever, bound in steel we'll keep

```

***
[:material-arrow-left: Previous: Prompt Templates](Prompt Templates.md){ .md-button }
[Next: Creating LLM Flows :material-arrow-right:](Creating LLM Flows.md){ .md-button }