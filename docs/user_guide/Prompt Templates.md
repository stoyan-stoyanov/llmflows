## TL;DR

```python
from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate

prompt_template = PromptTemplate(
    prompt="Generate a title for a 90s hip-hop song about {topic}."
)
llm_prompt = prompt_template.get_prompt(topic="friendship")

print(llm_prompt)

llm = OpenAI()
song_title = llm.generate(llm_prompt)
print(song_title)

```

## Guide
Not implemented