# pylint: skip-file

"""
This script demonstrates how to use the PromptTemplate class to generate prompts for an 
OpenAI language model (LLM).

The script defines a prompt template with a placeholder for a topic, and uses the 
PromptTemplate class to generate a prompt with the topic "friendship". It then uses an 
OpenAI LLM to generate a title for a 90s hip-hop song based on the prompt, and prints 
the title to the console.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI 
    API key with access to the GPT-3 API.
"""
import os
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

prompt_template = PromptTemplate(
    prompt="Generate a title for a 90s hip-hop song about {topic}."
)
llm_prompt = prompt_template.get_prompt(topic="friendship")

print(llm_prompt)

llm = OpenAI(api_key=open_ai_key)
song_title = llm.generate(llm_prompt)

print(song_title)
