# pylint: skip-file

"""
This script demonstrates how to use the OpenAI class from the llmflows package to
generate text using the OpenAI GPT-3 language model.

The script creates an instance of the OpenAI class, which provides a simple interface 
for generating text using the OpenAI GPT-3 API. It then uses the generate method of 
the OpenAI class to generate a cool title for an 80s rock song, and prints the result 
to the console.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI 
    API key with access to the GPT-3 API.
"""
import os
from llmflows.llms import OpenAI

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

llm = OpenAI(api_key=open_ai_key)
result, call_data, model_config = llm.generate(prompt="Generate a cool title for an 80s rock song")
print(result)
