# pylint: skip-file

"""
This script demonstrates how to use the OpenAI class from the llmflows package to
generate text using the OpenAI GPT-3 language model.

The script creates an instance of the OpenAI class, which provides a simple interface 
for generating text using the OpenAI GPT-3 API. It then uses the generate method of 
the OpenAI class to generate a cool title for an 80s rock song, and prints the result 
to the console.

Example:
    $ python 1_llm.py
    "Rockin' the Night Away"

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API key with access to the GPT-3 API.
"""

from llmflows.llms import OpenAI


llm = OpenAI()
result = llm.generate(prompt="Generate a cool title for an 80s rock song")
print(result)
