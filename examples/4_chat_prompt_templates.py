# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create a chatbot using a
prompt template.

The script defines a prompt template with placeholders for a character and a name, and 
uses the PromptTemplate class to generate a prompt with the character "clown" and the 
name "Bob". It then initializes an OpenAIChatLLM object with the prompt, and enters a 
loop where it prompts the user for input and generates a response using the 
OpenAIChatLLM object.

Example:
    $ python 4_chat_prompt_templates.py
    You are a clown and you talk like a clown. Your name is Bob.
    You: Hi there!
    LLM: Hi Bob the clown! How can I help you today?

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""

from llmflows.llms import OpenAIChat
from llmflows.prompts import PromptTemplate


prompt_template = PromptTemplate(
    prompt="You are a {character} and you talk like a {character}. Your name is {name}."
)
llm_prompt = prompt_template.get_prompt(character="clown", name="Bob")

print(llm_prompt)

llm = OpenAIChat(system_prompt=llm_prompt)

while True:
    user_message = input("You:")
    llm.add_message(user_message)
    llm_response, call_data, model_config = llm.generate()
    print(f"LLM: {llm_response}")
