# pylint: skip-file

"""
This script demonstrates how to use the OpenAIChatLLM class to create a chatbot that 
responds to user input.

The script creates an instance of the OpenAIChatLLM class with a system prompt, and 
then enters a loop where it prompts the user for input, adds the input to the LLM's 
message history, generates a response using the LLM's chat method, and prints the 
response to the console.

Example:
    $ python 2_chat_llm.py
    You: Hi, can you help me with something?
    LLM: Sure, what do you need help with?
    You: I'm trying to install a Python package with pip, but it's not working.
    LLM: What error message are you getting?
    You: It says 'pip' is not recognized as an internal or external command.
    LLM: It sounds like pip is not in your system PATH. Have you tried adding it 
    to your PATH?

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI
    API key with access to the GPT-3 API.
"""

from llmflows.llms import OpenAIChat


llm = OpenAIChat(system_prompt="You are a useful assistant")

while True:
    user_message = input("You:")
    llm.add_message(user_message)
    llm_response, call_data, model_config = llm.generate()
    print(f"LLM: {llm_response}")
