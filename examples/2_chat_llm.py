# pylint: skip-file

"""
This script demonstrates how to use the OpenAIChatLLM class to create a chatbot that 
responds to user input.

The script creates an OpenAIChatLLM object, and uses it to generate a response to a 
user message. It then prints the response to the console. The user messages are stored
in a MessageHistory object, which is used to generate the prompt for the LLM.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI
    API key with access to the GPT-3 API.
"""
import os
from llmflows.llms import OpenAIChat, MessageHistory

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

llm = OpenAIChat(api_key=open_ai_key)

message_history = MessageHistory()

while True:
    user_message = input("You:")
    message_history.add_user_message(user_message)

    llm_response, call_data, model_config = llm.generate(message_history)
    message_history.add_ai_message(llm_response)

    print(f"LLM: {llm_response}")
