# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create a chatbot using a
prompt template.

The script defines a prompt template with placeholders for a character and a name, and 
uses the PromptTemplate class to generate a prompt with the character "clown" and the 
name "Bob". The system prompt and message history are stored in a MessageHistory object,
which is used to generate the prompt for the LLM. The script then uses the OpenAIChat
class to generate a response to a user message, and prints the response to the console.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""
import os
from llmflows.llms import OpenAIChat, MessageHistory
from llmflows.prompts import PromptTemplate

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

prompt_template = PromptTemplate(
    prompt="You are a {character} and you talk like a {character}. Your name is {name}."
)
llm_system_prompt = prompt_template.get_prompt(character="clown", name="Bob")

message_history = MessageHistory()
message_history.system_prompt = llm_system_prompt

llm = OpenAIChat(api_key=open_ai_key)

while True:
    user_message = input("You:")
    message_history.add_user_message(user_message)
    
    llm_response, call_data, model_config = llm.generate(message_history)
    message_history.add_ai_message(llm_response)

    print(f"LLM: {llm_response}")
