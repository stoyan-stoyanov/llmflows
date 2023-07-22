# pylint: skip-file
import os
from llmflows.flows import Flow, ChatFlowStep, FunctionalFlowStep
from llmflows.llms import OpenAIChat, MessageHistory
from prompts import system_prompt, react_prompt_template
from tools import tool_selector

open_ai_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

message_history = MessageHistory()
message_history.system_prompt = system_prompt


# Create flowsteps
thought_action = ChatFlowStep(
    name="Thought + Action Step",
    llm=OpenAIChat(api_key=open_ai_key, model="gpt-4"),
    message_history=message_history,
    message_prompt_template=react_prompt_template,
    message_key="question",
    output_key=f"thought_action",
)

observation = FunctionalFlowStep(
    name="Observation Step",
    flowstep_fn=tool_selector,
    output_key="observation",
)

# Connect flowsteps
thought_action.connect(observation)

# Create the flow
react_agent_flow = Flow(thought_action)

problem = "What is the age difference between Barak Obama and Michelle Obama?"
max_steps = 5
react_history = ""

for i in range(max_steps):
    result = react_agent_flow.start(
        question=problem, react_history=react_history, verbose=True
    )

    if result["Observation Step"]["generated"] == "<final_answer>":
        break
    else:
        # add the thought, action and observations from this flow to the history
        react_history += (
            result["Thought + Action Step"]["generated"]
            + result["Observation Step"]["generated"]
            + "\n"
        )
