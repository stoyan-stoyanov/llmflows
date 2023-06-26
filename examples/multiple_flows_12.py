# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create a complex data
processing pipeline using multiple flow steps.

The script creates an OpenAI language model (LLM) and several prompt templates, and 
uses them to define four flow steps: one for generating a movie title, one for 
generating a song title for the movie, one for generating two main characters for 
the movie, and one for generating lyrics for a song based on the movie title and 
main characters. The script then connects the flow steps together to create a data 
processing pipeline.

Example:
    $ python 8_complex_flows.py
    {
        "movie_title": "The Last Unicorn",
        "song_title": "The Last Unicorn",
        "main_characters": "Amalthea and Schmendrick",
        "lyrics": "In a world of darkness and despair, two heroes rise to fight the 
            evil that threatens to destroy them..."
    }

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""

import asyncio
from llmflows.flows.flow import Flow
from llmflows.flows.async_flow import AsyncFlow
from llmflows.flows.flowstep import FlowStep
from llmflows.flows.async_flowstep import AsyncFlowStep
from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate

# Create LLM
open_ai_llm = OpenAI(temperature=0.9, max_tokens=1000)

# Create prompt templates
idea_template = PromptTemplate("Give me an example of a {subject}")

planning_template = PromptTemplate(
    "Based here is an idea for a {subject}:"
    "{idea}"
    "Write me a 5-step plan to make this happen."
)

evaluation_template = PromptTemplate(
    "Here is an example of a {subject}:"
    "{idea}"
    "Here are three plans to achive it:"
    "{plan_1}"
    "---"
    "{plan_2}"
    "---"
    "{plan_3}"
    "---"
    "Explain which is the best plan and then repeat the numbered list as it is:"
)

plan_details_template = PromptTemplate(
    "Birthday plan:"
    "{best_plan}"
    "---"
    "Provide additional details for each of the steps in the plan provided above:"
)


# Create planning flowsteps
idea_flowstep = AsyncFlowStep(
    name="Idea Flowstep",
    llm=open_ai_llm,
    prompt_template=idea_template,
    output_key="idea",
)


for i in range(3):
    idea_flowstep.connect(
        AsyncFlowStep(
            name=f"Planning Flowstep {i+1}",
            llm=open_ai_llm,
            prompt_template=planning_template,
            output_key=f"plan_{i+1}",
        )
    )

subject = "wild birthday party"


async def run_flow():
    planning_flow = AsyncFlow(idea_flowstep)
    flow_results = await planning_flow.execute(subject=subject)

    for result in flow_results:
        print(result)

    return flow_results

plans = asyncio.run(run_flow())

# create evaluation flowsteps:
evaluation_flowstep = FlowStep(
    name="Evaluation Flowstep",
    llm=open_ai_llm,
    prompt_template=evaluation_template,
    output_key="best_plan",
)

details_flowstep = FlowStep(
    name="Details Flowstep",
    llm=open_ai_llm,
    prompt_template=plan_details_template,
    output_key="plan_details",
)

evaluation_flowstep.connect(details_flowstep)
best_plan_with_details = Flow(evaluation_flowstep)

complete_plan = best_plan_with_details.execute(
    subject=subject,
    idea=plans[0]["output_value"],
    plan_1=plans[1]["output_value"],
    plan_2=plans[2]["output_value"],
    plan_3=plans[3]["output_value"]
)
print(complete_plan)
