Thank you for taking the time to read our user guide! In this guide, we explored the 
various abstractions provided by LLMFlows for building LLM-powered applications. We 
covered the main concepts of `LLMs`, `Prompt Templates`, `MessageHistory`, `Flows`, 
`FlowSteps`, `VectorStores`, `VectorDocs`, and `Callbacks`. 

In addition to reviewing the main abstractions, we saw how we could use LLMFlows to 
build a question-answering application, a web app with FastAPI, and even a 
state-of-the-art autonomous agent.

LLMFlows allows developers to easily interact with LLMs such as OpenAI's ChatGPT and 
utilize their text-generating capabilities. We learned how to configure LLM classes, 
generate text using prompts, and work with chat-based LLMs for interactive 
conversations.

Prompt Templates provided a convenient way to create dynamic prompts by using variables 
and generating prompt strings based on those variables.

Flows and FlowSteps were instrumental in structuring our LLM applications. We learned 
how to define flows, create flow steps, connect them in a directed acyclic graph (DAG) 
structure, and start the flow to execute the steps in the desired order. We saw how 
VectorStoreFlowSteps can integrate with vector databases for efficient and scalable 
storage and retrieval of vector embeddings.

FunctionalFlowSteps allowed us to incorporate custom string manipulation functions into 
flows when we don't need LLM calls, allowing for more versatile applications.

We explored how FunctionalCallbacks can be utilized in flows to execute callback 
functions at different stages, enabling integrations with logging, tracing, or other 
custom requirements.

We hope you've gained a solid understanding of the library and its capabilities. We are 
grateful for your time and interest in LLMFlows, and we hope that it will empower you 
to build incredible LLM-powered applications. Thank you!

***
[:material-arrow-left: Previous: Agents](Agents.md){ .md-button }
[Next: FAQ :material-arrow-right:](FAQ.md){ .md-button }