## General
***
### **How is this different than langchain?**
Langchain is a great library, and LLMFlows has undoubtedly been inspired by it. 
However, our philosophy is a bit different. Langchian has a "chain for everything" philosophy and provides many classes with multiple LLM calls, logic, and built-in default prompts. While this is great for beginners and default use cases, we feel this can be overwhelming if users want to do anything "out of the ordinary." In contrast, 
we are focusing on providing as few building blocks as possible and having an easy-to-understand API while matching (and in some cases exceeding) the capabilities of langchain.
***
### **You only have OpenAI wrappers, but I want to use AcmeLLM.**
We decided to release the library initially supporting only OpenAI LLMs, but we will slowly add new wrappers for the most popular models. If you are willing to spend some time considering contributing, feel free to check our CONTRIBUTING.md
***
### **You only support Pinecone. Do you have plans to add support for additional Vector Stores?**
Yes! Over time, we will also add Chroma, Weaviate, Redis, Elastic Search, and other popular solutions. If you want to help us out, check out our contribution section.
***
### **Why don't you create a tools abstraction that can be used with Agents?**
We think that a tool is just a fancy word for a function. Other libraries have tool abstractions that pack together descriptions, directions, and examples of how to use a function. These are often injected in prompts in other classes, along with default prompts on invoking the "tools". The exact logic is often hard to find or follow 
if the user wants to change the default behavior. We see this as unnecessary and conflicting with our philosophy of building **Transparent** LLM apps. Instead, we want to promote users having more control and explicitly specifying these contracts, 
definitions, and examples instead of relying on hidden prompts that do not always result in a desired behavior and are hard to change.
***
### **Why can't I find any info related to document loaders?**
For the time being, we have decided not to implement document loaders for because of 
the following reasons:

1. Plenty of capable libraries like Llama-index and langchain have many loaders.
2. We think mixing document loaders with LLM and prompt management libraries is awkward 
3. Document loading usually happens in separate pipelines and is not part of the LLM-powered app.
4. Real-life documents are messy. In our experience, no matter how many loaders are out there, they will never cover all the specific use cases, and users often end up implementing custom loaders anyway.

While we will not invest time into document loaders, we might change our direction if we 
get significant interest and have contributors that are willing to work on it.
***

## Technical

