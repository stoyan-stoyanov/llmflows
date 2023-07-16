## TL;DR

```python
from llmflows.llms import OpenAIEmbeddings, OpenAI
from llmflows.prompts import PromptTemplate
from llmflows.vectorstores import VectorDoc, Pinecone
import os

"""
Before starting this guide you need to have completed the previous one and upsert the VectorDocs to Pinecone.
"""

piencone_api_key = "<pinecone-api-key>"
openai_api_key = "<openai-api-key>"

# Create embeddings LLM
embeddings_llm = OpenAIEmbeddings(api_key=openai_api_key)

# initialize Pinecone
vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key=piencone_api_key,
    environment="us-west4-gcp-free",
)

# Define a question, create a question VectorDoc and create it's embeddings
question = VectorDoc(doc="How was dark energy discovered?")
embedded_question = embeddings_llm.generate(question)

# Search Pinecone with the question embedding to find the document with the
# most-relevant text
search_results, _, _ = vector_db.search(embedded_question, top_k=2)
context = search_results[0]["metadata"]["text"]

# Provide the most-relevant document text to a llm and use the text as a context
# to generate the final answer
llm = OpenAI()
prompt_template = PromptTemplate(
    prompt=(
        "Answer the question based on the"
        " context.\nContext:\n{context}\nQuestion:\n{question}"
    )
)

llm_prompt = prompt_template.get_prompt(question=question.doc, context=context)
print(llm_prompt)

final_answer = llm.generate(llm_prompt)
print("Final answer:", final_answer)

```
***
## Guide
!!! warning
    Before starting this guide, you must complete the 
    [previous guide](Vector Databases.md) and upsert the VectorDocs into Pinecone.

In the previous guide, we got familiar with the `VectorStore` and `VectorDoc` classes. 
We used some text from Wikipedia to create a list of VectorDocs, and then we generated 
word embeddings for each one using the `OpenAIEmbeddings` class. Finally, we uploaded 
the vectors to the Pinecone vector database with the help of the `Pinecone` client.

Now that we have some data in Pinecone, we can start putting it to work. Question 
answering is one of the most popular applications of LLMs and vector databases. 
Here is a common question-answering approach:

1. We start with a string representing a question
2. We convert the string to a vector embedding
3. We use the vector embedding to search a vector database for a text with a similar 
4. vector embedding
5. We obtain the text from the most similar vector
6. We use the original question and the obtained text together with an LLM, and we ask 
7. the LLM to generate a response

Let's create the question and its vector embeddings:

```python
question = VectorDoc(doc="How was dark energy discovered?")
embeddings_llm = OpenAIEmbeddings(api_key="<openai-api-key>")
embedded_question = embeddings_llm.generate(question)
```

We can use the question embeddings to search Pinecone:

```python
vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key="<pinecone-api-key>",
    environment="us-west4-gcp-free",
)

search_results, _, _ = vector_db.search(embedded_question)
context = search_results[0]["metadata"]["text"]

```

Let's see what the context provided by the search result looks like.

```python
print(context)
```

```commandline
In physical cosmology and astronomy, dark energy is an unknown form of energy 
that affects the universe on the largest scales. The first observational 
evidence for its existence came from  measurements of supernovas, which 
showed that the universe does not expand at a constant rate;  rather, the 
universe's expansion is accelerating. Understanding the universe's evolution 
requires knowledge of its starting conditions and composition. Before these 
observations, scientists thought that all forms of matter and energy in the 
universe would only cause the expansion to slow down over time. Measurements 
of the cosmic microwave background (CMB) suggest the universe began in a hot 
Big Bang, from which general relativity explains its evolution and the 
subsequent large-scale motion. Without introducing a new form of energy, there
was no way to explain an accelerating expansion of the universe. Since the 
1990s, dark energy has been the most accepted premise to account for the 
accelerated expansion.

```
Great! The retrieved text is relevant and contains the information required to answer 
the question.

Now we have the question and the context, and we can use an LLM to answer the question 
based on the context:

```python
llm = OpenAI(api_key="<openai-api-key>")
prompt_template = PromptTemplate(
    prompt=(
        "Answer the question based on the"
        " context.\nContext:\n{context}\nQuestion:\n{question}"
    )
)

llm_prompt = prompt_template.get_prompt(question=question.doc, context=context)
final_answer = llm.generate(llm_prompt)

```

Let's see how did the LLM do:

```python
print(final_answer)
```

```commandline
Dark energy was discovered through measurements of supernovas, which showed
that the universe was expanding at an accelerating rate, contrary to what 
scientists had previously believed. Measurements of the cosmic microwave 
background (CMB) supported this conclusion and provided further evidence of 
dark energy's existence.
```

Impressive, right? We managed to fetch the proper context from many texts stored in a 
vector database and use a large language model to answer our question!

Now that we know what makes vector stores tick, we will see how to use them in flows 
in the next guide.

***
[:material-arrow-left: Previous: Vector Stores](Vector Stores.md){ .md-button }
[Next: Vector Stores in Flows :material-arrow-right:](Vector Stores in Flows.md){ .md-button }