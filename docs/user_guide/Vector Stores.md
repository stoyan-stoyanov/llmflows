## TL;DR

```python
from llmflows.llms import OpenAIEmbeddings, OpenAI
from llmflows.prompts import PromptTemplate
from llmflows.vectorstores import VectorDoc, Pinecone
import os


piencone_api_key = "<pinecone-api-key>"
openai_api_key = "<openai-api-key>"

docs = [
    (
        "In physical cosmology and astronomy, dark energy is an unknown form "
        "of energy that affects the universe on the largest scales. The first "
        "observational evidence for its existence came from measurements of "
        "supernovas, which showed that the universe does not expand at a constant "
        "rate; rather, the universe's expansion is accelerating. Understanding "
        "the universe's evolution requires knowledge of its starting conditions "
        "and composition. Before these observations, scientists thought that all "
        "forms of matter and energy in the universe would only cause the expansion "
        "to slow down over time. Measurements of the cosmic microwave background "
        "(CMB) suggest the universe began in a hot Big Bang, from which general "
        "relativity explains its evolution and the subsequent large-scale motion. "
        "Without introducing a new form of energy, there was no way to explain an "
        "accelerating expansion of the universe. Since the 1990s, dark energy has "
        "been the most accepted premise to account for the accelerated expansion."
    ),
    (
        "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
        "theoretical physicist,[5] widely acknowledged to be one of the greatest "
        "and most influential physicists of all time. Best known for developing "
        "the theory of relativity, he also made important contributions to the "
        "development of the theory of quantum mechanics. Relativity and quantum "
        "mechanics are the two pillars of modern physics.[1][6] His mass–energy "
        "equivalence formula E = mc2, which arises from relativity theory, has "
        'been dubbed "the world\'s most famous equation".[7] His work is also '
        "known for its influence on the philosophy of science.[8][9] He received "
        'the 1921 Nobel Prize in Physics "for his services to theoretical '
        "physics, and especially for his discovery of the law of the photoelectric "
        'effect",[10] a pivotal step in the development of quantum theory. His '
        'intellectual achievements and originality resulted in "Einstein" '
        'becoming synonymous with "genius".[11] Einsteinium, one of the synthetic '
        "elements in the periodic table, was named in his honor.[12]"
    ),
    (
        "A wormhole (Einstein-Rosen bridge) is a hypothetical structure connecting "
        "disparate points in spacetime, and is based on a special solution of the "
        "Einstein field equations.[1] A wormhole can be visualized as a tunnel with "
        "two ends at separate points in spacetime (i.e., different locations, "
        "different points in time, or both). Wormholes are consistent with the general "
        "theory of relativity, but whether wormholes actually exist remains to be "
        "seen. Many scientists postulate that wormholes are merely projections of a "
        "fourth spatial dimension, analogous to how a two-dimensional (2D) being could "
        "experience only part of a three-dimensional (3D) object.[2] Theoretically, a "
        "wormhole might connect extremely long distances such as a billion light "
        "years, or short distances such as a few meters, or different points in time, "
        "or even different universes.[3] In 1995, Matt Visser suggested there may be "
        "many wormholes in the universe if cosmic strings with negative mass were "
        "generated in early universe.[4][5] Some physicists, such as Kip Thorne, have "
        "suggested how to make wormholes artificially.[6]"
    ),
]

# Create embeddings LLM
embeddings_llm = OpenAIEmbeddings(api_key=openai_api_key)

# Convert text texts to VectorDocs and generate embeddings
vector_docs = [VectorDoc(doc=doc) for doc in docs]
embedded_docs = embeddings_llm.generate(vector_docs)

# initialize Pinecone
vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key=piencone_api_key,
    environment="us-west4-gcp-free",
)

# Add the embedded documents to the vector database
vector_db.upsert(docs=embedded_docs)
print(vector_db.describe())

```
***
## Guide
!!! warning

    Before starting this guide, you must have a Pinecone account and a Pinecone API key 
    and create an index called `llmflows-tutorial` with a dimension of 1536 (the default 
    dimension or openai's embeddings).
    If you don't have an account, you can register for free at 
    [pinecone.io](https://www.pinecone.io/). 

In the last several guides, we went over a couple of examples with increasing 
complexity, and we introduced several main LLMFlows abstractions (LLMs, Prompt 
Templates, Flows, Flowsetps) and their different flavors (AsyncFlows, AsyncFlowSteps, 
and AsyncChatFLows). We discussed using these to build arbitrary, explicit, 
and transparent flows that run parallel flow steps.

In the following few guides, we are going to expand to the last few abstractions, 
and we are going to see how we can build LLM-powered apps that also utilize vector 
databases (or vector stores) to augment the generative apps with ground-truth text data.

In this guide, we are going to introduce LLMFlows' `VectorStore` and `VectorDoc` 
abstractions, and we will use the `Pinecone` vector database to store word embeddings 
generated from OpenAI's embedding API by using LLMFlows `OpenAIEmbeddings` LLM class.

Vector databases usually store an embedding vector accompanied by some metadata 
provided as a dictionary.

An example data point that goes into a vector database usually looks like this:

```python
("uid," "text," [0.1, 0.15, 0.3], {"author": "Carl Sagan," "genre":"science"})
```

To help users work with this format, LLMFlows provides a simple class called 
`VectorDoc.`

Let's see how we can use this class with some texts from Wikipedia 
[[1]](https://en.wikipedia.org/wiki/Dark_energy) 
[[2]](https://en.wikipedia.org/wiki/Albert_Einstein) 
[[3]](https://en.wikipedia.org/wiki/Wormhole)
:
```python
docs = [
    (
        "In physical cosmology and astronomy, dark energy is an unknown form "
        "of energy that affects the universe on the largest scales. The first "
        "observational evidence for its existence came from measurements of "
        "supernovas, which showed that the universe does not expand at a constant "
        "rate; rather, the universe's expansion is accelerating. Understanding "
        "the universe's evolution requires knowledge of its starting conditions "
        "and composition. Before these observations, scientists thought that all "
        "forms of matter and energy in the universe would only cause the expansion "
        "to slow down over time. Measurements of the cosmic microwave background "
        "(CMB) suggest the universe began in a hot Big Bang, from which general "
        "relativity explains its evolution and the subsequent large-scale motion. "
        "Without introducing a new form of energy, there was no way to explain an "
        "accelerating expansion of the universe. Since the 1990s, dark energy has "
        "been the most accepted premise to account for the accelerated expansion."
    ),
    (
        "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
        "theoretical physicist,[5] widely acknowledged to be one of the greatest "
        "and most influential physicists of all time. Best known for developing "
        "the theory of relativity, he also made important contributions to the "
        "development of the theory of quantum mechanics. Relativity and quantum "
        "mechanics are the two pillars of modern physics.[1][6] His mass–energy "
        "equivalence formula E = mc2, which arises from relativity theory, has "
        'been dubbed "the world\'s most famous equation".[7] His work is also '
        "known for its influence on the philosophy of science.[8][9] He received "
        'the 1921 Nobel Prize in Physics "for his services to theoretical '
        "physics, and especially for his discovery of the law of the photoelectric "
        'effect",[10] a pivotal step in the development of quantum theory. His '
        'intellectual achievements and originality resulted in "Einstein" '
        'becoming synonymous with "genius".[11] Einsteinium, one of the synthetic '
        "elements in the periodic table, was named in his honor.[12]"
    ),
    (
        "A wormhole (Einstein-Rosen bridge) is a hypothetical structure connecting "
        "disparate points in spacetime, and is based on a special solution of the "
        "Einstein field equations.[1] A wormhole can be visualized as a tunnel with "
        "two ends at separate points in spacetime (i.e., different locations, "
        "different points in time, or both). Wormholes are consistent with the general "
        "theory of relativity, but whether wormholes actually exist remains to be "
        "seen. Many scientists postulate that wormholes are merely projections of a "
        "fourth spatial dimension, analogous to how a two-dimensional (2D) being could "
        "experience only part of a three-dimensional (3D) object.[2] Theoretically, a "
        "wormhole might connect extremely long distances such as a billion light "
        "years, or short distances such as a few meters, or different points in time, "
        "or even different universes.[3] In 1995, Matt Visser suggested there may be "
        "many wormholes in the universe if cosmic strings with negative mass were "
        "generated in early universe.[4][5] Some physicists, such as Kip Thorne, have "
        "suggested how to make wormholes artificially.[6]"
    ),
]
```

Now that we have the texts, we can convert them to a list of `VectorDoc` classes. We 
can also provide a metadata dictionary with any data we want.

```python
vector_docs = [VectorDoc(doc=doc, metadata={"source": "Wikipedia"}) for doc in docs]
```
!!! info
    
    LLMFlows will automatically assign a unique `doc_id`, but this can also be 
    provided explicitly.

The next step is to use the VectorDocs together with an embeddings model to create the 
embedding vectors for each `VectorDoc`:

```python
embeddings_llm = OpenAIEmbeddings()
embedded_docs = embeddings_llm.generate(vector_docs)
```

Each `VectorDoc` has an embedding property that gets updated by the embedding model, 
and we are now ready to upload the VectorDocs to a vector database. 

For this example, we are going to use the Pinecone vector database. LLMFlows provides 
a `Pinecone` class we can use in the following way:

```python
vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free",
)
```

The final step is to upload the VectorDocs we created with the help of the `Pinecone` 
client:
```python
vector_db.upsert(docs=embedded_docs)
```

Finally, let's confirm that the upload was succesfull:
```python
print(vector_db.describe())
```

In the following guide, we will see how we can use LLMs to create a question-answering 
application with the help of Pinecone and the vectors we just uploaded.

***
[:material-arrow-left: Previous: Async Flows](Async Flows.md){ .md-button }
[Next: Question Answering :material-arrow-right:](Question Answering.md){ .md-button }