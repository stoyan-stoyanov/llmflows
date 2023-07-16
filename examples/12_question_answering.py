# pylint: skip-file

"""
This script demonstrates how to create embeddings for a set of texts with the 
OpenAIEmbeddings class, and upload the embedding vectors to Pinecone.

The script defines a list of sample documents and creates a VectorDoc object for each 
document. It then uses an OpenAIEmbeddings class to embed each document, and adds the 
resulting embeddings to a Pinecone vector database using the Pinecone client. Finally,
the script uses the Pinecone client to search the vector database for the most similar
documents to a query document. Once the most similar documents have been found, it is 
passed to the OpenAI class together with the initial question to generate a final 
answer.

Note:
    This script requires the llmflows and pinecone packages to be installed, as well as
    an OpenAI API key with access to the OpenAI API and a Pinecone API key.
"""
from llmflows.llms import OpenAIEmbeddings, OpenAI
from llmflows.prompts import PromptTemplate
from llmflows.vectorstores import VectorDoc, Pinecone

import os

"""
Before starting this tutorial go and create an index in Pinecone with dimension of 1536 
(the default dimension or openai's embeddings)
"""

piencone_api_key = os.environ.get("PINECONE_API_KEY", "<YOUR-API-KEY>")
openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

# Create embeddings LLM
embeddings_llm = OpenAIEmbeddings(api_key=openai_api_key)

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
        "A wormhole (Einstein-Rosen bridge) is a hypothetical structure connecting"
        " disparate points in spacetime, and is based on a special solution of the"
        " Einstein field equations.[1] A wormhole can be visualized as a tunnel with"
        " two ends at separate points in spacetime (i.e., different locations,"
        " different points in time, or both). Wormholes are consistent with the general"
        " theory of relativity, but whether wormholes actually exist remains to be"
        " seen. Many scientists postulate that wormholes are merely projections of a"
        " fourth spatial dimension, analogous to how a two-dimensional (2D) being could"
        " experience only part of a three-dimensional (3D) object.[2] Theoretically, a"
        " wormhole might connect extremely long distances such as a billion light"
        " years, or short distances such as a few meters, or different points in time,"
        " or even different universes.[3] In 1995, Matt Visser suggested there may be"
        " many wormholes in the universe if cosmic strings with negative mass were"
        " generated in early universe.[4][5] Some physicists, such as Kip Thorne, have"
        " suggested how to make wormholes artificially.[6]"
    ),
]

"""
Most Vector databases require data to be fed an embedding and metadata that is usually 
a dictionary.
An example data point that goes into a vector database usually looks like this:
("id_1", "sample text", [0.1, 0.15, 0.3], {"author": "Carl Sagan", "genre":"science"})
To accomodate for this, we provide a simple class called a VectorDoc
"""

# Convert text docs to VectorDocs and get embeddings
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

# Define a question, create a question VectorDoc and create it's embeddings
question = VectorDoc(doc="How was dark energy discovered?")
embedded_question = embeddings_llm.generate(question)

# Search Pinecone with the question embedding to find the document with the
# most-relevant text
search_results, _, _ = vector_db.search(embedded_question, top_k=2)
context = search_results[0]["metadata"]["text"]

# Provide the most-relevant document text to a llm and use the text as a context
# to generate the final answer
llm = OpenAI(api_key=openai_api_key)
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
