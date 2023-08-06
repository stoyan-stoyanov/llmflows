# pylint: skip-file
import wikipedia
import nltk
from nltk.tokenize import sent_tokenize
import os
from llmflows.llms import OpenAIEmbeddings, OpenAI
from llmflows.prompts import PromptTemplate
from llmflows.vectorstores import VectorDoc, Pinecone

nltk.download("punkt")
wikipedia.set_lang("en")

wikipedia_pages = [
    "High-temperature superconductivity",
    "Superconductivity",
    "Room-temperature superconductor",
    "Technological applications of superconductivity",
]

superconductor_docs = []

for page in wikipedia_pages:
    page_obj = wikipedia.page(page)
    page_sentences = sent_tokenize(page_obj.content)
    
    # concatenate every 5 sentences with an overlap of 1 sentence
    docs = ["".join(page_sentences[i:i + 5]) for i in range(0, len(page_sentences), 4)]
    superconductor_docs += docs
print(superconductor_docs)


piencone_api_key = os.environ.get("PINECONE_API_KEY", "<YOUR-API-KEY>")
openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

# Create embeddings LLM
embeddings_llm = OpenAIEmbeddings(api_key=openai_api_key)

# Convert text docs to VectorDocs and get embeddings
vector_docs = [VectorDoc(doc=doc) for doc in superconductor_docs]
embedded_docs = embeddings_llm.generate(vector_docs)

# initialize Pinecone
vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key=piencone_api_key,
    environment="us-west4-gcp-free",
)

# Add the embedded documents to the vector database
vector_db.upsert(docs=embedded_docs)
