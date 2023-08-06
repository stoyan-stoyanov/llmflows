# pylint: skip-file

"""
This script demonstrates how to use the llmflows package to create a complex flow with
multiple flow steps using chat LLMs together with vector stores.

The script uses the OpenAI and OpenAIChat classes together with several prompt templates
to generate a question related to the information stored in the vector store.
The question is then passed to the vector store to retrieve the most similar context. 
After the context is retrieved, it is passed to the next flowstep together with the 
initial question. The next flowstep then generates an answer to the question based on
the context. The answer is then passed to the next flowstep which paraphrases the
answer in an ELI5 style.

Note:
    This script requires the llmflows package to be installed, as well as an OpenAI API
    key with access to the GPT-3 API.
"""

import os
from llmflows.flows import Flow, ChatFlowStep, VectorStoreFlowStep
from llmflows.llms import OpenAIChat, OpenAIEmbeddings, MessageHistory
from llmflows.vectorstores import Pinecone
from prompts import question_template, vs_template, response_template, eli5_template

openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")

vector_db = Pinecone(
    index_name="llmflows-tutorial",
    api_key=os.environ.get("PINECONE_API_KEY", "<YOUR-API-KEY>"),
    environment="us-west4-gcp-free",
)

def create_flow():
    # Create flowsteps
    q_flowstep = ChatFlowStep(
        name="Question Flowstep",
        llm=OpenAIChat(api_key=openai_api_key),
        message_prompt_template=question_template,
        message_key="user_question",
        output_key="question",
    )

    vs_flowstep = VectorStoreFlowStep(
        name="Vectorstore Flowstep",
        embeddings_model=OpenAIEmbeddings(api_key=openai_api_key),
        vector_store=vector_db,
        top_k=3,
        append_top_k=True,
        prompt_template=vs_template,
        output_key="context",
    )

    answer_flowstep = ChatFlowStep(
        name="Answer Flowstep",
        llm=OpenAIChat(api_key=openai_api_key),
        message_prompt_template=response_template,
        message_key="question",
        output_key="answer",
    )

    eli5_flowstep = ChatFlowStep(
        name="ELI5 Flowstep",
        llm=OpenAIChat(api_key=openai_api_key),
        message_prompt_template=eli5_template,
        message_key="answer",
        output_key="eli5_answer",
    )

    # Connect flowsteps
    q_flowstep.connect(vs_flowstep, answer_flowstep)
    vs_flowstep.connect(answer_flowstep)
    answer_flowstep.connect(eli5_flowstep)

    # Create and run the Flow
    qa_flow = Flow(q_flowstep)

    return qa_flow


if __name__ == "__main__":
    qa_flow = create_flow()
    conversation_history_str = ""
    while True:
        user_question = input("You: ")
        conversation_history_str += user_question + "\n"
        results = qa_flow.start(
            conversation_history=conversation_history_str,
            user_question=user_question,
            verbose=True
        )
        answer = results["ELI5 Flowstep"]["generated"]
        conversation_history_str += answer + "\n"
        print(answer)
        print("---")
