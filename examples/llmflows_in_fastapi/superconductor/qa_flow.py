# pylint: skip-file

import os
from llmflows.flows import Flow, ChatFlowStep, VectorStoreFlowStep
from llmflows.llms import OpenAIChat, OpenAIEmbeddings, MessageHistory
from llmflows.vectorstores import Pinecone
from prompts import vs_template, response_template, eli5_template, system_prompt


def create_flow():
    openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")
    vector_db = Pinecone(
        index_name="llm-99",
        api_key=os.environ.get("PINECONE_API_KEY", "<YOUR-API-KEY>"),
        environment="us-west4-gcp-free",
    )

    mh = MessageHistory()
    mh.system_prompt = system_prompt

    vs_flowstep = VectorStoreFlowStep(
        name="Vectorstore Flowstep",
        embeddings_model=OpenAIEmbeddings(api_key=openai_api_key),
        vector_store=vector_db,
        top_k=5,
        append_top_k=True,
        prompt_template=vs_template,
        output_key="context",
    )

    answer_flowstep = ChatFlowStep(
        name="Answer Flowstep",
        llm=OpenAIChat(api_key=openai_api_key, max_tokens=500),
        message_history=mh,
        message_prompt_template=response_template,
        message_key="question",
        output_key="answer",
    )

    eli5_flowstep = ChatFlowStep(
        name="ELI5 Flowstep",
        llm=OpenAIChat(api_key=openai_api_key, max_tokens=500),
        message_prompt_template=eli5_template,
        message_key="answer",
        output_key="eli5_answer",
    )

    # Connect flowsteps
    vs_flowstep.connect(answer_flowstep)
    answer_flowstep.connect(eli5_flowstep)

    # Create and run the Flow
    qa_flow = Flow(vs_flowstep)

    return qa_flow


if __name__ == "__main__":
    qa_flow = create_flow()
    conversation_history_str = ""
    while True:
        user_question = input("You: ")
        conversation_history_str += user_question + "\n"
        results = qa_flow.start(
            conversation_history=conversation_history_str,
            question=user_question,
            verbose=True
        )
        answer = results["ELI5 Flowstep"]["generated"]
        conversation_history_str += answer + "\n"
        print(results["Vectorstore Flowstep"]["call_data"]["raw_outputs"])
        print(answer)
        print("---")
