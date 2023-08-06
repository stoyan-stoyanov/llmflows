# pylint: skip-file

from llmflows.prompts import PromptTemplate

question_template = PromptTemplate(
    "Generate a final question based on the following conversation history and user "
    "question:\n"
    "Conversation history:\n"
    "{conversation_history}\n"
    "---\n"
    "User question:\n"
    "{user_question}\n"
    "---\n"
    "The final question should be the same as the user question but include any "
    "relevant context from the conversation\n\n"
    "Final question:\n")

vs_template = PromptTemplate("I have the following question: {question}")

response_template = PromptTemplate(
    "Answer the question based on the context.\n"
    "Context:\n"
    "{context}\n"
    "Question:\n"
    "{question}"
    "Only use the information in the context!!! Do not use any other information.\n"
    "Answer:\n"
)

eli5_template = PromptTemplate(
    "paraphrase the following text in an ELI5 style:\n{answer}"
)