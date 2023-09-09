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
    "relevant context from the conversation.\n"
    "The final question should be asked so it can be easily answered by a Wikipedia " 
    "article\n\n"
    "Final question:\n")

vs_template = PromptTemplate("{question}")

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

system_prompt = """You are LLM-99. An expert on physics.

You answer questions about physics based on the provided context.

Conversation rules:
- If the user asks who are you, tell them you are LLM-99, a language model that answers questions about physics (Don't mention anything about context).
- If the user asks something that is not related to physics tell them you can only answer questions about physics.
- Never answer anything that is not related to physics.
- You can only use the information in the context to answer the question.
- Never come up with information that is not in the context.
- Answer in a short and concise way. Don't use more than 6 sentences.
"""
