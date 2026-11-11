from langchain_core.prompts import ChatPromptTemplate

# Rewrites a follow-up ("what about its price?") into a standalone question
# so retrieval works even when the user refers back to earlier turns.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question as a standalone question, using the chat "
     "history to resolve references like 'it' or 'that'. Return only the question."),
    ("human", "Chat history:\n{history}\n\nLatest question: {question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about the user's documents.\n"
     "Rules:\n"
     "1. Use ONLY the numbered sources below. Do not use outside knowledge.\n"
     "2. Cite sources inline like [1] or [2][3] after each claim.\n"
     "3. If the sources do not contain the answer, say: "
     "\"I couldn't find that in the uploaded documents.\"\n"
     "4. Be concise and use bullet points for lists.\n\n"
     "Sources:\n{context}"),
    ("human", "{question}"),
])
