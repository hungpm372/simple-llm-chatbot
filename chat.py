import os
import time

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=".env.dev")

store: dict = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "1"}}

while True:
    user_input = input("You: ")
    print(f"Assistant: ", end="")
    for r in with_message_history.stream(
            [HumanMessage(content=user_input)],
            config=config,
    ):
        print(r.content, end="")
        time.sleep(0.05)

    print()
