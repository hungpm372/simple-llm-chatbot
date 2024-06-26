import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=".env.dev")

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

result = model.invoke(
    [
        HumanMessage(content="Hi! I'm Hung"),
        AIMessage(content="Hello Hung! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

print(result)
