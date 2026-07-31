from typing import Annotated

from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from config import AI_KEY


class RAGGraphState(BaseModel):
    messages = Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(
    api_key=SecretStr(AI_KEY),
    base_url='https://api.proxyapi.ru/openai/v1',
    model='gpt-4.1-nano'
)


def agentic_node(state: RAGGraphState):
    system_prompt = SystemMessage(content='''
        Ты RAG система для поиска информации по текстовой базе данных
    ''')

    pass