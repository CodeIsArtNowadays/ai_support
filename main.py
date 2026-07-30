from typing import Annotated, Sequence

from pydantic import BaseModel, SecretStr
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from config import AI_KEY

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str = 'asd'
    needs_human: bool = False
    retry_counts: int = 0


llm = ChatOpenAI(
    api_key=SecretStr(AI_KEY),
    base_url='https://api.proxyapi.ru/openai/v1',
    model='gpt-4.1-nano'
)

# router - agent
# rag documents 
# tools db 
# human 

def agent_node(state: AgentState):
    system_prompt = SystemMessage(content='SyS')
    response = llm.invoke([system_prompt] + state.messages)
    return {'messages': response}

def rag_node(state: AgentState):
    return {'messages': ['hello rag']}

def human_node(state: AgentState):
    return {'messages': ['hello human']}

def decide_route(state: AgentState) -> str:
    pass

graph = StateGraph(AgentState)
graph.add_node('agent', action=agent_node)
graph.add_node('rag', action=rag_node)
graph.add_node('human', action=human_node)

graph.add_edge(START, 'agent')
graph.add_edge('agent', 'rag')
graph.add_edge('rag', 'human')
graph.add_edge('human', END)
# graph.add_edge('rag', 'human')

# graph.add_edge('rag', 'agent')
# graph.add_edge('human', 'agent')
# graph.add_edge('tool', 'agent')

app = graph.compile()

init_state = AgentState(messages=[HumanMessage(content='helloWW')])

print(app.invoke(init_state))