from typing import Annotated

from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import BaseMessage, add_messages


class RAGAgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str

def retrieve_node(state: RAGAgentState) -> RAGAgentState:
    pass

def grade_node(state: RAGAgentState) -> RAGAgentState:
    pass

def generate_node(state: RAGAgentState) -> RAGAgentState:
    pass

def router_from_grade(state: RAGAgentState) -> str:
    pass

graph = StateGraph(RAGAgentState)
graph.add_node('retriever', action=retrieve_node)
graph.add_node('grader', action=grade_node)
graph.add_node('generater', action=generate_node)

graph.add_edge(START, 'retriever')
graph.add_edge('retriever', 'grader')
graph.add_edge('generate', END)

app = graph.compile()
app.invoke(RAGAgentState(messages=[], query='What is RAG'))
