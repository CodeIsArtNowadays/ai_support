import json
from typing import Annotated

from pydantic import BaseModel, SecretStr
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from config import AI_KEY


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str = 'asd'
    needs_human: bool = False
    retry_counts: int = 0

# router - agent
# rag documents 
# tools db 
# human 
@tool
def get_from_db(model: 'str', id: int):
    '''Search in DB to get information
        Args:
            model: str - What data to retrieve
            id: int - ID of raw to retrieve
    '''
    print('TOOLS: ', model)
    match model:
        case 'users':
            orders = [4, 54, 77, 544]
            discount = 10 if id % 2 == 0 else 15
            return {'user_id': id, 'orders_id': orders, 'discount': discount}
        case 'orders':
            delivery = 1 if id % 2 == 0 else 2
            return {'order': id, 'delivery_in': delivery}
        case _:
            return {'status': 'not found in db'}

tools = [get_from_db]

llm = ChatOpenAI(
    api_key=SecretStr(AI_KEY),
    base_url='https://api.proxyapi.ru/openai/v1',
    model='gpt-4.1-mini'
).bind_tools(tools)

def agent_node(state: AgentState):
    system_prompt = SystemMessage(content=
        '''Ты AI ассистент техподдержки отвечающий за обработку сообщений от пользователей.
            У тебя в подчинение есть несколько других агентов со следующими возможностями:
                document_agent: Поиск по текстовой базе данных для ответа на вопросы о работе компании, обслуживании клиентов и тому подобные.
                tools_agent: Поиск информации в базе данных по клиентам, заказах, доставках. Имеет возможность изменять статус заказов и доставок. Так же обрабатывает отмены. Не вызывай его более одного раза с одиннаковыми параметрами. 
                            Доступные модели в базе данных 
                                users - возвращает id, активные заказы пользователя и уровень скидочной карты, 
                                products - возвращает id и цену, 
                                orders - возвращает id заказа и срок его доставки              
                finale: ответить пользователю
            Они все вернуться к тебе с ответами.
            Твоя основная задача - вернуть ответ пользователю отвечать на сообщения клиентов, для этого ты можешь вызывать агентов и инструменты когда необходимо.
          
            Ты можешь отвечать ТОЛЬКО в правильном JSON формате, без вступлений, заключений и любого другого текста. В ответе ДОЛЖНЫ содержаться поля "next_step", "content"
           Пример твоего ответа
          {
            "next_step": "имя агента которому ты передаешь управление, или finale если ответ готов",
            "content": "Запрос к агенту, или ответ пользователю"
          }

          Когда ты захочешь вернуть ответ пользователю или что либо уточнить необходимо указать 
          "next_step": "finale", "content": "сообщение которое отправить пользователю"
          
        ''')
    response = llm.invoke([system_prompt] + state.messages)
    
    # json_response = json.loads(response.content) # pyright: ignore[reportArgumentType]
    # print(json_response)
    return {
        'messages': response
    }

def rag_node(state: AgentState):
    return {'messages': ['hello rag']}

def human_node(state: AgentState):
    return {'messages': ['hello human']}

def decide_route(state: AgentState) -> str:

    last_message = state.messages[-1]

    if last_message.tool_calls:
        return 'tools'
    return 'end'
    # match state.current_agent:
    #     case 'document_agent':
    #         return 'rag'
    #     case 'finale':
    #         return 'end'
    #     case _:
    #         return 'agent'

graph = StateGraph(AgentState)
graph.add_node('agent', action=agent_node)
graph.add_node('rag', action=rag_node)
graph.add_node('human', action=human_node)
tool_node = ToolNode(tools=tools)
graph.add_node('tools', action=tool_node)

graph.add_edge(START, 'agent')
graph.add_conditional_edges(
    source='agent',
    path=decide_route,
    path_map={
        'tools': 'tools',
        'end': END
    }
)
graph.add_edge('tools', 'agent')
# graph.add_edge('rag', 'human')

# graph.add_edge('rag', 'agent')
# graph.add_edge('human', 'agent')
# graph.add_edge('tool', 'agent')

app = graph.compile()

init_state = AgentState(messages=[HumanMessage(content='сколько стоит товар №43')])

print(app.invoke(init_state))