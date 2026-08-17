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
def get_from_db(model: str, id: int):
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
        '''
            Ты самостоятельный AI ассистент технической поддержки в компании ритейлинга. Твоя главная задача - помочь обратившемся клиентам разобраться в их вопросе.
            Общайся грамотно, учтиво, вежливо, но не стандартизируй ответы, пользователи должны понимать ответы.
           У тебя есть несколько функций, которые ты можешь вызывать через function calling, tool_calls для обработки данных:
                1. Доступ к внутренней базе данных с информацией о клиентах, товарах, заказах.
                    Ты можешь получать данные из этих таблиц, а так же изменять их:
                        - users - возвращает id, активные заказы и процент скидки пользователя.
                        - products - возвращает id и цену (за единицу) товара
                        - orders - возвращает id и срок доставки заказа
        
            Помимо функций у тебя в подчинении есть AI ассистент - rag для поиска правил компании, как офлайн так и онлайн, используй его если пользователя интересует общая информация о работе компании. 
            
            У тебя есть возможность переспрашивать клиента, если у тебя недостаточно данных для решения вопроса.
        
            Твой ответ на любое сообщение должен быть в виде правильной JSON структуры, без вступлений, без outro, без любого дополнительного текста, без разметки markdown 
            JSON структура имеет 2 обязательных поля - next_agent, content
        
            значения которые может принимать поле next_agent
                rag - вызов аи ассистента для поиска по документации компании 
                finale - отправить сообщение content пользователю
        
            next_agent может быть НЕ finale только в одном случае - если у тебя есть все данные, чтобы сформулировать content как готовую команду/запрос для агента. Если твой content - это вопрос, адресованный пользователю, next_agent ВСЕГДА должен быть finale 
            
            пример твоего ответа 
            {
              "next_agent": "имя агента которому ты передаешь управление, или finale если ответ готов",
              "content": "Запрос к агенту, или ответ пользователю"
            }
        ''')
    response = llm.invoke([system_prompt] + state.messages)
    print(response)
    result = {'message': response}
    if response.content:
        json_response = json.loads(response.content) # pyright: ignore[reportArgumentType]
        print(json_response)
        result['current_agent'] = json_response.next_step
    return result

def rag_node(state: AgentState):
    return {'messages': ['hello rag']}

def human_node(state: AgentState):
    return {'messages': ['hello human']}

def decide_route(state: AgentState) -> str:

    last_message = state.messages[-1]

    if last_message.tool_calls: # pyright: ignore[reportAttributeAccessIssue]
        return 'tools'

    match state.current_agent:
        case 'document_agent':
            return 'rag'
        case 'finale':
            return 'end'

    return 'end'
    
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

init_state = AgentState(messages=[HumanMessage(content='до скольки открыт пункт выдачи?')])

print(app.invoke(init_state))