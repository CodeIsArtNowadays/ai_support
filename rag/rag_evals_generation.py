import json

from rag.rag_graph import get_index
from rag.rag_settings import Settings


def generate_golden_dataset(nodes):
    golden_dataset = [] 
    for node in nodes:
        prompt = f'Generate a question that would be answered by this text \n TEXT \n {node.get_content()} \n Response only with 1 question, without any additional text and markdown.'
        question = Settings.llm.complete(prompt)
        golden_dataset.append(question.text)
    return golden_dataset


def geneva(nodes, index):
    
    query_engine = index.as_query_engine()
    golden_dataset = generate_golden_dataset(nodes)
    
    faithfulness = 0
    statements_amount = 0

    for question in golden_dataset:
        response = query_engine.query(question)
        
        statements_prompt = f'''
        Given a question and an answer, break the answer down into a set of
        standalone, fully understandable statements.
        
        Rules:
        - Each statement must make sense on its own, without needing the original text.
        - Replace all pronouns (he, it, this, they) with the actual noun/entity they refer to.
        - Do not add any information that is not present in the answer.
        - Do not merge unrelated facts into one statement.
        
        Respond only in JSON format, no intro, no outro, no markdown.
        Format: {{"statements": ["statement1", "statement2"]}}
        
        QUESTION: {question}
        ANSWER: {response.response}
        '''

        raw_response = Settings.llm.complete(statements_prompt).text
        statements = []
        try:
            statements = json.loads(raw_response)['statements']
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse JSON for question: {question}")
            print(f"Raw output: {raw_response}")
        context = '\n'.join([n.text for n in response.source_nodes])
        for statement in statements:
            faithfulness_prompt = f'''
            You have been given CONTEXT and AFFIRMATION. 
            Your task is to determine whether the AFFIRMATION can be logically deduced strictly from the CONTEXT. 
            Don't use your own knowledge of the world. 
            Answer 'Yes' if the statement is fully supported, and 'No' if there is insufficient information or contradiction in the context.
            CONTEXT:
                {context}
            AFFIRMATION:
                {statement}
            '''
            faithful_response = Settings.llm.complete(faithfulness_prompt)
            if faithful_response.text.strip().lower().startswith('yes'):
                faithfulness += 1
            statements_amount += 1
    return faithfulness / statements_amount

if __name__ == '__main__':
    
    index = get_index()
    nodes = index.vector_store.get_nodes(node_ids=None)[:5]
    print(geneva(nodes, index))