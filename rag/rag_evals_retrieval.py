import asyncio

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.evaluation import RetrieverEvaluator

from rag.rag_settings import chunk_splitter

documents = SimpleDirectoryReader('rag/data').load_data()


def generate_golden_dataset(nodes):
    golden_dataset = [] 
    for node in nodes:
        prompt = f'Generate a question that would be answered by this text \n TEXT \n {node.get_content()} \n Response only with 1 question, without any additional text and markdown.'
        question = Settings.llm.complete(prompt)
        golden_dataset.append({'node_id': node.node_id, 'question': question.text})
    return golden_dataset

def get_metrics_manual(index, dataset):
    retriever = index.as_retriever(similarity_top_k=3)
    hr = 0
    mrr = 0
    for data in dataset:
        nodes = retriever.retrieve(data['question'])
        for i, node in enumerate(nodes):
            if node.node_id == data['node_id']:
                hr += 1
                mrr += 1 / (i+1)
    return {'hit_rate': hr / len(dataset), 'mrr': mrr / len(dataset)}

async def get_metrics_via_evaluator(index, dataset):
    retriever = index.as_retriever(similarity_top_k=3)
    evaluator = RetrieverEvaluator.from_metric_names(
        ['hit_rate', 'mrr'], retriever=retriever
    )
    hr = 0
    mrr = 0
    for data in dataset:
        result = await evaluator.aevaluate(data['question'], expected_ids=[data['node_id']])
        hr += result.metric_vals_dict['hit_rate']
        mrr += result.metric_vals_dict['mrr']

    return {'hit_rate': hr / len(dataset), 'mrr': mrr / len(dataset)}


def retrieval_metrics():
    nodes = chunk_splitter(documents)
    dataset = generate_golden_dataset(nodes)
    index = VectorStoreIndex(nodes=nodes)

    # MANUAL
    print(get_metrics_manual(index, dataset))

    # LLAMAINDEX
    print(asyncio.run(get_metrics_via_evaluator(index, dataset)))

if __name__ == '__main__':
    retrieval_metrics()