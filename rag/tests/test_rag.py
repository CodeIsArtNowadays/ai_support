import pytest
import pytest_asyncio

from llama_index.core import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader

from rag.rag_settings import chunk_splitter
from rag.rag_graph import get_index
from rag.rag_evals_retrieval import get_metrics_manual, generate_golden_dataset, get_metrics_via_evaluator
from rag.rag_evals_generation import geneva


@pytest.mark.asyncio(loop_scope="class")
class TestRAGRetrievalEvaluation:

    @pytest_asyncio.fixture(scope='class', autouse=True)
    @classmethod
    async def setup(cls, request):
        documents = SimpleDirectoryReader('rag/data').load_data()
        cls.nodes = chunk_splitter(documents)
        cls.dataset = generate_golden_dataset(request.cls.nodes)
        cls.index = VectorStoreIndex(nodes=request.cls.nodes)
        cls.manual_metrics = get_metrics_manual(request.cls.index, request.cls.dataset)
        cls.evaluator_metrics = await get_metrics_via_evaluator(request.cls.index, request.cls.dataset)

    async def test_generation_golden_dataset(self):
        assert len(self.dataset) > 0
    
    async def test_manual_metrics_is_close_to_evaluator(self):
        assert self.evaluator_metrics == pytest.approx(self.manual_metrics, abs=0.15)

    async def test_evaluator_metrics_is_passing_score(self):
        for metric, score in self.evaluator_metrics.items():
            assert score >= 0.6, f'{metric} score is: {score}'


@pytest.mark.asyncio(loop_scope="class")
class TestRAGGenerationEvaluation:

    @pytest_asyncio.fixture(scope='class', autouse=True)
    @classmethod
    async def setup(cls, request):
        cls.index = get_index()
        cls.nodes = cls.index.vector_store.get_nodes(node_ids=None)[:15]

    async def test_generation_faithfulness(self):
        assert geneva(self.nodes, self.index) > 0.6
    