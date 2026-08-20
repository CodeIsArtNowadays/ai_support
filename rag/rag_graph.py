from pathlib import Path

from chromadb import PersistentClient

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import AI_KEY
from rag.rag_settings import chunk_splitter, Settings


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
PERSIST_DIR = BASE_DIR / 'storage'


def load_storage_context() -> StorageContext:
    chroma_client = PersistentClient(path=str(PERSIST_DIR / 'chroma'))
    chroma_collection = chroma_client.get_or_create_collection('ragged')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if (PERSIST_DIR / 'docstore.json').exists():
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(PERSIST_DIR)
        )
    else:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
    return storage_context

def run_pipeline(storage_context: StorageContext, documents):
    pipeline = IngestionPipeline(
        transformations=[chunk_splitter, Settings.embed_model],
        documents=documents,
        vector_store=storage_context.vector_store,
        docstore=storage_context.docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS
    )
    nodes = pipeline.run(documents=documents)
    storage_context.persist(persist_dir=str(PERSIST_DIR))
    return nodes

def init_index() -> VectorStoreIndex:
    storage_context = load_storage_context()
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    run_pipeline(storage_context, documents)

    return VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        storage_context=storage_context
    ) 

def add_file(file_path: str) -> VectorStoreIndex:
    storage_context = load_storage_context()
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    run_pipeline(storage_context, documents)
    
    return VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        storage_context=storage_context
    )

def get_index():
    return init_index()


if __name__ == '__main__':
    index = get_index()
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query('If someone needs a GPU for gaming, which laptop should they buy and why?')
