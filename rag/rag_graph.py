from chromadb import Client

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import AI_KEY
from rag.rag_settings import chunk_splitter


# chroma_client = PersistentClient(path='./data/chroma')
chroma_client = Client()
chroma_collection = chroma_client.get_or_create_collection('ragged')

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader('rag/data').load_data()


index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    transformations=[chunk_splitter]
)

query_engine = index.as_query_engine()
print(query_engine.query('If someone needs a GPU for gaming, which laptop should they buy and why?'))
