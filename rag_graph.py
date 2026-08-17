from chromadb import Client

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from config import AI_KEY

Settings.llm = OpenAI(
    model='gpt-4.1-nano',
    api_base='https://api.proxyapi.ru/openai/v1',
    api_key=AI_KEY
)
Settings.embed_model = OpenAIEmbedding(
    model='text-embedding-3-small',
    api_base='https://api.proxyapi.ru/openai/v1',
    api_key=AI_KEY
)


# chroma_client = PersistentClient(path='./data/chroma')
chroma_client = Client()
chroma_collection = chroma_client.get_or_create_collection('ragged')

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader('data').load_data()

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

query_engine = index.as_query_engine()
print(query_engine.query('If someone needs a GPU for gaming, which laptop should they buy and why?'))