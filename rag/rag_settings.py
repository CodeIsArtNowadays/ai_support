from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter


Settings.llm = Ollama(
    model='llama3.2:3b',
    api_base='http://localhost:11434',
    api_key='123',
    context_window=4096,
    request_timeout=500
)
 
Settings.embed_model = OllamaEmbedding(
    base_url='http://localhost:11434',
    model_name='qwen3-embedding:0.6b'
)

chunk_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)