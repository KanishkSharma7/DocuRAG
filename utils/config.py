CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_DIMENSION = 384

#Language Model Settings
LLM_MODEL_PATH = 'Llama-3.2-1B-Instruct'
LLM_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'

#Paths and Directories
DATA_DIR = './data'
FAISS_INDEX_PATH = './faiss_index.idx'
INDEX_TO_KEY_PATH = './index_to_key.pkl'
ROCKSDB_PATH = './rocksdb_storage'
FILE_EXTENSION = 'txt'
SQLITE_DB_PATH = './chunks.db'

# Environment variables For Meta Chunking
FAISS_INDEX_PATH_META = './faiss_index_meta.idx'
INDEX_TO_KEY_PATH_META = './index_to_key_meta.pkl'
SQLITE_DB_PATH_META= './meta_chunks.db'

### OPENAI CONFIGURATIONS
OPENAI_API_KEY="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"

### CONTEXTUAL RETRIEVAL
SQLITE_DB_PATH_CONTEXTUAL = "./contextual_chunks.db"
INDEX_TO_KEY_PATH_CONTEXTUAL = './index_to_key_contextual.pkl'
FAISS_INDEX_PATH_CONTEXTUAL = './faiss_index_contextual.idx'

#Environment variables For Late Chunking
FAISS_INDEX_PATH_LATE = './faiss_index_late.idx'
INDEX_TO_KEY_PATH_LATE = './index_to_key_late.pkl'
SQLITE_DB_PATH_LATE= './late_chunks.db'

