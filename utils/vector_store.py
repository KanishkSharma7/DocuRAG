import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from utils.config import *
from utils.utils import *
from langchain.vectorstores import FAISS
import nltk

nltk.download('punkt')

def create_vector_store(keys, embeddings):
    embeddings = np.array(embeddings).astype('float32')
    print(embeddings.shape)
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, VECTOR_DIMENSION)
    elif embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array with shape (num_documents, embedding_dimension).")

    if embeddings.size == 0:
        raise ValueError("No embeddings to add to the Faiss index.")

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings)
    
    index_to_key = {i: key for i, key in enumerate(keys)}
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    save_faiss_index(index_to_key)  
    
    return index

def load_vector_store():
    """Load the FAISS index from disk."""
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

def load_index_to_key():
    """Load the index_to_key mapping from disk."""
    return load_faiss_index(INDEX_TO_KEY_PATH)

# Create a vector store using Meta Chunking
def create_vector_store_meta(keys, embeddings):
    embeddings = np.array(embeddings).astype('float32')
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, VECTOR_DIMENSION)
    elif embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array with shape (num_documents, embedding_dimension).")

    if embeddings.size == 0:
        raise ValueError("No embeddings to add to the Faiss index.")

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings)
    
    index_to_key = {i: key for i, key in enumerate(keys)}
    
    faiss.write_index(index, FAISS_INDEX_PATH_META)
    save_faiss_index_meta(index_to_key)  
    
    return index

def load_vector_store_meta():
    """Load the FAISS index from disk."""
    index = faiss.read_index(FAISS_INDEX_PATH_META)
    return index

def load_index_to_key_meta():
    """Load the index_to_key mapping from disk."""
    return load_faiss_index_meta(INDEX_TO_KEY_PATH_META)
  
  #================================================CONTEXT RE========================================


# Create a vector store using Contextual Retrieval
def create_vector_store_contextual(keys, embeddings):
    embeddings = np.array(embeddings).astype('float32')
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, VECTOR_DIMENSION)
    elif embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array with shape (num_documents, embedding_dimension).")

    if embeddings.size == 0:
        raise ValueError("No embeddings to add to the Faiss index.")

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings)
    
    index_to_key = {i: key for i, key in enumerate(keys)}
    
    faiss.write_index(index, FAISS_INDEX_PATH_CONTEXTUAL)
    save_faiss_index_contextual(index_to_key)  
    
    return index
  
def load_vector_store_contextual():
    """Load the FAISS index from disk."""
    index = faiss.read_index(FAISS_INDEX_PATH_CONTEXTUAL)
    return index

def load_index_to_key_contextual():
    """Load the index_to_key mapping from disk."""
    return load_faiss_index_contextual(INDEX_TO_KEY_PATH_CONTEXTUAL) 
 
# Create a vector store using Late Chunking
def create_vector_store_late(keys, embeddings):
    embeddings = np.array(embeddings).astype('float32')
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, VECTOR_DIMENSION)
    elif embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array with shape (num_documents, embedding_dimension).")

    if embeddings.size == 0:
        raise ValueError("No embeddings to add to the Faiss index.")

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    index.add(embeddings)
    
    index_to_key = {i: key for i, key in enumerate(keys)}
    faiss.write_index(index, FAISS_INDEX_PATH_LATE)
    save_faiss_index_late(index_to_key)  
    
    return index

def load_vector_store_late():
    """Load the FAISS index from disk."""
    index = faiss.read_index(FAISS_INDEX_PATH_LATE)
    return index

def load_index_to_key_late():
    """Load the index_to_key mapping from disk."""
    return load_faiss_index_late(INDEX_TO_KEY_PATH_LATE)
