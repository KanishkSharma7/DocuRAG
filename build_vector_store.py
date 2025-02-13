from enum import Enum
from utils.data_preprocessing import *
from utils.vector_store import create_vector_store, create_vector_store_meta, create_vector_store_contextual, create_vector_store_late
from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

class ChunkingMethod(Enum):
    REGULAR = "regular"
    META = "meta"
    CONTEXTUAL = "context"
    LATE = "late"


@dataclass
class ChunkingConfig:
    base_model: str = 'PPL Chunking'
    language: str = 'en'
    ppl_threshold: float = 0.5
    chunk_length: int = 100

def process_with_chunking(config: ChunkingConfig, method: ChunkingMethod) -> Tuple[List[str], List[np.ndarray]]:
    """
    Process documents using the specified chunking method
    """
    if method == ChunkingMethod.REGULAR:
        return preprocess_data()
    elif method == ChunkingMethod.CONTEXTUAL:
        return data_preprocess_context_retrieval()
    elif method == ChunkingMethod.META:
        return preprocess_data_meta_chunking(
            base_model=config.base_model,
            language=config.language,
            ppl_threshold=config.ppl_threshold,
            chunk_length=config.chunk_length
        )
    elif method == ChunkingMethod.LATE:
        return preprocess_data_late_chunking()

if __name__ == "__main__":
    # Define a common configuration for chunking
    config = ChunkingConfig(
        # method=ChunkingMethod.META,  # Change to ChunkingMethod.REGULAR for regular chunking
        base_model='PPL Chunking',
        language='en',
        ppl_threshold=0.5,
        chunk_length=100
    )

    method = ChunkingMethod.CONTEXTUAL

    keys, embeddings = process_with_chunking(config, method)
        
    if method == ChunkingMethod.REGULAR:
        create_vector_store(keys, embeddings)
    elif method == ChunkingMethod.META:
        create_vector_store_meta(keys, embeddings)
    elif method == ChunkingMethod.CONTEXTUAL:
        create_vector_store_contextual(keys, embeddings)
    elif method == ChunkingMethod.LATE:
        create_vector_store_late(keys, embeddings)


    print(f"Finished processing with {method.value} chunking method.\n")