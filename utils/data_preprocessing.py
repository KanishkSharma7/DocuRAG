import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from utils.utils import load_documents
from utils.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    DATA_DIR,
    SQLITE_DB_PATH,
    SQLITE_DB_PATH_META,
    SQLITE_DB_PATH_CONTEXTUAL,
    SQLITE_DB_PATH_LATE
)
from utils.config import *
from pages import *

import uuid
from utils.late_chunking import *
from utils.vector_store import create_vector_store
from utils.meta_chunking import *
from utils.contextual_retrieval import *


def preprocess_data():
    documents = load_documents(DATA_DIR)

    print(f"Number of documents loaded: {len(documents)}")

    if not documents:
        raise ValueError("No documents found in the specified data directory.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    docs = text_splitter.split_documents(documents)
    
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    embeddings = embeddings_model.embed_documents([doc.page_content for doc in docs])
    print(f"Number of embeddings generated: {len(embeddings)}")  
    
    if not embeddings:
        raise ValueError("No embeddings generated for the document chunks.")
    
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT)')
    
    keys = []
    for doc in docs:
        key = str(uuid.uuid4())
        value = doc.page_content
        print(value)
        cursor.execute('INSERT INTO chunks (key, content) VALUES (?, ?)', (key, value))
        keys.append(key)
    
    conn.commit()
    conn.close()
    
    return keys, embeddings

def preprocess_data_meta_chunking(
    base_model: str = 'PPL Chunking',
    language: str = 'en',
    ppl_threshold: float = 0.5,
    chunk_length: int = 100
):
    documents = load_documents(DATA_DIR)

    print(f"Number of documents loaded: {len(documents)}")

    if not documents:
        raise ValueError("No documents found in the specified data directory.")

    all_chunks = []

    for doc in documents:
        original_text = doc.page_content


        chunked_text = meta_chunking(
            original_text,
            base_model,
            language,
            ppl_threshold,
            chunk_length
        )

        chunks = chunked_text.split('\n\n')

        for chunk in chunks:
            if chunk.strip():  
                new_doc = doc.copy()  
                new_doc.page_content = chunk.strip()
                all_chunks.append(new_doc)

    print(f"Number of chunks created: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("No chunks were created from the documents.")

    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


    # Generate embeddings for the chunks
    embeddings = embeddings_model.embed_documents([doc.page_content for doc in all_chunks])
    print(f"Number of embeddings generated: {len(embeddings)}")  

    if not embeddings:
        raise ValueError("No embeddings generated for the document chunks.")


    conn = sqlite3.connect(SQLITE_DB_PATH_META)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT)')

    keys = []
    for doc in all_chunks:
        key = str(uuid.uuid4())
        value = doc.page_content
        cursor.execute('INSERT INTO chunks (key, content) VALUES (?, ?)', (key, value))
        keys.append(key)

    conn.commit()
    conn.close()
    return keys, embeddings


def data_preprocess_context_retrieval():
    """ Documentation goes here """

    documents = load_documents(DATA_DIR)

    mergedDocuments = "\n".join([doc.page_content for doc in documents])

    contextualizedChunks = process_document(mergedDocuments)

    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in contextualizedChunks])
    print(f"Number of embeddings generated: {len(embeddings)}")

    connection = sqlite3.connect(SQLITE_DB_PATH_CONTEXTUAL)
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT)")

    keys = []

    for chunk in contextualizedChunks:
        key = str(uuid.uuid4())
        value = chunk.page_content
        cursor.execute("INSERT INTO chunks (key, content) VALUES (?, ?)", (key, value))
        keys.append(key)

    connection.commit()
    connection.close()

    return keys, embeddings

    # vectorStore = create_vectorstore(contextualizedChunks)

    # # Retrieve top-3 similar chunks
    # similarChunks = vectorStore.similarity_search(question, k=3)
    # answer = get_answer_from_llm(question, [doc.page_content for doc in similarChunks])

    # top3Chunks = [doc.page_content for doc in similarChunks]
    # topChunks = ""
    # count = 1
    # for chunk in top3Chunks:
    #     topChunks += str(count) + ". " + chunk + "\n\n"
    #     count += 1

    # return answer, topChunks

#     return keys, embeddings



def preprocess_data_late_chunking(chunk_size: int = 128):
    lc = LateChunking()
    late_embeddings, late_chunks, _ = lc.get_document_embeddings()
    
    conn = sqlite3.connect(SQLITE_DB_PATH_LATE)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT)')

    keys = []
    for doc in late_chunks.values():
        for value in doc:
            key = str(uuid.uuid4())
            cursor.execute('INSERT INTO chunks (key, content) VALUES (?, ?)', (key, value))
            keys.append(key)

    embeddings = []
    for doc in late_embeddings.values():
        for value in doc:
            vector = np.array(value).astype('float32')
            if len(vector) > VECTOR_DIMENSION:  
                vector = vector[:VECTOR_DIMENSION]
            elif len(vector) < VECTOR_DIMENSION:
                padding = np.zeros(VECTOR_DIMENSION - len(vector), dtype='float32')
                vector = np.concatenate([vector, padding])
            embeddings.append(vector)

    print("embeddings", len(embeddings[0]))
    conn.commit()
    conn.close()
    
    return keys, embeddings
