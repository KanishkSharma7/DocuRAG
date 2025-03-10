# import torch
# from transformers import pipeline
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from utils.vector_store import load_vector_store, load_index_to_key
# from langchain.docstore.in_memory import InMemoryDocstore
# from langchain.docstore.document import Document
# from utils.config import EMBEDDING_MODEL, SQLITE_DB_PATH, LLM_MODEL
# import sqlite3

# def chatbot_pipeline():
#     model_id = "meta-llama/Llama-3.2-1B-Instruct"
#     pipe = pipeline(
#         "text-generation",
#         model=model_id,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )
#     return pipe

# def generate_response(pipe, context, query):
#     messages = [
#         {"role": "system", "content": "You are a helpful chatbot who always responds to the best of your abilities!"},
#         {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
#     ]
#     outputs = pipe(messages, max_new_tokens=512)
#     return outputs[0]["generated_text"]

# def get_retriever():
#     index = load_vector_store()
#     index_to_key = load_index_to_key()
#     embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     docstore = InMemoryDocstore({
#         str(key): Document(page_content="", metadata={"key": str(key)})
#         for key in index_to_key.values()
#     })
#     vector_store = FAISS(
#         embedding_function=embeddings_model.embed_query,
#         index=index,
#         docstore=docstore,
#         index_to_docstore_id=index_to_key
#     )
#     return vector_store.as_retriever()

# def fetch_chunks(keys):
#     conn = sqlite3.connect(SQLITE_DB_PATH)
#     cursor = conn.cursor()
#     chunks = []
#     for key in keys:
#         cursor.execute('SELECT content FROM chunks WHERE key=?', (key,))
#         result = cursor.fetchone()
#         if result:
#             chunks.append(result[0])
#     conn.close()
#     return chunks

# def answer_query(query):
#     retriever = get_retriever()
#     docs_and_scores = retriever.get_relevant_documents(query)
#     keys = [doc.metadata['key'] for doc in docs_and_scores]
#     chunks = fetch_chunks(keys)
#     context = ' '.join(chunks)

#     pipe = chatbot_pipeline()

#     answer = generate_response(pipe, context, query)
#     return answer
import openai

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from utils.vector_store import *
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from transformers import AutoModel, AutoTokenizer
from utils.config import *

import sqlite3
import requests

# Set OpenAI API Key

openai.api_key = ""

def openai_chatbot_pipeline():
    def generate_response(context, query):
        prompt = f"""
        You are a helpful chatbot who always responds to the best of your abilities!
        
        Context: {context}
        
        Question: {query}
        
        Answer:"""
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        print(response)
        content = response.choices[0].message.content.strip()
        return content


    return generate_response

def get_retriever():
    index = load_vector_store()
    index_to_key = load_index_to_key()
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docstore = InMemoryDocstore({
        str(key): Document(page_content="", metadata={"key": str(key)})
        for key in index_to_key.values()
    })
    vector_store = FAISS(
        embedding_function=embeddings_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_key
    )
    return vector_store.as_retriever()

def fetch_chunks(keys):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    chunks = []
    for key in keys:
        cursor.execute('SELECT content FROM chunks WHERE key=?', (key,))
        result = cursor.fetchone()
        if result:
            chunks.append(result[0])
    conn.close()
    return chunks

def answer_query(query):
    retriever = get_retriever()
    docs_and_scores = retriever.get_relevant_documents(query)
    keys = [doc.metadata['key'] for doc in docs_and_scores]
    chunks = fetch_chunks(keys)
    context = ' '.join(chunks)

    generate_response = openai_chatbot_pipeline()

    answer = generate_response(context, query)
    return answer, context

############################################# Retrieval functions for Meta Chunking  ###########################################################
def fetch_chunks_meta(keys):
    conn = sqlite3.connect(SQLITE_DB_PATH_META)
    cursor = conn.cursor()
    chunks = []
    for key in keys:
        cursor.execute('SELECT content FROM chunks WHERE key=?', (key,))
        result = cursor.fetchone()
        if result:
            chunks.append(result[0])
    conn.close()
    return chunks

def get_retriever_meta_chunking():
    index = load_vector_store_meta()
    index_to_key = load_index_to_key_meta()
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docstore = InMemoryDocstore({
        str(key): Document(page_content="", metadata={"key": str(key)})
        for key in index_to_key.values()
    })
    vector_store = FAISS(
        embedding_function=embeddings_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_key
    )
    return vector_store.as_retriever()

def answer_query_meta_chunking(query):
    retriever = get_retriever_meta_chunking()
    docs_and_scores = retriever.get_relevant_documents(query)
    keys = [doc.metadata['key'] for doc in docs_and_scores]
    chunks = fetch_chunks_meta(keys)
    context = ' '.join(chunks)

    generate_response = openai_chatbot_pipeline()

    answer = generate_response(context, query)
    return answer, context




############################################# Retrieval functions for Contxetual Retrieval  ###########################################################
def fetch_chunks_contextual(keys):
    conn = sqlite3.connect(SQLITE_DB_PATH_CONTEXTUAL)
    cursor = conn.cursor()
    chunks = []
    for key in keys:
        cursor.execute('SELECT content FROM chunks WHERE key=?', (key,))
        result = cursor.fetchone()
        if result:
            chunks.append(result[0])
    conn.close()
    return chunks
  

def get_retriever_contextual_chunking():
    index = load_vector_store_contextual()
    index_to_key = load_index_to_key_contextual()
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docstore = InMemoryDocstore({
        str(key): Document(page_content="", metadata={"key": str(key)})
        for key in index_to_key.values()
    })
    vector_store = FAISS(
        embedding_function=embeddings_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_key
    )
    return vector_store.as_retriever()
  
def answer_query_contextual_chunking(query):
    retriever = get_retriever_contextual_chunking()
    docs_and_scores = retriever.invoke(query)
    keys = [doc.metadata['key'] for doc in docs_and_scores]
    chunks = fetch_chunks_contextual(keys)

    topChunks = ""
    count = 1
    for chunk in chunks:
        topChunks += str(count) + ". " + chunk + "\n\n"
        count += 1

    generate_response = openai_chatbot_pipeline()

    answer = generate_response(topChunks, query)
    return answer, topChunks

    



###################################################Reteriver function for Late Chunking###########################################################
def fetch_chunks_late(keys):
    conn = sqlite3.connect(SQLITE_DB_PATH_LATE)
    cursor = conn.cursor()
    chunks = []
    for key in keys:
        cursor.execute('SELECT content FROM chunks WHERE key=?', (key,))
        result = cursor.fetchone()
        if result:
            chunks.append(result[0])
    conn.close()
    return chunks


def get_retriever_late_chunking():
    index = load_vector_store_late()
    index_to_key = load_index_to_key_late()
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docstore = InMemoryDocstore({
        str(key): Document(page_content="", metadata={"key": str(key)})
        for key in index_to_key.values()
    })
    vector_store = FAISS(
        embedding_function=embeddings_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_key
    )
    return vector_store.as_retriever()

def get_retriever():
    index = load_vector_store()
    index_to_key = load_index_to_key()
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docstore = InMemoryDocstore({
        str(key): Document(page_content="", metadata={"key": str(key)})
        for key in index_to_key.values()
    })
    vector_store = FAISS(
        embedding_function=embeddings_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_key
    )
    return vector_store.as_retriever()



def answer_query_late_chunking(query):
    retriever = get_retriever_late_chunking()
    docs_and_scores = retriever.get_relevant_documents(query)
    keys = [doc.metadata['key'] for doc in docs_and_scores]
    chunks = fetch_chunks_late(keys)
    context = ' '.join(chunks)

    generate_response = openai_chatbot_pipeline()

    answer = generate_response(context, query)
    return answer, context
