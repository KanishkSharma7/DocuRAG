import uuid
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import *



def generate_context(document, chunk):
    """ Generate context for the chunk using LLM """
    prompt = ChatPromptTemplate.from_template("""
        Given the document and a chunk of the document, your task is to generate a brief and relevant context for the chunk.
        Here is the document:
        {document}
                                              
        Here is the chunk:
        {chunk}
                                              
        Generate a brief context in 3-4 sentences for this chunk using the following:
        1. In the response, use the context directly. Do not use phrases like "This sentence mentions" or "This section discusses".
        2. Always include the metrics and figures (values, date, percentages) that provide the important context.
        3. Clearly describe the context of the chunk with the pronouns replaced with nouns. The response must be able to clearly explain the context of the chunk with regards to the entire document.
        4. Only provide the context and nothing else.
                                              
        Please generate a short context to situate this chunk within the overall document to improve the search retrieval of the chunk.                                          
    """)
    model = ChatOpenAI(model="gpt-3.5-turbo", max_retries=2, openai_api_key=OPENAI_API_KEY)
    messages = prompt.format_messages(document=document, chunk=chunk)
    response = model.invoke(messages)
    return response.content


def create_contextualized_chunks(document, chunks):
    """ Generate contextualized chunks for the chunks passed in arguments """
    contextualizedChunks = []
    for chunk in chunks:
        context = generate_context(document, chunk.page_content)
        contextualizedChunk = f"{context}\n\n{chunk.page_content}"
        contextualizedChunks.append(Document(page_content=contextualizedChunk, metadata=chunk.metadata))
    return contextualizedChunks


def process_document(document):
    """ Split the documents into chunks and generate contexts for those """
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = textSplitter.create_documents([document])
    contextualizedChunks = create_contextualized_chunks(document, chunks)
    return contextualizedChunks


def create_vectorstore(chunks):
    """ Create a vector store from the chunks """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embeddings)


def generate_key():
    """ Generate a unique hash-key """
    return str(uuid.uuid4())


def get_answer_from_llm(question, similarChunks):
    """ From the top-k similar chunks extract the answer for the question using LLM """
    prompt = ChatPromptTemplate.from_template("""
    Given some information and a question, please answer the question accurately by only using the information provided.
    If the information provided is not sufficient to answer the question, respond that you do not have enough information to answer the question.
                                              
    Instructions:
    1. Only consider the information provided to answer the question.
    2. Do not consider any other information to answer the question, except for the information given.
    3. If you do not find an answer or unable to answer the question using the information, Say that you do not have sufficient information to answer the question.
    
    Question: {question}
    
    Relevant information:
    {similarChunks}
    """)
    model = ChatOpenAI(model="gpt-3.5-turbo", max_retries=2, openai_api_key=OPENAI_API_KEY)
    messages = prompt.format_messages(question=question, similarChunks="\n\n".join(similarChunks))
    response = model.invoke(messages)
    return response.content
