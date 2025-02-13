from utils.utils import load_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import DATA_DIR, CHUNK_SIZE

import logging
import numpy as np
import requests
from transformers import AutoModel, AutoTokenizer
import torch


class LateChunking:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        self.embeddings = {}
        self.chunks = {}
        self.embeddings_traditional_chunking = {}


    #Get input from documents
    def get_document_embeddings(self):
        self.documents = load_documents(DATA_DIR)
        for doc in self.documents:
            text = doc.page_content
            metadata = doc.metadata
            file_name = metadata['source'].split('/')[-1]

            chunks, span_annotations = self.chunk_by_sentences(text)
            chunks, span_annotations = self.chunk_by_tokenizer_api(text)
            # print("=======================")
            # print('Chunks:\n- "' + '"\n- "'.join(chunks) + '"')
            self.chunks[file_name] = chunks

            #Chunk Before
            embeddings_traditional_chunking = self.model.encode(chunks)
            self.embeddings_traditional_chunking[file_name] = embeddings_traditional_chunking
            
            #Chunk afterwards 
            inputs = self.tokenizer(text, return_tensors='pt')
            model_output = self.model(**inputs)
            self.embeddings[file_name] = self.late_chunking(model_output, [span_annotations])[0]
    
        return self.embeddings, self.chunks, self.embeddings_traditional_chunking
            


    #Chunk by sentence
    def chunk_by_sentences(self, input_text: str):
        inputs = self.tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
        punctuation_mark_id = self.tokenizer.convert_tokens_to_ids('.')
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        token_offsets = inputs['offset_mapping'][0]
        token_ids = inputs['input_ids'][0]
        chunk_positions = [
            (i, int(start + 1))
            for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
            if token_id == punctuation_mark_id
            and (
                token_offsets[i + 1][0] - token_offsets[i][1] > 0
                or token_ids[i + 1] == sep_id
            )
        ]
        chunks = [
            input_text[x[1] : y[1]]
            for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        span_annotations = [
            (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
        ]
        return chunks, span_annotations
    
    #Chunk by api
    def chunk_by_tokenizer_api(self, input_text: str):
        """
        Use an external API to chunk the input text.

        Args:
            input_text (str): The text to be chunked.

        Returns:
            tuple: List of chunks and their positions.
        """
        url = 'https://tokenize.jina.ai/'
        payload = {
            "content": input_text,
            "return_chunks": "true",
            "max_chunk_length": CHUNK_SIZE
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an error for bad responses
            response_data = response.json()

            # Extract chunks and their positions
            chunks = response_data.get("chunks", [])
            chunk_positions = response_data.get("chunk_positions", [])
            span_annotations = [(start, end) for start, end in chunk_positions]

            return chunks, span_annotations

        except requests.exceptions.RequestException as e:
            print(f"Error while calling chunking API: {e}")
            return [], []

    #Late chunking
    def late_chunking(self, model_output: 'BatchEncoding', span_annotations: list, max_length=None):
        token_embeddings = model_output[0]
        outputs = []
        for embeddings, annotations in zip(token_embeddings, span_annotations):
            if (max_length is not None):  # remove annotations which go bejond the max-length of the model
                annotations = [(start, min(end, max_length - 1)) for (start, end) in annotations
                    if start < (max_length - 1)]
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]
            pooled_embeddings = [
                embedding.detach().cpu().numpy() for embedding in pooled_embeddings
            ]
            outputs.append(pooled_embeddings)

        return outputs  
    
    #Save the embedding to vectorDB
    #Display the output to frontend

    
    #TEST
    def test(self):
        cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        paris_embedding = self.model.encode('Paris')

        for chunk, new_embedding, trad_embeddings in zip(self.chunks['test1.txt'], self.embeddings['test1.txt'], self.embeddings_traditional_chunking['test1.txt']):
            print(f'similarity_new("Paris", "{chunk}"):', cos_sim(paris_embedding, new_embedding))
            print(f'similarity_trad("Paris", "{chunk}"):', cos_sim(paris_embedding, trad_embeddings))
    
