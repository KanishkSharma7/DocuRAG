# DocuRAG - An intelligent document retrieval framework

Follow the steps below to setup the project and test it out!

1. Open the directory you want the repo in and then clone the repository using the link 
```python
git clone https://github.com/KanishkSharma7/DocuRAG.git
```
2. Open anaconda and run the following commands down below in the directory you have cloned the repository
If you do not have anaconda download it from the link (https://docs.anaconda.com/anaconda/install/)
```python
conda create -n rag_system 
conda activate rag_system
pip install -r requirements.txt

```
```python
git clone https://github.com/facebook/rocksdb.git
git clone https://github.com/huggingface/transformers.git
```

3. Add a data folder within the repo and add the file you want to ingest into RocksDB, make sure the file has only a .txt extension

4. Run the command below to ingest the data from your file into the vector database
```python
python build_vector_store.py
```
5. Run the command
```python
huggingface-cli login
```
Once prompted to enter the hugging face user access token, enter

7. Once the data is ingested run the command down below to open our chatbot
```python
streamlit run main.py
```
After running the command we see our UI, once you ask a question be patient it takes over 10 minutes to generate one answer 

