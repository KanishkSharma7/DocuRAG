import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.data_preprocessing import data_preprocess_context_retrieval
from utils.retrieval import answer_query_contextual_chunking

def main():
    st.title("Contextual Retrieval System")
    
    st.write("Welcome to the Contextual Retrieval System. Please enter your question below, and our system will retrieve the most relevant context to provide you with an accurate answer.")
    
    # User input for the query
    query = st.text_input("Enter your question:")

    # Action on submit button
    if st.button("Submit"):
        if query:
            with st.spinner("Retrieving the most relevant context..."):
                try:
                    # Retrieve answer for the given query
                    answer, similarChunks = answer_query_contextual_chunking(query)
                    st.write("**Answer:**")
                    st.write(answer)
                    st.write("**Chunks most relevant to the question used to generate the answer:**")
                    st.write(similarChunks)
                except Exception as e:
                    st.error(f"An error occurred while processing your request: {e}")
        else:
            st.warning("Please enter a question to proceed.")

if __name__ == "__main__":
    main()