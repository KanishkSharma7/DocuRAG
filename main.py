import streamlit as st
import time
import psutil
from utils.retrieval import (
    answer_query, 
    answer_query_meta_chunking,
    answer_query_late_chunking,
    answer_query_contextual_chunking
)
import os
from utils.config import FAISS_INDEX_PATH, FAISS_INDEX_PATH_META, FAISS_INDEX_PATH_LATE

st.set_page_config(
    page_title="Retrieval-Augmented Generation System with Llama and RocksDB",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def get_response_with_error_handling(query_func, query, chunking_type):
    start_time = time.time()  # Start timer
    start_memory = psutil.Process().memory_info().rss  # Get memory at start
    
    try:
        response, _ = query_func(query)
        if not response:
            response = f"No response generated from {chunking_type}."
    except Exception as e:
        if "FileIOReader" in str(e):
            response = f"Vector store not initialized for {chunking_type}. Please check the index files."
        elif "Ran out of input" in str(e):
            response = f"The system is not properly initialized. Please check the vector store."
        else:
            response = str(e)
    
    end_time = time.time()  # End timer
    end_memory = psutil.Process().memory_info().rss  # Get memory at end
    
    time_taken = end_time - start_time  # Calculate time taken
    memory_used = (end_memory - start_memory) / (1024 * 1024)  # Convert to MB
    
    return response, time_taken, memory_used

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: white;
        }
        
        .chat-message {
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-radius: 0.5rem;
            background-color: #2d2d2d;
            width: 100%;
            min-height: 100px;
        }
        
        .user-message {
            border-left: 4px solid #ff4b4b;
            margin: 1rem 0;
            background-color: #2d2d2d;
        }
        
        .regular-message {
            border-left: 4px solid #00ff00;
        }
        
        .meta-message {
            border-left: 4px solid #0088ff;
        }

        .late-message {
            border-left: 4px solid #ff8800;
        }

        .context-message {
            border-left: 4px solid #010101;
        }
        
        .stTextInput > div > div > input {
            background-color: #2d2d2d;
            color: white;
            border: 1px solid #4a4a4a;
            border-radius: 5px;
            padding: 0.5rem;
            margin-top: 1rem;
        }
        
        h1, h2, h3 {
            color: white !important;
            margin-bottom: 1rem;
            padding: 0.5rem 0;
        }

        .response-container {
            margin-bottom: 2rem;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Retrieval-Augmented Generation System with Llama and RocksDB")

    # Display chat history
    for i in range(0, len(st.session_state.messages), 4):
        if i < len(st.session_state.messages):
            # User message
            st.markdown(
                f'<div class="chat-message user-message">{st.session_state.messages[i]["content"]}</div>',
                unsafe_allow_html=True
            )
            
            # Create response container
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            
            # Create three columns for responses
            col1, col2, col3, col4 = st.columns(4)
            
            # Regular chunking response
            with col1:
                st.markdown("<h3>Regular Chunking</h3>", unsafe_allow_html=True)
                if i + 1 < len(st.session_state.messages):
                    response, time_taken, memory_used = st.session_state.messages[i+1]["content"]
                    st.markdown(
                        f'<div class="chat-message regular-message">{response}<br><br><strong>Time Taken:</strong> {time_taken:.2f}s<br><strong>Memory Used:</strong> {memory_used:.2f} MB</div>',
                        unsafe_allow_html=True
                    )
            
            # Meta chunking response
            with col2:
                st.markdown("<h3>Meta Chunking</h3>", unsafe_allow_html=True)
                if i + 2 < len(st.session_state.messages):
                    response, time_taken, memory_used = st.session_state.messages[i+2]["content"]
                    st.markdown(
                        f'<div class="chat-message meta-message">{response}<br><br><strong>Time Taken:</strong> {time_taken:.2f}s<br><strong>Memory Used:</strong> {memory_used:.2f} MB</div>',
                        unsafe_allow_html=True
                    )

            # Late chunking response
            with col3:
                st.markdown("<h3>Late Chunking</h3>", unsafe_allow_html=True)
                if i + 3 < len(st.session_state.messages):
                    response, time_taken, memory_used = st.session_state.messages[i+3]["content"]
                    st.markdown(
                        f'<div class="chat-message late-message">{response}<br><br><strong>Time Taken:</strong> {time_taken:.2f}s<br><strong>Memory Used:</strong> {memory_used:.2f} MB</div>',
                        unsafe_allow_html=True
                    )
            
            # Contextual chunking response
            with col4:
                st.markdown("<h3>Contextual Retrieval</h3>", unsafe_allow_html=True)
                if i + 4 < len(st.session_state.messages):
                    response, time_taken, memory_used = st.session_state.messages[i+4]["content"]
                    st.markdown(
                        f'<div class="chat-message context-message">{response}<br><br><strong>Time Taken:</strong> {time_taken:.2f}s<br><strong>Memory Used:</strong> {memory_used:.2f} MB</div>',
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

    def handle_input():
        if st.session_state.user_input and st.session_state.user_input != st.session_state.current_input:
            user_input = st.session_state.user_input
            st.session_state.current_input = user_input
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Processing responses..."):
                # Get responses with error handling
                regular_response = get_response_with_error_handling(answer_query, user_input, "Regular Chunking")
                meta_response = get_response_with_error_handling(answer_query_meta_chunking, user_input, "Meta Chunking")
                late_response = get_response_with_error_handling(answer_query_late_chunking, user_input, "Late Chunking")
                context_response = get_response_with_error_handling(answer_query_contextual_chunking, user_input, "Contextual Retrieval")
                
                # Add responses to session state
                st.session_state.messages.extend([
                    {"role": "regular", "content": regular_response},
                    {"role": "meta", "content": meta_response},
                    {"role": "late", "content": late_response},
                    {"role": "context", "content": context_response}
                ])

    # Input field with callback
    st.text_input(
        "",
        placeholder="Enter your prompt...",
        key="user_input",
        on_change=handle_input
    )

if __name__ == "__main__":
    main()
