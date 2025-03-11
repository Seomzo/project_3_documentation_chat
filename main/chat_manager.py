import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional, Dict, Any, Union

# Import the consolidated create_conversation_chain from app.py
from app import create_conversation_chain

def initialize_chat_state():
    """Initialize all chat-related session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "scraped_url" not in st.session_state:
        st.session_state.scraped_url = None
    if "scraping_complete" not in st.session_state:
        st.session_state.scraping_complete = False
    if "scraping_in_progress" not in st.session_state:
        st.session_state.scraping_in_progress = False

def should_stop_scraping():
    """Check if the scraping process should be stopped."""
    return st.session_state.should_stop_scraping

def stop_scraping():
    """Stop the scraping process and show notification."""
    st.session_state.should_stop_scraping = True
    st.toast("Stopping scraping process...", icon="🛑")

def update_scraping_status(status):
    """Update the scraping status in the session state."""
    st.session_state.scraping_status = status


def reset_chat_only():
    """Reset only the chat messages while preserving the vectorstore and settings."""
    # Preserve API key, model selection, vectorstore
    api_key = st.session_state.api_key if "api_key" in st.session_state else ""
    selected_model = st.session_state.selected_model if "selected_model" in st.session_state else "gpt-4o-mini"
    max_pages = st.session_state.max_pages if "max_pages" in st.session_state else 100
    vectorstore = st.session_state.vectorstore if "vectorstore" in st.session_state else None
    
    # Keep the retrieval chain but reset chat messages
    if vectorstore:
        # Get a retriever from the existing vectorstore
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        # Create a new conversation chain
        st.session_state.conversation = create_conversation_chain(retriever, api_key, selected_model)
    
    # Reset chat-related state
    for key in ["messages", "chat_history"]:
        if key in st.session_state:
            del st.session_state[key]
    
    # Initialize empty messages
    st.session_state.messages = []
    st.session_state.chat_history = []
    
    # Keep the vectorstore and restore settings
    st.session_state.api_key = api_key
    st.session_state.selected_model = selected_model
    st.session_state.max_pages = max_pages
    st.session_state.vectorstore = vectorstore
    
    return True

def reset_everything():
    """Reset all session state variables except for API key and model settings."""
    # Preserve only API key and model selection
    api_key = st.session_state.api_key if "api_key" in st.session_state else ""
    selected_model = st.session_state.selected_model if "selected_model" in st.session_state else "gpt-4o-mini"
    max_pages = st.session_state.max_pages if "max_pages" in st.session_state else 100
    
    # Reset all state variables
    for key in list(st.session_state.keys()):
        if key not in ["api_key", "selected_model", "max_pages"]:
            del st.session_state[key]
    
    # Reinitialize essential state
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.scraping_complete = False
    st.session_state.scraping_in_progress = False
    
    # Restore minimal settings
    st.session_state.api_key = api_key
    st.session_state.selected_model = selected_model
    st.session_state.max_pages = max_pages
    
    return True

def display_chat_interface():
    """Display the chat interface with message history and input."""
    # Display conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for user questions
    if prompt := st.chat_input("Ask a question about the documentation:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from the conversation chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the conversation chain
                    response = st.session_state.conversation.invoke(prompt)
                    
                    # Display the response
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error: {error_msg}. Please try again or ask a different question."})