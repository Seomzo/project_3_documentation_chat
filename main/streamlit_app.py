import streamlit as st
import os
import time

# Import core functionality
from app import (
    load_api_key,
    run_async_scraper,
    create_vectorstore,
    get_retriever,
    create_conversation_chain,
    process_scraped_content,
    add_documents_to_vectorstore
)

# Import chat functionality
from chat_manager import (
    initialize_chat_state,
    should_stop_scraping,
    stop_scraping,
    update_scraping_status,
    reset_chat_only,
    reset_everything,
    display_chat_interface
)

# Configure page
st.set_page_config(
    page_title="Document Chat Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Document Chatbot")
st.markdown("Enter a URL to scrape and chat with the content!")

# Initialize session state variables if they don't exist
initialize_chat_state()
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "max_pages" not in st.session_state:
    st.session_state.max_pages = 100
if "should_stop_scraping" not in st.session_state:
    st.session_state.should_stop_scraping = False
if "scraping_status" not in st.session_state:
    st.session_state.scraping_status = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-mini"

# Function to scrape and create chatbot
def scrape_and_create_bot(url, max_pages=50):
    try:
        with st.status("Scraping website...", expanded=True) as status:
            st.write(f"Scraping {url}")
            
            # Reset the stop flag before starting
            st.session_state.should_stop_scraping = False
            st.session_state.scraping_in_progress = True
            
            # Run the async scraper
            try:
                all_markdowns = run_async_scraper(
                    url, 
                    max_pages, 
                    status_callback=update_scraping_status, 
                    stop_callback=should_stop_scraping
                )
                
                # Check if scraping was stopped
                if st.session_state.should_stop_scraping:
                    st.error("Scraping was stopped by user.")
                    st.session_state.scraping_complete = False
                    st.session_state.scraping_in_progress = False
                    return False
                    
                # Display the current status from session state
                if st.session_state.scraping_status:
                    st.write(st.session_state.scraping_status)
            except Exception as e:
                st.error(f"Error during scraping: {str(e)}")
                st.session_state.scraping_in_progress = False
                return False
            
            if not all_markdowns or len(all_markdowns) == 0:
                st.error("No content was scraped. Please try a different URL.")
                st.session_state.scraping_in_progress = False
                return False
                
            st.write(f"Collected {len(all_markdowns)} markdown pages for embedding.")
            status.update(label="Creating embeddings...", state="running", expanded=True)
            
            # Process the scraped content
            documents = process_scraped_content(all_markdowns)
            
            progress = st.progress(0.6, text="Creating vectorstore...")
            
            # Create vectorstore
            vectorstore = create_vectorstore(documents)
            
            # Store the vector store in session state for later use
            st.session_state.vectorstore = vectorstore
            
            # Get retriever
            retriever = get_retriever(vectorstore, k=5)
            
            # Initialize conversation chain
            st.session_state.conversation = create_conversation_chain(
                retriever=retriever, 
                api_key=st.session_state.api_key, 
                model_name=st.session_state.selected_model
            )
            
            st.session_state.chat_history = []
            st.session_state.scraping_done = True
            st.session_state.scraping_in_progress = False
            
            # Add welcome message
            st.session_state.messages = [
                {"role": "assistant", "content": f"Documentation from {url} has been processed. I'm ready to answer your questions about it!"}
            ]
            
            # Set state to indicate scraping is complete
            st.session_state.scraping_complete = True
            st.session_state.scraped_url = url
            
            status.update(label="Ready!", state="complete", expanded=False)
            
        return True
        
    except Exception as e:
        st.error(f"Error during embedding or chatbot creation: {str(e)}")
        st.session_state.scraping_in_progress = False
        return False

# Function to add documents to an existing bot
def add_documents_to_existing_bot(url, max_pages=50):
    """Add documents from a new URL to the existing vectorstore"""
    try:
        if not st.session_state.get("vectorstore"):
            st.error("No existing chatbot found. Please scrape a URL first.")
            return False
            
        with st.status(f"Adding content from {url}...", expanded=True) as status:
            # Reset the stop flag before starting
            st.session_state.should_stop_scraping = False
            st.session_state.scraping_in_progress = True
            
            # Run the async scraper
            all_markdowns = run_async_scraper(
                url, 
                max_pages, 
                status_callback=update_scraping_status, 
                stop_callback=should_stop_scraping
            )
            
            # Check if scraping was stopped
            if st.session_state.should_stop_scraping:
                st.error("Adding documents was stopped by user.")
                st.session_state.scraping_in_progress = False
                return False
                
            if not all_markdowns or len(all_markdowns) == 0:
                st.error("No content was scraped from the additional URL.")
                st.session_state.scraping_in_progress = False
                return False
                
            status.update(label=f"Processing {len(all_markdowns)} new documents...", state="running")
            
            # Process the content
            documents = process_scraped_content(all_markdowns)
            
            # Use the existing vectorstore to add these documents
            vectorstore = st.session_state.vectorstore
            
            # Log for debugging
            st.write(f"Adding {len(documents)} new documents to vectorstore")
            
            # Add documents to the vectorstore
            with st.spinner("Adding documents to vectorstore..."):
                add_documents_to_vectorstore(vectorstore, documents)
                st.success(f"Successfully added {len(documents)} documents to the vectorstore")
            
            # Persist the updated vectorstore back to session state
            st.session_state.vectorstore = vectorstore
            
            # Get a new retriever from the updated vectorstore
            retriever = get_retriever(vectorstore, k=4)
            
            status.update(label="Updating conversation chain...", state="running")
            
            # Update the conversation chain
            st.session_state.conversation = create_conversation_chain(
                retriever=retriever, 
                api_key=st.session_state.api_key, 
                model_name=st.session_state.selected_model
            )
            
            # Add message about the added content
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Added content from {url}. I now have knowledge from multiple sources and can answer questions about both!"}
            )
            
            # Update scraped_url to reflect multiple sources
            if isinstance(st.session_state.scraped_url, list):
                st.session_state.scraped_url.append(url)
            else:
                st.session_state.scraped_url = [st.session_state.scraped_url, url]
            
            status.update(label="Added successfully!", state="complete", expanded=False)
            st.session_state.scraping_in_progress = False
            
            return True
            
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        st.session_state.scraping_in_progress = False
        return False

# Create the sidebar with API key input, model selection, and scraping settings
def create_sidebar():
    st.sidebar.title("Documentation Chat")
    
    # Get API Key
    default_api_key = load_api_key()
    api_key_input = st.sidebar.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Your OpenAI API key starting with 'sk-'. This will be used for the chat functionality.",
        value=default_api_key
    )
    
    # Save API key to session state when entered
    if api_key_input:
        st.session_state.api_key = api_key_input
        if st.sidebar.button("Update API Key"):
            st.sidebar.success("API Key updated! ✅")
    
    # Model selection dropdown
    st.sidebar.subheader("Model Selection")
    model_options = {
        "GPT-4o": "gpt-4o",
        "GPT-4o Mini": "gpt-4o-mini",
        "o1": "o1",
        "o3": "o3",
        "o3 Mini": "o3-mini",
        "GPT-4.5": "gpt-4.5"
    }
    selected_model_name = st.sidebar.selectbox(
        "Select OpenAI Model:",
        options=list(model_options.keys()),
        index=1,  # Default to GPT-4o Mini
        help="Choose which OpenAI model to use for generating responses."
    )
    st.session_state.selected_model = model_options[selected_model_name]
    
    # Scraping settings
    st.sidebar.subheader("Scraping Settings")
    max_pages = st.sidebar.slider("Maximum pages to scrape:", 
                       min_value=10, max_value=1000, value=100, step=10,
                       help="Higher values will scrape more content but take longer.")
    st.session_state.max_pages = max_pages
    
    # Add "Add Another Doc" section to the sidebar - only show if initial scraping is complete
    if st.session_state.get("scraping_complete", False) and "vectorstore" in st.session_state:
        st.sidebar.subheader("Add Another Document")
        additional_url = st.sidebar.text_input(
            "Additional URL:", 
            placeholder="https://another-example.com/docs/", 
            key="sidebar_additional_url"
        )
        
        if st.sidebar.button("Add Document Source", disabled=not additional_url):
            if add_documents_to_existing_bot(additional_url, st.session_state.max_pages):
                st.sidebar.success(f"Successfully added content from {additional_url}")
                st.rerun()  # Refresh to show the updated chat
    
    # Add Reset Chat button at the bottom of the sidebar
    st.sidebar.markdown("---")
    
    # Add two separate reset buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Reset Chat Only", key="reset_chat_button"):
            if reset_chat_only():
                st.sidebar.success("Chat has been reset!")
                st.rerun()  # Force rerun to update the UI
    
    with col2:
        if st.button("Reset Everything", key="reset_all_button", type="primary"):
            if reset_everything():
                st.sidebar.success("Everything has been reset! Please scrape a new URL to begin.")
                st.rerun()
    
    st.sidebar.caption("This application uses OpenAI's models to process and respond to your queries. Your API key is required to use this functionality.")
    
    # create a button to exit the application\
    if st.sidebar.button("Exit Application", type="primary"):
        st.error("Shutting down the application...")  # Display shutdown message
        os._exit(0)  # Forcefully terminate the app
    

    
# Call the create_sidebar function to generate the sidebar UI
create_sidebar()

# URL Input section (only show if scraping not complete)
if not st.session_state.scraping_complete:
    # URL input with columns for buttons
    st.write("Enter a documentation website URL to create a specialized chat assistant.")
    
    url_input = st.text_input("URL:", placeholder="https://example.com/docs/")
    
    # Create columns for the buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # creare a start scraping button that is disabled if no API key or URL is entered but shows as enabled once both are entered
        # Note: The button is disabled if no API key or URL is provided
        st.session_state.api_key and url_input
        if st.button("Start Scraping", disabled=not (st.session_state.api_key and url_input)):
            # Start the scraping process in the background
            if scrape_and_create_bot(url_input, st.session_state.max_pages):
                st.rerun()




        # # Start scraping button (disabled if no API key or URL)
        # start_disabled = not (st.session_state.api_key and url_input)
        # if st.button("Start Scraping", disabled=start_disabled):
        #     st.session_state.scraped_url = url_input
        #     st.session_state.messages = []
            
            # Start the scraping process in the background
            if scrape_and_create_bot(url_input, st.session_state.max_pages):
                st.rerun()
    
    with col2:
        # Stop button (only enabled during scraping)
        stop_disabled = not st.session_state.scraping_in_progress
        if st.button("Stop Scraping", disabled=stop_disabled, type="secondary"):
            stop_scraping()
    
    # Show status messages
    if not st.session_state.api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    
    # Show current scraping status message
    if st.session_state.scraping_in_progress and st.session_state.scraping_status:
        st.info(st.session_state.scraping_status)

# Chat interface (only show when scraping is complete)
else:
    # Use the chat interface function from chat_manager
    display_chat_interface()
    
    # Show info about the scraped URL(s)
    with st.sidebar:
        if isinstance(st.session_state.scraped_url, list):
            st.subheader("📚 Documentation Sources")
            for i, url in enumerate(st.session_state.scraped_url):
                st.info(f"Source {i+1}: {url}")
            st.caption(f"I can answer questions about all {len(st.session_state.scraped_url)} documentation sources.")
        else:
            st.info(f"Currently answering questions about: {st.session_state.scraped_url}") 