import streamlit as st
import os
import time
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import asyncio
import tempfile
import nest_asyncio
from dotenv import load_dotenv
import warnings

# Import LangChain modules
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Fix asyncio issues in Streamlit
nest_asyncio.apply()

# Configure page first - this must be the first st command
st.set_page_config(
    page_title="Document Chat Bot",
    page_icon="🤖",
    layout="wide"
)

# Playwright setup for Streamlit Cloud - now after page config
try:
    # Check if running in Streamlit Cloud
    if os.environ.get('STREAMLIT_SHARING', '') or os.path.exists('/home/appuser'):
        st.info("Running in Streamlit Cloud environment. Using fallback methods for web scraping if needed.")
        # Don't attempt to install browsers here - that will be handled during scraping
except Exception as e:
    st.warning(f"Error during environment setup: {str(e)}")

# Directly define the functions from app.py that we need
def load_api_key():
    """Load and return the OpenAI API key from environment variables."""
    # Check for Streamlit secrets (used in Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
            return st.secrets['openai']['api_key']
    except:
        pass
    
    # Try loading from various possible .env files
    for env_file in ['apikey.env', 'example.env', '.env']:
        try:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                break
        except:
            pass
    
    # Check various possible environment variable names for the API key
    for env_var in ["open_ai_api_key", "OPENAI_API_KEY", "OPENAI_KEY"]:
        api_key = os.getenv(env_var)
        if api_key and api_key != "sk-your-api-key-here":
            return api_key
    
    # If no valid API key is found, return None
    return None

async def fallback_scrape_without_browser(url, status_callback=None):
    """
    A fallback method to scrape content without using a browser.
    This is less effective but works when browser automation isn't available.
    """
    try:
        from urllib.parse import urljoin, urlparse
        
        if status_callback:
            status_callback(f"Using fallback scraping method for {url}")
        
        # Parse the base URL
        parsed_url = urlparse(url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Make the request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text and create markdown
        title = soup.title.string if soup.title else "No Title"
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        # Extract clean text
        if main_content:
            # Remove script and style elements
            for script in main_content.find_all(['script', 'style']):
                script.decompose()
                
            text = main_content.get_text(separator='\n\n')
        else:
            text = soup.get_text(separator='\n\n')
            
        # Clean up the text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n\n'.join(lines)
        
        # Create markdown
        markdown = f"# {title}\n\n{cleaned_text}"
        
        # Try to extract links for further scraping
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/') or href.startswith(base_domain):
                full_url = urljoin(base_domain, href)
                links.append(full_url)
                
        if status_callback:
            status_callback(f"Extracted {len(links)} links and {len(markdown)} characters of content")
            
        return markdown, links
    except Exception as e:
        if status_callback:
            status_callback(f"Error in fallback scraping: {str(e)}")
        return f"# Error scraping {url}\n\nError: {str(e)}", []

async def get_all_cleaned_markdown(inputurl=None, max_pages=100, status_callback=None, stop_callback=None):
    """
    Simple browser-free web scraping approach using requests and BeautifulSoup.
    Works well in Streamlit Cloud where browser automation isn't available.
    """
    if inputurl is None:
        inputurl = input("Enter the URL to crawl: ")
    
    if status_callback:
        status_callback("Using simplified scraping without browser automation")
    
    # Collect results without browser
    all_cleaned_markdown = []
    visited_urls = set()
    urls_to_visit = [inputurl]
    
    while urls_to_visit and len(all_cleaned_markdown) < max_pages:
        if stop_callback and stop_callback():
            if status_callback:
                status_callback("Scraping stopped by user.")
            break
            
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls:
            continue
            
        visited_urls.add(current_url)
        
        if status_callback:
            status_callback(f"Scraping {len(all_cleaned_markdown)+1}/{max_pages}: {current_url}")
        
        markdown, links = await fallback_scrape_without_browser(current_url, status_callback)
        all_cleaned_markdown.append(markdown)
        
        # Add new links to visit
        for link in links:
            if link not in visited_urls and link not in urls_to_visit:
                urls_to_visit.append(link)
        
        # Limit to max_pages
        if len(all_cleaned_markdown) >= max_pages:
            break
    
    if status_callback:
        status_callback(f"Completed scraping with {len(all_cleaned_markdown)} pages")
        
    return all_cleaned_markdown

def run_async_scraper(url, max_pages, status_callback=None, stop_callback=None):
    """Run the async scraper with a new event loop."""
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async function in the new loop
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = loop.run_until_complete(
                get_all_cleaned_markdown(
                    inputurl=url,
                    max_pages=max_pages,
                    status_callback=status_callback,
                    stop_callback=stop_callback
                )
            )
        return result
    finally:
        # Close the loop
        loop.close()

def process_scraped_content(all_markdowns):
    """Process raw markdown content into Document objects."""
    documents = []
    for i, markdown in enumerate(all_markdowns):
        doc = Document(
            page_content=markdown,
            metadata={"source": f"document_{i}", "index": i}
        )
        documents.append(doc)
    return documents

def create_vectorstore(documents):
    """Create a vector store from the documents."""
    # Create a temporary directory for the vector store
    persist_directory = tempfile.mkdtemp()
    
    # Create the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="docs_collection"
    )
    
    return vectorstore

def get_retriever(vectorstore, k=5):
    """Get a retriever from the vectorstore with the specified k value."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def create_conversation_chain(retriever, api_key, model_name="gpt-4o-mini"):
    """Create a conversation chain for the chatbot."""
    # Define the format_docs function
    def format_docs(docs):
        if not docs:
            return "No specific information about this was found in the documentation."
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the prompt template
    template = """You are a helpful assistant for answering questions about the documentation that has been provided to you.
    Use only the following context to answer the user's question. If you don't know the answer or the information is not in the context, say "I don't have information about that in the documentation, but here's what I know about related topics: [relevant information if available]. Would you like me to help with something else?"
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model_name=model_name, 
        openai_api_key=api_key,
        temperature=0.2,
        streaming=True  # Enable streaming for better UX
    )
    
    # Create the conversation chain
    conversation = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return conversation

def add_documents_to_vectorstore(vectorstore, documents):
    """Add new documents to an existing vectorstore."""
    vectorstore.add_documents(documents)
    return vectorstore

# Import chat functionality (we'll keep this because chat_manager.py doesn't depend on app.py)
from chat_manager import (
    initialize_chat_state,
    should_stop_scraping,
    stop_scraping,
    update_scraping_status,
    reset_chat_only,
    reset_everything,
    display_chat_interface
)

st.title("🤖 Document Chatbot")
st.markdown("Enter a URL to scrape and chat with the content!")

# Initialize session state variables if they don't exist
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
if "scraping_done" not in st.session_state:
    st.session_state.scraping_done = False
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
        # Start scraping button (disabled if no API key or URL)
        start_disabled = not (st.session_state.api_key and url_input)
        if st.button("Start Scraping", disabled=start_disabled):
            st.session_state.scraped_url = url_input
            st.session_state.messages = []
            
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