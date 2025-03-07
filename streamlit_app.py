import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import time

# Import LangChain modules
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory

# Import scraper functionality
from scraper import get_all_cleaned_markdown

# Load environment variables from file
@st.cache_resource
def load_api_key():
    load_dotenv('apikey.env')  # change to example.env if you are using the example file
    return os.getenv("open_ai_api_key")

# Configure page
st.set_page_config(
    page_title="Document Chat Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Document Chatbot")
st.markdown("Enter a URL to scrape and chat with the content!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "scraped_url" not in st.session_state:
    st.session_state.scraped_url = None

if "scraping_complete" not in st.session_state:
    st.session_state.scraping_complete = False

# URL input form
with st.form("url_form", clear_on_submit=False):
    url_input = st.text_input("Enter URL to scrape:", placeholder="https://example.com")
    submit_button = st.form_submit_button("Start Scraping")

    if submit_button and url_input:
        st.session_state.scraped_url = url_input
        st.session_state.scraping_complete = False
        st.session_state.messages = []

# Function to scrape and create chatbot
async def scrape_and_create_bot(url):
    with st.status("Scraping website...", expanded=True) as status:
        st.write(f"Scraping {url}")
        all_markdowns = await get_all_cleaned_markdown(url)
        
        st.write(f"Collected {len(all_markdowns)} markdown pages for embedding.")
        status.update(label="Creating embeddings...", state="running", expanded=True)
        
        # Convert each markdown page into a Document object
        documents = [Document(page_content=md) for md in all_markdowns]
        
        # Optionally split documents into smaller chunks for better retrieval
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=2000)
        docs = text_splitter.split_documents(documents)
        
        # Initialize the embedding model
        st.write("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create a vectorstore from the documents
        st.write("Creating vector store...")
        vectorstore = Chroma.from_documents(docs, embedding_model)
        
        # Get a retriever from the vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        st.session_state.retriever = retriever
        
        # Initialize ChatOpenAI (LLM)
        openai_api_key = load_api_key()
        llm = ChatOpenAI(
            temperature=0.2, 
            model_name="gpt-4o", 
            openai_api_key=openai_api_key
        )
        
        # Set up memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create ConversationalRetrievalChain
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        
        st.session_state.conversation = conversation
        status.update(label="Ready to chat!", state="complete")
        st.session_state.scraping_complete = True
        
    return True

# When URL is submitted, run the scraping process
if st.session_state.scraped_url and not st.session_state.scraping_complete:
    asyncio.run(scrape_and_create_bot(st.session_state.scraped_url))

# Display chat interface once scraping is complete
if st.session_state.scraping_complete:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new message
    if prompt := st.chat_input("Ask a question about the scraped content:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                result = st.session_state.conversation({"question": prompt})
                response = result["answer"]
            
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}) 