import os
import asyncio
import tempfile
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

# Import scraper functionality
from scraper import get_all_cleaned_markdown

def load_api_key():
    """Load and return the OpenAI API key from environment variables."""
    # Check for Streamlit secrets (used in Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'openai' in st.secrets and 'OPENAI_API_KEY' in st.secrets['openai']:
            return st.secrets['openai']['OPENAI_API_KEY']
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
        if api_key:
            return api_key
    
    return None

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

def create_vectorstore(documents):
    """Create and return a vectorstore from the provided documents."""
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a temporary directory for the Chroma database
    persist_directory = tempfile.mkdtemp()
    
    try:
        # Create a vectorstore (Chroma) from the documents
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name="docs_collection" 
        )
    except Exception:
        # Fallback to in-memory Chroma if persistent storage fails
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
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
        temperature=0.2
    )
    
    # Create the conversation chain
    conversation = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return conversation

def process_scraped_content(all_markdowns):
    """Process raw markdown content into Document objects."""
    if not all_markdowns or len(all_markdowns) == 0:
        return None
        
    # Convert each markdown page into a Document object
    documents = [Document(page_content=md) for md in all_markdowns]
    return documents

def add_documents_to_vectorstore(vectorstore, documents):
    """Add new documents to an existing vectorstore."""
    if not documents or len(documents) == 0:
        return False
        
    # Add documents to the vectorstore
    vectorstore.add_documents(documents)
    return True

async def main():
    """Main function for standalone usage of the app."""
    # Get API key
    api_key = load_api_key()
    if not api_key:
        print("No API key found. Please add your OpenAI API key to apikey.env")
        return
    
    # Get URL from user
    url = input("Enter a URL to scrape: ")
    max_pages = int(input("Enter maximum number of pages to scrape (default 50): ") or "50")
    
    print(f"Starting to scrape {url}...")
    # Get the list of cleaned markdown pages
    all_markdowns = await get_all_cleaned_markdown(inputurl=url, max_pages=max_pages)
    print(f"Collected {len(all_markdowns)} markdown pages for embedding.")
    
    # Process the content
    documents = process_scraped_content(all_markdowns)
    if not documents:
        print("No content was scraped. Please try a different URL.")
        return
    
    # Create vectorstore
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(documents)
    
    # Get retriever
    retriever = get_retriever(vectorstore)
    
    # Create conversation chain
    conversation = create_conversation_chain(retriever, api_key)
    
    # Interactive chat loop
    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        try:
            response = conversation.invoke(query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error: {str(e)}")

# Only run main() if this file is executed directly, not when imported
if __name__ == "__main__":
    asyncio.run(main())