import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from file.
load_dotenv('apikey.env') # change to example.env if you are using the example file.
openai_api_key = os.getenv("open_ai_api_key")

# Import LangChain modules.
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Import your asynchronous function from test.py that returns the markdown pages.
from scraper import get_all_cleaned_markdown

async def main():
    # Get the list of cleaned markdown pages from test.py.
    all_markdowns = await get_all_cleaned_markdown()
    print(f"Collected {len(all_markdowns)} markdown pages for embedding.")
    
    # Convert each markdown page into a Document object.
    documents = [Document(page_content=md) for md in all_markdowns]
    
    # Optionally split documents into smaller chunks for better retrieval.
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=2000)
    docs = text_splitter.split_documents(documents)
    
    # Initialize the embedding model.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a vectorstore (Chroma) from the documents.
    vectorstore = Chroma.from_documents(docs, embedding_model)
    
    # Get a retriever from the vectorstore.
    retriever = vectorstore.as_retriever()
    
    # Initialize your LLM (here using GPT-4 model via OpenAI).
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
    
    # Create a Retrieval-Augmented Generation (RAG) chain.
    rag_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    
    # Get a query from the user and run the RAG chain.
    query = input("Enter your query: ")
    response = rag_chain.run(query)
    print("Response:", response)

if __name__ == "__main__":
    asyncio.run(main())