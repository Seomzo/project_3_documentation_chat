# Documentation Chat Bot

A powerful Streamlit-based chat application that lets you scrape any documentation website and have interactive conversations about its content. The application uses advanced web scraping, natural language processing, and retrieval-augmented generation to provide accurate answers from documentation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)

## Overview

Documentation Chat Bot transforms technical documentation into an interactive knowledge base. Simply provide the URL of any documentation website, and the application will scrape its content, process it, and create a specialized chatbot that can answer your specific questions about that documentation.

## Features

- **Intelligent Web Scraping**: Automatically crawls documentation websites, including subpages
- **Multi-stage Crawling**: Uses sitemaps and follows links to maximize content coverage
- **Markdown Processing**: Converts web content to clean, structured markdown format
- **Vector Embeddings**: Creates searchable vector representations of content
- **RAG Architecture**: Uses retrieval-augmented generation for accurate responses
- **Interactive Chat**: Clean, user-friendly chat interface
- **Multiple Document Sources**: Ability to add multiple documentation sources to the same chatbot
- **Configurable Parameters**: Adjust scraping depth, model selection, and more
- **API Key Management**: Use your own OpenAI API key
- **Chat Management**: Reset chats while preserving knowledge base

## Technology Stack

The application leverages several powerful technologies:

- **crawl4ai**: Advanced web crawling and scraping for documentation sites
- **LangChain**: Framework for building context-aware AI applications
- **Chroma DB**: Vector database for storing and retrieving document embeddings
- **HuggingFace**: Sentence transformer models for text embeddings
- **OpenAI**: GPT models for natural language understanding and generation
- **Streamlit**: Web application framework for the user interface
- **BeautifulSoup**: HTML parsing for extracting links and content
- **asyncio**: Asynchronous programming for efficient web scraping

## How It Works

The application follows a sophisticated pipeline:

1. **Web Scraping Phase**:
   - First attempts to find and process a sitemap.xml for comprehensive URL discovery
   - Falls back to direct page crawling if no sitemap is available
   - Uses a three-stage crawling approach to maximize content coverage
   - Employs BeautifulSoup and regex patterns to extract and clean links

2. **Content Processing**:
   - Converts HTML to clean markdown format
   - Removes navigation elements and standardizes formatting
   - Creates document objects from the processed content

3. **Vector Database Creation**:
   - Generates embeddings for each document using HuggingFace models
   - Stores embeddings in a Chroma vector database
   - Creates an efficient retrieval system for finding relevant content

4. **Chat Interface**:
   - Presents a clean Streamlit chat interface
   - Handles user queries and manages conversation state
   - Provides status updates during scraping and processing

5. **Question Answering**:
   - Uses a retrieval-augmented generation (RAG) approach
   - Retrieves the most relevant documents for each question
   - Passes retrieved context to the LLM for accurate responses

## File Structure

The main folder contains the following key files:

- **streamlit_app.py**: The main Streamlit application that provides the user interface, manages the scraping process, and coordinates the chat functionality.

- **app.py**: Core application logic including vector store creation, document processing, and LLM chain setup. Handles the bridge between scraping and chat functionality.

- **chat_manager.py**: Manages the chat state, conversation chain, and interaction flow. Handles resetting chats, formatting responses, and maintaining chat history.

- **scraper.py**: Contains all the web scraping functionality, including multi-stage crawling logic, URL extraction, content cleaning, and markdown conversion.

- **example.env**: Template for environment variables, primarily for storing the OpenAI API key.

## Installation

Follow these steps to install and run the application locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/documentation-chat-bot.git
   cd documentation-chat-bot
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to the main directory**:
   ```bash
   cd main
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

**Note**: You don't need to edit the example.env file. You can enter your OpenAI API key directly in the Streamlit interface.

## Usage Guide

1. **Start the application**:
   ```bash
   cd main
   streamlit run streamlit_app.py
   ```

2. **Enter your OpenAI API key** in the sidebar.

3. **Input a documentation URL** in the main interface (e.g., "https://python.langchain.com/docs/").

4. **Click "Start Scraping"** and wait for the process to complete. The application will display progress updates.

5. **Once scraping is complete**, you can start asking questions about the documentation.

6. **Add additional documentation sources** via the sidebar if needed.

7. **Reset chat or reset everything** using the buttons in the sidebar if you want to start fresh.

## Advanced Features

### Model Selection
Choose different OpenAI models from the dropdown in the sidebar to balance between performance and cost:
- GPT-4o: Most capable model
- GPT-4o Mini: Good balance of performance and speed
- GPT-4.5: Latest GPT-4 generation 
- o1/o3: Specialized models for certain tasks

### Scraping Settings
Adjust the maximum number of pages to scrape using the slider in the sidebar. Higher values will scrape more content but take longer.

### Adding Additional Documents
After the initial scraping is complete, you can add content from additional URLs to enhance your chatbot's knowledge.

### Chat Management
- **Reset Chat Only**: Clears the chat history while preserving the document knowledge base
- **Reset Everything**: Complete reset to start with a new documentation source

## Troubleshooting

- **Scraping fails**: Try a different URL or reduce the maximum page count
- **No content found**: Ensure the URL points to a documentation site with accessible content
- **Slow performance**: Reduce the maximum page count or try a simpler documentation site
- **API key errors**: Verify your OpenAI API key is correct and has sufficient credits
- **Import errors**: Ensure you're running the application from the main directory

## Contributors

- Omar Alsadoon
- Mei Kam
- Tamara Freeman
- Milen King
- Yiannis Pagkalos

---

*This project uses OpenAI's models to process and respond to your queries. Your API key is required to use this functionality.*
