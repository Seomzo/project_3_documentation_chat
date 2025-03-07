# Documentation Chat Bot

A Streamlit-based web application that allows you to scrape websites, process the content, and have conversations about the scraped data using OpenAI's language models.

## Features

- Web scraping of URLs and subpages
- Conversion of web content to markdown format
- Text embedding using Hugging Face's sentence transformers
- Conversational interface powered by OpenAI's GPT models
- Streamlit web interface for easy interaction

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create an `apikey.env` file with your OpenAI API key:
   ```
   open_ai_api_key=your_openai_api_key_here
   ```
4. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

1. Enter a URL you want to scrape in the input field
2. Click "Start Scraping" to begin the scraping process
3. Wait for the scraping and embedding process to complete
4. Once ready, you can start asking questions about the content
5. The chat history will be maintained during your session

## Files

- `streamlit_app.py`: Main Streamlit application
- `scraper.py`: Contains web scraping functionality
- `app.py`: Original command-line version of the app
- `requirements.txt`: Required dependencies

## Notes

- The scraper can handle multiple subpages, but it may take some time for large websites
- Embedding and indexing also takes time based on the amount of content
- The application uses ConversationalRetrievalChain to maintain conversation context