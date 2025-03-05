import os
import sys
import asyncio
import requests
from xml.etree import ElementTree

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode
)

# Setup file paths.
__location__ = os.path.dirname(os.path.abspath(__file__))
markdown_output_dir = os.path.join(__location__, "md_output")
os.makedirs(markdown_output_dir, exist_ok=True)
markdown_file_path = os.path.join(markdown_output_dir, "test_content.md")

# List to accumulate content from each URL.
extracted_markdown = []

def get_docs_urls():
    """
    Fetches all URLs from the Crawl4AI documentation sitemap.
    Returns:
        List[str]: List of URLs to crawl.
    """
    sitemap_url = "https://docs.crawl4ai.com/"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def remove_sidebar(markdown_text):
    """
    Removes the navigation sidebar from the markdown output.
    This function assumes that the main content starts with a level-1 header ("# ").
    It splits the text at the first occurrence of "\n# " and returns only the main content.
    """
    parts = markdown_text.split("\n# ", 1)
    if len(parts) == 2:
        # Prepend "# " to the main content and return.
        return "# " + parts[1]
    return markdown_text

async def crawl_all(urls):
    print("\n=== Crawling and Extracting Markdown Content ===")
    
    # Configure browser settings.
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    # Configure run settings.
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )
    
    # Use an asynchronous context manager for proper resource management.
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Crawl all URLs concurrently using arun_many.
        results = await crawler.arun_many(urls=urls, config=run_config)
        
        # Process each result.
        for result in results:
            extracted_markdown.append(f"## {result.url}\n")
            if result.success:
                markdown_content = result.markdown if hasattr(result, "markdown") and result.markdown else "No markdown content extracted."
                # Remove the sidebar from the markdown.
                clean_markdown = remove_sidebar(markdown_content)
                extracted_markdown.append(clean_markdown + "\n")
                print(f"Successfully crawled {result.url}")
            else:
                error_str = f"Error crawling {result.url}: {result.error_message}"
                print(error_str)
                extracted_markdown.append(error_str + "\n")
            extracted_markdown.append("\n---\n")

async def main():
    urls = get_docs_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_all(urls)
    else:
        print("No URLs found to crawl")
    
    # Save all accumulated content to a Markdown file.
    combined_md = "\n".join(extracted_markdown)
    with open(markdown_file_path, "w", encoding="utf-8") as f:
        f.write(combined_md)
    print(f"\nExtracted content saved to {markdown_file_path}")

if __name__ == "__main__":
    asyncio.run(main())