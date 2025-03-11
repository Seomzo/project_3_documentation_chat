import os, json, asyncio, re
from urllib.parse import urlparse, urlunparse, urljoin
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)
from typing import List, Set, Dict, Any, Optional
import time
from bs4 import BeautifulSoup

def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False,
    )

def normalize_url_path(url_path: str) -> str:
    """
    Normalize URL paths by removing redundant segments.
    For example:
    /docs/introduction/docs/concept/ -> /docs/concept/
    """
    if not url_path:
        return ""
    
    # Split path into segments
    segments = [seg for seg in url_path.split('/') if seg]
    
    # Find and remove redundant patterns
    cleaned_segments = []
    i = 0
    while i < len(segments):
        # Check if this segment starts a redundant pattern
        duplicate_found = False
        for j in range(1, len(segments) - i):
            # Look for patterns like ['docs', 'introduction', 'docs', ...]
            if i + j < len(segments) and segments[i] == segments[i + j]:
                # Found a potential repetition starting point
                pattern_length = 1
                while (i + pattern_length < len(segments) and 
                       i + j + pattern_length < len(segments) and 
                       segments[i + pattern_length] == segments[i + j + pattern_length]):
                    pattern_length += 1
                
                if pattern_length > 0:
                    # We found a repetition, keep only the first occurrence
                    duplicate_found = True
                    for k in range(pattern_length):
                        if i + k < len(segments):
                            cleaned_segments.append(segments[i + k])
                    # Skip past the duplicated pattern
                    i += j + pattern_length
                    break
        
        if not duplicate_found:
            if i < len(segments):
                cleaned_segments.append(segments[i])
            i += 1
    
    # Rebuild the path
    return '/' + '/'.join(cleaned_segments)

def clean_extracted_url(raw_url: str, base_url: str) -> str:
    """
    Clean and normalize URLs, handling relative paths, removing unwanted characters,
    and fixing redundant path segments.
    """
    try:
        if not raw_url:
            return ""
        
        # Parse the base URL to get its components
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
            
        # Handle relative URLs (both /path and path formats)
        if raw_url.startswith('/'):
            # Absolute path relative to domain
            raw_url = f"{parsed_base.scheme}://{parsed_base.netloc}{raw_url}"
        elif not raw_url.startswith(('http://', 'https://')):
            # Relative to current path
            raw_url = urljoin(base_url, raw_url)
            
        # Remove unwanted characters
        cleaned = raw_url.replace("/</", "/").replace("<", "").replace(">", "").replace("./", "")
        
        # Parse the cleaned URL
        parsed_url = urlparse(cleaned)
        
        # Normalize the path to remove redundant segments
        normalized_path = normalize_url_path(parsed_url.path)
        
        # Check for and remove embedded URLs in the path
        path = normalized_path
        
        # Look for patterns that indicate external URLs embedded in the path
        domains_in_path = re.findall(r'/(https?:?/?/?|www\.)([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', path)
        
        if domains_in_path:
            # This is likely an embedded URL - could be invalid
            # Extract just the core path up to where the embedded URL starts
            first_domain_pos = path.find(domains_in_path[0][0])
            if first_domain_pos > 0:
                path = path[:first_domain_pos]
            else:
                # If the embedded URL is at the start, use just the base path
                path = "/"
                
        # Rebuild the clean URL
        clean_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment
        ))
        
        return clean_url
    except Exception as e:
        print(f"Error cleaning URL {raw_url}: {str(e)}")
        return ""

def extract_links_from_html(html_content: str, base_url: str) -> List[str]:
    """
    Extract links from HTML content, focusing on anchor tags and handling 
    both relative and absolute URLs.
    """
    links = []
    try:
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract links from anchor tags
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip empty hrefs, javascript, and anchors
            if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                continue
                
            # Clean and add the URL
            clean_url = clean_extracted_url(href, base_url)
            if clean_url:
                links.append(clean_url)
                
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
                
        return unique_links
    except Exception as e:
        print(f"Error extracting links from HTML: {str(e)}")
        return links

def extract_href_attributes(html_content: str, base_url: str) -> List[str]:
    """
    Direct extraction of href attributes from HTML using regex,
    as a fallback when BeautifulSoup doesn't find links.
    """
    urls = []
    try:
        # Extract all href attributes using regex
        href_pattern = re.compile(r'href=[\'"]([^\'"]+)[\'"]')
        matches = href_pattern.findall(html_content)
        
        parsed_base = urlparse(base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
        
        for href in matches:
            # Skip anchors, javascript, etc.
            if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
                continue
                
            # Build full URL if it's relative
            if href.startswith('/'):
                full_url = f"{base_domain}{href}"
            elif not href.startswith(('http://', 'https://')):
                full_url = urljoin(base_url, href)
            else:
                full_url = href
                
            # Clean the URL
            clean_url = clean_extracted_url(full_url, base_url)
            if clean_url:
                urls.append(clean_url)
                
        # Remove duplicates
        return list(set(urls))
    except Exception as e:
        print(f"Error extracting hrefs with regex: {str(e)}")
        return urls

def should_skip_url(url: str, visited_urls: Set[str], base_url: str) -> bool:
    """
    Enhanced URL filtering with comprehensive checks for external domains,
    common non-content URLs, and already visited pages.
    """
    try:
        if not url:
            return True
            
        # Skip if already visited
        if url in visited_urls:
            return True
            
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_url)
        
        # Skip if not same domain (external link)
        if parsed_url.netloc != parsed_base.netloc:
            return True
            
        # Skip common file types and paths that aren't content
        skip_patterns = [
            '#',                                  # Anchors
            'mailto:', 'tel:', 'javascript:',     # Protocol handlers
            '/static/', '/assets/', '/images/', '/css/', '/js/',  # Static assets
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js', '.ico', '.woff', '.ttf',  # File extensions
            '/api/', '/rss/', '/feed/', '/sitemap', '/raw/', '/download/',  # Special paths
            'login', 'signin', 'signup', 'register', 'logout', 'auth',  # Auth pages
            'twitter.com', 'github.com', 'facebook.com', 'linkedin.com',  # Social media
        ]
        
        # Look for embedded domains in the path which indicate external links
        path_lower = parsed_url.path.lower()
        if any(external_domain in path_lower for external_domain in ['github.com', 'twitter.com', 'facebook.com', 
                                                                     'youtube.com', 'linkedin.com']):
            return True
            
        # Check URL against skip patterns
        lower_url = url.lower()
        return any(pattern in lower_url for pattern in skip_patterns)
    except Exception as e:
        print(f"Error checking URL {url}: {str(e)}")
        return True

def remove_sidebar(markdown_text: str) -> str:
    """
    Clean up markdown content by removing navigation elements and standardizing format.
    """
    try:
        # Remove sidebar/navigation
        parts = markdown_text.split("\n# ", 1)
        if len(parts) == 2:
            markdown_text = "# " + parts[1]
        
        # Additional cleanup
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)  # Remove excess newlines
        markdown_text = re.sub(r'(?m)^\s+$', '', markdown_text)   # Remove blank lines with spaces
        
        return markdown_text.strip()
    except Exception as e:
        print(f"Error cleaning markdown: {str(e)}")
        return markdown_text

async def process_batch(crawler, urls, visited_urls):
    """
    Process a batch of URLs and return valid markdown content
    """
    results = []
    try:
        batch_results = await crawler.arun_many(urls=urls)
        for res in batch_results:
            if res and res.markdown:
                cleaned_md = remove_sidebar(res.markdown)
                # Skip 404 or empty pages
                if not cleaned_md.strip().lower().startswith(("# page not found", "# 404")):
                    results.append({
                        "url": res.url,
                        "markdown": cleaned_md,
                        "html": res.html  # Keep the HTML for potential link extraction
                    })
                    visited_urls.add(res.url)
                    print(f"Successfully crawled: {res.url}")
                else:
                    print(f"Skipped page (404/Not Found): {res.url}")
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
    
    return results

async def extract_urls_from_sitemap(crawler, base_url: str) -> List[str]:
    """
    Extract URLs from sitemap.xml by stripping the base URL and appending sitemap.xml.
    
    Enhanced to:
    1. First try the standard sitemap.xml location
    2. Handle nested sitemaps (sitemap index files)
    3. Preserve full paths from original sitemap instead of modifying them
    4. Handle non-standard sitemap formats (like TensorFlow's plain text URLs)
    """
    try:
        # Parse the base URL to get the root domain
        parsed_url = urlparse(base_url)
        root_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = f"{root_url}/sitemap.xml"
        
        print(f"Attempting to fetch sitemap from: {sitemap_url}")
        
        # Fetch the sitemap
        result = await crawler.arun(url=sitemap_url)
        if result and result.html:
            all_urls = set()
            
            # Check if this is a sitemap index (contains multiple sitemaps)
            is_sitemap_index = '<sitemapindex' in result.html.lower()
            
            if is_sitemap_index:
                print("Found a sitemap index file, processing nested sitemaps...")
                # Extract nested sitemap URLs
                sitemap_urls = re.findall(r'<loc>(.*?\.xml)</loc>', result.html)
                
                # Process each nested sitemap
                for nested_sitemap_url in sitemap_urls:
                    print(f"Processing nested sitemap: {nested_sitemap_url}")
                    nested_result = await crawler.arun(url=nested_sitemap_url)
                    if nested_result and nested_result.html:
                        # First try standard XML format with <loc> tags
                        loc_matches = re.findall(r'<loc>(.*?)</loc>', nested_result.html)
                        if loc_matches:
                            all_urls.update(loc_matches)
                        else:
                            # Try non-standard format (plain text URLs like TensorFlow's sitemap)
                            # This pattern looks for URLs followed by dates (YYYY-MM-DD format)
                            plain_urls = re.findall(r'(https?://[^\s]+)\s+\d{4}-\d{2}-\d{2}', nested_result.html)
                            if plain_urls:
                                print(f"Found {len(plain_urls)} URLs in plain text format")
                                # Clean URLs by removing trailing T and everything after
                                clean_plain_urls = [url.split()[0] if ' ' in url else url for url in plain_urls]
                                all_urls.update(clean_plain_urls)
            else:
                # Regular sitemap - extract URLs from <loc> tags
                loc_matches = re.findall(r'<loc>(.*?)</loc>', result.html)
                if loc_matches:
                    all_urls.update(loc_matches)
                else:
                    # Try non-standard format (plain text URLs)
                    plain_urls = re.findall(r'(https?://[^\s]+)\s+\d{4}-\d{2}-\d{2}', result.html)
                    if plain_urls:
                        print(f"Found {len(plain_urls)} URLs in plain text format")
                        # Clean URLs by removing trailing T and everything after
                        clean_plain_urls = [url.split()[0] if ' ' in url else url for url in plain_urls]
                        all_urls.update(clean_plain_urls)
                
                # Also look for direct URLs in case the sitemap is using a different format
                if not all_urls:
                    url_matches = re.findall(r'https?://[^\s<>"\']+', result.html)
                    all_urls.update(url_matches)
            
            # Clean and filter URLs
            cleaned_urls = []
            for url in all_urls:
                # For sitemap URLs, don't modify the path structure
                if parsed_url.netloc in url:  # Only keep URLs from same domain
                    cleaned_urls.append(url)
            
            print(f"Found {len(cleaned_urls)} URLs from sitemap")
            return cleaned_urls
            
    except Exception as e:
        print(f"Error fetching sitemap: {str(e)}")
    return []

async def direct_crawl_for_href(crawler, url: str) -> List[str]:
    """
    Directly access the DOM via JavaScript to extract href attributes.
    This helps capture links rendered by JavaScript frameworks like React.
    """
    try:
        # Create a run config with JavaScript execution enabled
        js_run_config = CrawlerRunConfig(
            # Enable JavaScript execution
            execute_js=True,
            # Add a custom script to extract all links
            custom_js="""
            () => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                return links.map(link => link.href);
            }
            """
        )
        
        # Run the crawler with JS execution
        result = await crawler.arun(
            url=url,
            config=js_run_config
        )
        
        if result:
            # First try to get links from result.custom_js_result
            if hasattr(result, 'custom_js_result') and result.custom_js_result:
                js_result = result.custom_js_result
                links = []
                
                # Process the JavaScript result
                if isinstance(js_result, list):
                    links = js_result
                elif isinstance(js_result, str):
                    try:
                        # Try to parse as JSON if it's a string
                        parsed = json.loads(js_result)
                        if isinstance(parsed, list):
                            links = parsed
                    except:
                        # If JSON parsing fails, try to extract URLs using regex
                        links = re.findall(r'https?://[^\s"\']+', js_result)
                
                # Clean and return the URLs
                valid_links = []
                for link in links:
                    if link and isinstance(link, str) and not link.startswith('javascript:'):
                        cleaned = clean_extracted_url(link, url)
                        if cleaned:
                            valid_links.append(cleaned)
                
                print(f"Extracted {len(valid_links)} links from JavaScript DOM")
                return valid_links
            
            # Fallback: Extract links from the rendered HTML
            if result.html:
                links = extract_links_from_html(result.html, url)
                print(f"Extracted {len(links)} links from HTML after JavaScript rendering")
                return links
        
    except Exception as e:
        print(f"Error in direct DOM crawling: {str(e)}")
    
    print("Direct DOM crawling returned no results")
    return []

def generate_common_doc_paths(base_url: str) -> List[str]:
    """
    Generate common documentation paths as a fallback when no links are found.
    """
    parsed_url = urlparse(base_url)
    base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Common paths in documentation sites
    common_paths = [
        "/docs", "/documentation", "/guide", "/guides", "/tutorial", "/tutorials",
        "/api", "/api-reference", "/reference", "/examples", "/getting-started",
        "/overview", "/introduction", "/concepts", "/faq", "/help",
        "/quickstart", "/installation", "/usage", "/advanced"
    ]
    
    # Generate URLs with the common paths
    urls = [f"{base_domain}{path}" for path in common_paths]
    return urls

async def get_all_cleaned_markdown(inputurl=None, max_pages: int = 250, 
                                  status_callback=None, stop_callback=None) -> List[str]:
    """
    Enhanced web scraping with sitemap-first approach and three stages of crawling.
    
    Args:
        inputurl: The URL to crawl
        max_pages: Maximum number of pages to scrape
        status_callback: Optional callback function to report status
        stop_callback: Optional callback function to check if scraping should stop
    
    Returns:
        A list of cleaned markdown content from scraped pages
    """
    if inputurl is None:
        inputurl = input("Enter the URL to crawl: ")
    
    # Try to install Playwright browsers if not already installed
    try:
        import sys
        import subprocess
        
        # First check if we're in Streamlit Cloud environment
        if os.environ.get('STREAMLIT_SHARING', '') or os.path.exists('/home/appuser'):
            if status_callback:
                status_callback("Installing Playwright browsers for Streamlit Cloud...")
                
            # Use sys.executable to ensure we're using the correct Python interpreter
            install_cmd = [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"]
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = f"Failed to install Playwright browsers: {result.stderr}"
                if status_callback:
                    status_callback(error_msg)
                print(error_msg)
            else:
                if status_callback:
                    status_callback("Playwright browsers installed successfully")
        else:
            # For local environment, use the simpler command
            subprocess.run(['playwright', 'install', 'chromium'], check=True)
    except Exception as e:
        print(f"Warning: Failed to automatically install Playwright browsers: {str(e)}")
        if status_callback:
            status_callback(f"Warning: Failed to automatically install Playwright browsers: {str(e)}")
    
    base_url = inputurl
    browser_conf = get_browser_config()
    visited_urls = set()
    all_cleaned_markdown = []
    stage2_urls = []
    stage3_urls = []
    
    # Check if scraping should stop
    if stop_callback and stop_callback():
        print("Scraping stopped by user.")
        return all_cleaned_markdown
        
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # Stage 1: Try sitemap first, then fall back to base page crawl
        print(f"\nStarting crawl for: {base_url}")
        if status_callback:
            status_callback(f"Stage 1: Sitemap crawl for: {base_url}")
        
        # First attempt: Try to get URLs from sitemap
        sitemap_urls = await extract_urls_from_sitemap(crawler, base_url)
        
        if sitemap_urls:
            print(f"Successfully found {len(sitemap_urls)} URLs from sitemap")
            if status_callback:
                status_callback(f"Found {len(sitemap_urls)} URLs from sitemap")
            
            # Use these URLs for stage 2
            for url in sitemap_urls:
                cleaned_url = url  # We're keeping the original URLs from sitemap
                if cleaned_url and not should_skip_url(cleaned_url, visited_urls, base_url):
                    stage2_urls.append(cleaned_url)
                    visited_urls.add(cleaned_url)
        else:
            # Fallback: Crawl the base page if sitemap approach failed
            print("No sitemap found or sitemap empty. Falling back to standard crawling.")
            if status_callback:
                status_callback("No sitemap found. Using standard crawling approach.")
                
            try:
                # Crawl the base URL
                result = await crawler.arun(url=base_url)
                if result:
                    visited_urls.add(base_url)
                    
                    # Add the base page content to our collection
                    if result.markdown:
                        base_markdown = result.markdown
                        all_cleaned_markdown.append(remove_sidebar(base_markdown))
                    
                    # Try extraction methods in sequence until we get URLs
                    
                    # 1. Try markdown extraction
                    markdown_urls = set(re.findall(r'\((https?://[^\)]+)\)', result.markdown or ""))
                    
                    # 2. Try BeautifulSoup HTML parsing
                    html_urls = set()
                    if result.html:
                        html_urls = set(extract_links_from_html(result.html, base_url))
                    
                    # 3. Try direct DOM access with JavaScript
                    js_urls = set()
                    js_urls = set(await direct_crawl_for_href(crawler, base_url))
                    
                    # 4. Try regex as fallback
                    regex_urls = set()
                    if result.html and not (markdown_urls or html_urls or js_urls):
                        regex_urls = set(extract_href_attributes(result.html, base_url))
                    
                    # 5. Try common documentation paths as last resort
                    common_urls = set()
                    if not (markdown_urls or html_urls or js_urls or regex_urls):
                        common_urls = set(generate_common_doc_paths(base_url))
                    
                    # Combine all found URLs
                    all_urls = markdown_urls | html_urls | js_urls | regex_urls | common_urls
                    
                    # Clean and filter URLs
                    for url in all_urls:
                        cleaned_url = clean_extracted_url(url, base_url)
                        if cleaned_url and not should_skip_url(cleaned_url, visited_urls, base_url):
                            stage2_urls.append(cleaned_url)
                            visited_urls.add(cleaned_url)
            except Exception as e:
                print(f"Error in base page crawling: {str(e)}")
                if status_callback:
                    status_callback(f"Error in base page crawling: {str(e)}")
                
                # If standard crawling failed, try common documentation paths
                print("Base page crawling failed. Trying common documentation paths...")
                common_urls = generate_common_doc_paths(base_url)
                stage2_urls.extend(common_urls)
        
        # Deduplicate URLs
        stage2_urls = list(set(stage2_urls))
        
        # Limit number of pages for stage 2
        stage2_limit = min(len(stage2_urls), max_pages // 2)
        stage2_urls = stage2_urls[:stage2_limit]
        print(f"Found {len(stage2_urls)} URLs to crawl in Stage 2")
        if status_callback:
            status_callback(f"Found {len(stage2_urls)} URLs to crawl in Stage 2")
        
        # Stage 2: Continue with the crawling as before
        if stage2_urls:
            # Process in batches to avoid overwhelming the server
            batch_size = 5
            num_batches = (len(stage2_urls) + batch_size - 1) // batch_size
            
            for i in range(0, len(stage2_urls), batch_size):
                # Check if scraping should stop
                if stop_callback and stop_callback():
                    print("Scraping stopped by user during Stage 2.")
                    return all_cleaned_markdown
                    
                batch = stage2_urls[i:i+batch_size]
                batch_num = i // batch_size + 1
                print(f"Crawling Stage 2 batch {batch_num}/{num_batches}")
                if status_callback:
                    status_callback(f"Stage 2: Crawling batch {batch_num}/{num_batches}")
                
                # Process batch
                batch_results = await process_batch(crawler, batch, visited_urls)
                
                # Extract sublinks from stage 2 pages for stage 3
                for result in batch_results:
                    all_cleaned_markdown.append(result["markdown"])
                    
                    # Extract links from both markdown and HTML content
                    markdown_sublinks = re.findall(r'\((https?://[^\)]+)\)', result["markdown"])
                    html_sublinks = extract_links_from_html(result["html"], base_url) if result["html"] else []
                    
                    # Process all found links
                    for sublink in set(markdown_sublinks + html_sublinks):
                        cleaned_sublink = clean_extracted_url(sublink, base_url)
                        if cleaned_sublink and not should_skip_url(cleaned_sublink, visited_urls, base_url):
                            stage3_urls.append(cleaned_sublink)
                            visited_urls.add(cleaned_sublink)
                
                # Short delay between batches
                await asyncio.sleep(0)

        # Deduplicate and limit stage 3 URLs
        stage3_urls = list(set(stage3_urls))
        stage3_limit = min(len(stage3_urls), max_pages - len(all_cleaned_markdown))
        stage3_urls = stage3_urls[:stage3_limit]
        
        # Stage 3: Crawl sublinks found in stage 2 pages
        if stage3_urls:
            num_batches = (len(stage3_urls) + batch_size - 1) // batch_size
            print(f"\nFound {len(stage3_urls)} URLs to crawl in Stage 3")
            if status_callback:
                status_callback(f"Found {len(stage3_urls)} URLs to crawl in Stage 3")
            
            batch_size = 5
            # Process in batches
            for i in range(0, len(stage3_urls), batch_size):
                # Check if scraping should stop
                if stop_callback and stop_callback():
                    print("Scraping stopped by user during Stage 3.")
                    return all_cleaned_markdown
                    
                batch = stage3_urls[i:i+batch_size]
                batch_num = i // batch_size + 1
                print(f"Crawling Stage 3 batch {batch_num}/{num_batches}")
                if status_callback:
                    status_callback(f"Stage 3: Crawling batch {batch_num}/{num_batches}")
                
                # Process batch
                batch_results = await process_batch(crawler, batch, visited_urls)
                for result in batch_results:
                    all_cleaned_markdown.append(result["markdown"])
                
                # Short delay between batches
                await asyncio.sleep(0)
    
    print(f"\nSuccessfully collected {len(all_cleaned_markdown)} pages")
    if status_callback:
        status_callback(f"Successfully collected {len(all_cleaned_markdown)} pages")
    return all_cleaned_markdown

if __name__ == "__main__":
    cleaned_pages = asyncio.run(get_all_cleaned_markdown())
    print(f'Collected {len(cleaned_pages)} markdown pages.')