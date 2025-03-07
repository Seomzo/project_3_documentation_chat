import os, json, asyncio, re
from urllib.parse import urlparse, urlunparse
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)



def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False,
    )

def clean_extracted_url(raw_url: str, base_url: str) -> str:
    """
    Clean out unwanted characters and, if the URL is on the same domain as the base,
    remove the duplicate base path.
    """
    cleaned = raw_url.replace("/</", "/").replace("<", "").replace(">", "").replace("./", "")
    parsed_base = urlparse(base_url)
    parsed_raw = urlparse(cleaned)
    if parsed_base.netloc == parsed_raw.netloc:
        base_path = parsed_base.path
        raw_path = parsed_raw.path
        if raw_path.startswith(base_path):
            new_path = raw_path[len(base_path):]
            if not new_path.startswith("/"):
                new_path = "/" + new_path
            cleaned = urlunparse((parsed_base.scheme, parsed_base.netloc, new_path,
                                  parsed_raw.params, parsed_raw.query, parsed_raw.fragment))
    return cleaned

def remove_duplicate_common_prefix(url: str, base_url: str) -> str:
    parsed_base = urlparse(base_url)
    parsed_url = urlparse(url)
    base_segs = [seg for seg in parsed_base.path.split("/") if seg]
    url_segs = [seg for seg in parsed_url.path.split("/") if seg]
    # Compute common prefix between base_segs and url_segs.
    common_prefix = []
    for bs, us in zip(base_segs, url_segs):
        if bs == us:
            common_prefix.append(bs)
        else:
            break
    n = len(common_prefix)
    # If the URL's path starts with the common prefix twice, remove one copy.
    if n > 0 and len(url_segs) >= 2*n and url_segs[:n] == url_segs[n:2*n]:
        new_segs = url_segs[:n] + url_segs[2*n:]
        new_path = "/" + "/".join(new_segs)
        return urlunparse((parsed_url.scheme, parsed_url.netloc, new_path,
                           parsed_url.params, parsed_url.query, parsed_url.fragment))
    return url

def should_skip_url(url: str) -> bool:
    """
    Skip URLs that have multiple occurrences of 'https:' (duplicate scheme in the path)
    or that end with common image extensions.
    """
    if url.count("https:") > 1:
        return True
    if '#' in url:
        return True
    if url.startswith("https://img"):
        return True
    for ext in [".png", ".jpg", ".jpeg", ".svg", ".gif"]:
        if url.lower().endswith(ext):
            return True
    return False

def belongs_to_base(url: str, base_url: str) -> bool:
    """
    Returns True if the given URL has the same scheme and netloc as the base_url.
    """
    parsed_base = urlparse(base_url)
    parsed_url = urlparse(url)
    return parsed_url.scheme == parsed_base.scheme and parsed_url.netloc == parsed_base.netloc

def remove_sidebar(markdown_text: str) -> str:
    """
    Removes the sidebar and navigation elements from the markdown output.
    Assumes the main content starts with a level-1 header ("# ").
    Splits the text at the first occurrence of "\n# " and returns the main content.
    """
    parts = markdown_text.split("\n# ", 1)
    if len(parts) == 2:
        return "# " + parts[1]
    return markdown_text

async def get_all_cleaned_markdown(inputurl=None):
    # Input URL.
    if inputurl is None:
        inputurl = input("Enter the URL to crawl: ")
    base_url = inputurl
    browser_conf = get_browser_config()
    
    # Local accumulators.
    urls_extracted = []
    all_cleaned_markdown = []  # Valid markdown from stage 2 and stage 3.
    sublinks_extracted = []    # Sublinks extracted from stage 2 pages.
    
    # Stage 1: Crawl the base page and extract URLs.
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(url=base_url)
        markdown = result.markdown
        raw_urls = re.findall(r'\((https?://[^\)]+)\)', markdown)
        print(markdown)

        for url in raw_urls:
            if url.startswith("https://"):
                normalized_url = clean_extracted_url(url, base_url)
                normalized_url = remove_duplicate_common_prefix(normalized_url, base_url)

                if should_skip_url(normalized_url):
                    continue
                urls_extracted.append(normalized_url)
        urls_extracted = list(dict.fromkeys(urls_extracted))
        print("Extracted URLs from base page:")
        for url in urls_extracted:
            print(" -", url)
    
    # Stage 2: Crawl the extracted URLs concurrently.
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        results = await crawler.arun_many(urls=urls_extracted,verbose=True)
        print("\nStage 2 Results:")
        for res in results:
            if not res.markdown:
                print("No markdown content available for:", res.url, "\n")
                continue
            cleaned_markdown = remove_sidebar(res.markdown)
            if cleaned_markdown.strip().lower().startswith("# page not found") or \
                cleaned_markdown.strip().lower().startswith("# 404"):
                print("Skipped page (404/Not Found):", res.url, "\n")
                continue
            # Append valid markdown.
            all_cleaned_markdown.append(cleaned_markdown)
            print("Crawled URL:", res.url)
            print("Markdown snippet:", cleaned_markdown[:100], "\n")
            
            # Extract sublinks from the valid page.
            page_sublinks = re.findall(r'\((https?://[^\)]+)\)', cleaned_markdown)
            for sublink in page_sublinks:
                if sublink.startswith("https://"):
                    normalized_sublink = clean_extracted_url(sublink, base_url)
                    if not belongs_to_base(normalized_sublink, base_url):
                        continue
                    if should_skip_url(normalized_sublink):
                        continue
                    sublinks_extracted.append(normalized_sublink)
    
    # Deduplicate sublinks.
    sublinks_extracted = list(set(sublinks_extracted))
    print("\nExtracted Sublinks from Stage 2:")
    # for link in sublinks_extracted:
    #     print(" -", link)
    print(f"\nTotal valid markdown pages collected: {len(sublinks_extracted)}")
    
    # Stage 3: Crawl the sublinks concurrently.
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        sub_results = await crawler.arun_many(urls=sublinks_extracted)
        print("\nStage 3 Results:")
        for res in sub_results:
            if not res.markdown:
                print("No markdown content available for:", res.url, "\n")
                continue
            cleaned_markdown = remove_sidebar(res.markdown)
            if cleaned_markdown.strip().lower().startswith("# page not found") or \
                cleaned_markdown.strip().lower().startswith("# 404"):
                print("Skipped page (404/Not Found):", res.url, "\n")
                continue
            # Append valid subpage markdown.
            all_cleaned_markdown.append(cleaned_markdown)
            print("Crawled Sublink URL:", res.url)
            print("Markdown snippet:", cleaned_markdown[:1000], "\n")
    
    print(f"\nTotal valid markdown pages collected: {len(all_cleaned_markdown)}")
    # Proceed to your embedding step using all_cleaned_markdown.
    return all_cleaned_markdown
if __name__ == "__main__":
    cleaned_pages = asyncio.run(get_all_cleaned_markdown())
    print(f'Collected {len(cleaned_pages)} markdown pages.')
