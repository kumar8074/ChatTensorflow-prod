# ===================================================================================
# Project: ChatTensorFlow
# File: src/services/scrapper/tensorflow_scrapper.py
# Description: This file is used to scrape the TensorFlow documentation page.
# Author: LALAN KUMAR
# Created: [08-11-2025]
# Updated: [08-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================
# CAUTION: Please run this file only when there's a update to the documentation page.

import asyncio
import os
import sys
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Set
from urllib.parse import urlparse

import aiohttp
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.logger import logging
from src.config import SCRAPER_BASE_URL, SCRAPER_OUTPUT_DIR, SCRAPER_OUTPUT_FILE

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────

BASE_URL         = SCRAPER_BASE_URL
SITEMAP_URL      = f"{BASE_URL}/sitemap.xml"
OUTPUT_DIR       = SCRAPER_OUTPUT_DIR
OUTPUT_FILE      = SCRAPER_OUTPUT_FILE  # Combined file name

# Updated patterns to ONLY include Python API documentation and guides
INCLUDED_PATTERNS = [
    '/api_docs/python/',  # Python API reference
    '/guide/',            # TensorFlow guides (Python-focused)
    '/tutorials/',        # TensorFlow tutorials (Python-focused)
    '/install/',         # Installation guides
]

# Exclude non-Python specific content
EXCLUDED_PATTERNS = [
    '/blog/', '/versions/', '/ecosystem/', '/resources/',
    '/community/', '/about/', '/responsible_ai',
    '/api_docs/cc', '/api_docs/java', 
    '/js/',        # JavaScript docs
    '/swift/',     # Swift docs
    '/federated/', # Federated learning
    '/graphics/',  # TF Graphics
    '/hub/',       # TF Hub
    '/datasets/',  # TF Datasets
    '/neural_structured_learning/', # NSL
    '/probability/', # TF Probability
    '/quantum/',   # TF Quantum
    '/recommenders/', # TF Recommenders
    '/tensorboard/', # TensorBoard
    '/text/',      # TF Text
    '/addons/',    # TF Addons
    '/io/',        # TF IO
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── SITEMAP FETCHING ───────────────────────────────────────────────────────────

async def fetch_sitemap(url: str) -> List[str]:
    """
    Fetch all valid URLs from the sitemap (and any sub-sitemaps) in a single aiohttp session.
    Uses XML namespace 'http://www.sitemaps.org/schemas/sitemap/0.9' to locate <loc> entries.
    """
    urls: List[str] = []
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    async with aiohttp.ClientSession() as session:
        logging.info(f"Fetching sitemap: {url}")
        async with session.get(url) as resp:
            if resp.status != 200:
                logging.error(f"Sitemap fetch failed [{resp.status}]: {url}")
                return []
            sitemap_xml = await resp.text()

        root = ET.fromstring(sitemap_xml)
        index_entries = root.findall('.//sm:sitemap/sm:loc', ns)

        if index_entries:
            logging.info(f"Found {len(index_entries)} sub-sitemaps")
            for loc in index_entries:
                sub_url = loc.text or ""
                if not sub_url.startswith(BASE_URL):
                    continue
                logging.info(f"  → Sub-sitemap: {sub_url}")
                async with session.get(sub_url) as sub_resp:
                    if sub_resp.status != 200:
                        continue
                    sub_xml = await sub_resp.text()
                sub_root = ET.fromstring(sub_xml)
                for url_loc in sub_root.findall('.//sm:url/sm:loc', ns):
                    link = url_loc.text or ""
                    if is_python_doc(link):
                        urls.append(link)
        else:
            # Single sitemap case
            for url_loc in root.findall('.//sm:url/sm:loc', ns):
                link = url_loc.text or ""
                if is_python_doc(link):
                    urls.append(link)

    logging.info(f"Total Python documentation URLs to crawl: {len(urls)}")
    return urls

# ─── URL FILTER ─────────────────────────────────────────────────────────────────

def is_python_doc(url: str) -> bool:
    """
    Check if URL is a Python documentation page.
    Returns True if URL matches included patterns and doesn't match excluded patterns.
    """
    if not url.startswith(BASE_URL):
        return False
    
    # First check if it matches any excluded patterns
    if any(pattern in url for pattern in EXCLUDED_PATTERNS):
        return False
    
    # Then check if it matches any included patterns
    if any(pattern in url for pattern in INCLUDED_PATTERNS):
        return True
    
    return False

def should_process_url(url: str, seen: Set[str]) -> bool:
    """
    Return True if URL is a Python doc page, on-domain, and not already processed.
    Ensures we only crawl TensorFlow Python documentation pages.
    """
    if url in seen:
        return False
    parsed = urlparse(url)
    if not parsed.netloc.endswith("tensorflow.org"):
        return False
    if not is_python_doc(url):
        return False
    return True

# ─── MAIN CRAWLER ───────────────────────────────────────────────────────────────

async def crawl_tensorflow_docs():
    # 1. Load URLs from sitemap
    urls = await fetch_sitemap(SITEMAP_URL)

    # 2. Configure Crawl4AI
    browser_cfg = BrowserConfig(headless=True)  # Chromium in headless mode
    run_cfg     = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector=".devsite-article-body",    # Extract main article container
        word_count_threshold=50,
        screenshot=False
    )

    seen: Set[str]      = set()
    all_docs: List[dict] = []

    # 3. Crawl each page under a single AsyncWebCrawler context
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            if not should_process_url(url, seen):
                continue
            seen.add(url)

            #logging.info(f"Crawling: {url}")
            try:
                result = await crawler.arun(url=url, config=run_cfg)
            except Exception as e:
                logging.error(f"Error fetching {url}: {e}")
                continue

            if not result.success:
                logging.warning(f"Failed: {result.error_message}")
                continue

            # 4. Extract title from metadata (if present)
            metadata = result.metadata or {}
            title = metadata.get("title", "").strip()

            # 5. Fallback to first line of raw_markdown if no metadata title
            if not title and result.markdown:
                raw_md = (result.markdown.raw_markdown
                          if hasattr(result.markdown, "raw_markdown")
                          else result.markdown)
                title = raw_md.split("\n", 1)[0].strip()

            # 6. Determine content markdown
            content_md = (result.markdown.raw_markdown
                          if hasattr(result.markdown, "raw_markdown")
                          else (result.markdown or ""))

            # 7. Record crawl time explicitly
            crawled_at = datetime.utcnow().isoformat() + "Z"

            doc = {
                "url":        url,
                "title":      title,
                "content":    content_md,
                "crawled_at": crawled_at,
            }

            all_docs.append(doc)
            #logging.info(f"Processed: {url}")

    # 8. Write combined RAG JSON to temp directory
    combined_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2)
    logging.info(f"Combined RAG file saved: {combined_path}")
    logging.info(f"Total Python documentation pages scraped: {len(all_docs)}")

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(crawl_tensorflow_docs())