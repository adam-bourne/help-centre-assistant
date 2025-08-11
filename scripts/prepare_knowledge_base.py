from pathlib import Path
import time
import json
import os
import sys

from bs4 import BeautifulSoup
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from pinecone_text.sparse import BM25Encoder
import re
import unicodedata
import html

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TARGET_URLS

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_heading_level(tag_name: str) -> int:
    """
    Returns the integer level of an HTML heading tag (e.g., 'h2' -> 2).
    Returns a high number for non-heading tags.
    """
    if tag_name and tag_name.startswith('h') and len(tag_name) == 2 and tag_name[1].isdigit():
        return int(tag_name[1])
    return float('inf')

def chunk_web_page(urls: List[str]) -> List[str]:
    """
    Scrapes and chunks content from the provided URLs using Selenium and BeautifulSoup.
    Returns a list of text chunks with page titles.
    """
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.headless = True 
    
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
    
    chrome_options.add_argument('--disable-images')
    chrome_options.add_argument('--disable-javascript') 

    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    page_source = None
    driver = None
    results = []
    try:
        print("Initializing Chrome driver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.set_page_load_timeout(30)
        
        for url in urls:
            print(f"Navigating to URL: {url}")
            driver.get(url)
            
            time.sleep(5)
            
            print("Waiting for main content...")
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "article, main, body"))
                )
            except Exception as wait_error:
                print(f"Warning: Timeout waiting for main content: {wait_error}")
            
            print("Getting page source...")
            page_source = driver.page_source
            chunks = process_page_content(page_source, url)
            results += chunks
            print(f"Successfully extracted {len(chunks)} chunks from {url}")

    except Exception as e:
        print(f"Error using Selenium to fetch URL {url}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        return []
    finally:
        if driver:
            print("Closing Chrome driver...")
            driver.quit()

    return results

def process_page_content(page_source: str, url: str) -> List[str]:
    """
    Extract chunks from page source and add page title to each chunk.
    Returns a list of processed text chunks.
    """
    if not page_source:
        return []

    # Extract title from URL - get the last part after the last slash and replace hyphens
    page_title = url.split('/')[-1].replace('-', ' ').title()
    page_title = ' '.join(word for word in page_title.split() if not word[0].isdigit())
    title_prefix = f"Page Title: {page_title}\n\n"

    soup = BeautifulSoup(page_source, 'html.parser')
    main_content = soup.find('article') or soup.find('main') or soup.body

    # Remove irrelevant tags
    for tag in ['nav', 'header', 'footer', 'aside', 'script', 'style']:
        for s in main_content.find_all(tag):
            s.decompose()

    chunks = []
    headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    if not headings:
        # If no headings are found, fall back to chunking by paragraphs
        paragraphs = [p.get_text(strip=True) for p in main_content.find_all('p') if p.get_text(strip=True)]
        return [title_prefix + p for p in paragraphs]

    for heading in headings:
        current_level = get_heading_level(heading.name)
        chunk_content = [heading.get_text(strip=True)]
        
        for sibling in heading.find_next_siblings():
            sibling_level = get_heading_level(sibling.name)
            
            if sibling_level <= current_level:
                break

            text = sibling.get_text(separator=' ', strip=True)
            if text:
                chunk_content.append(text)
        
        if len(chunk_content) > 1:
            # Add the title prefix to each chunk
            chunks.append(title_prefix + '\n\n'.join(chunk_content))
            
    return chunks

def normalize_for_indexing(text: str) -> str:
    """
    Normalizes text for the BM25 model by removing HTML entities,
    normalizing Unicode characters, and replacing special characters.
    """
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[•●▪︎◦]', '-', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_sparse_vectors(chunks: List[str]) -> None:
    """
    Creates and saves BM25 sparse vectors from the provided chunks.
    """
    print("Initializing BM25 encoder...")
    sparse_model = BM25Encoder().default()

    # Normalize the chunks for indexing
    chunks = [normalize_for_indexing(chunk) for chunk in chunks]
    
    print("Fitting BM25 model on chunks...")
    sparse_model.fit(chunks)
    
    # Save the model
    bm25_path = os.path.join(PROJECT_ROOT, 'data', 'bm25_values.json')
    print(f"Saving BM25 values to {bm25_path}...")
    sparse_model.dump(bm25_path)
    print("BM25 values saved successfully!")

def main():

    # Create data directory if it doesn't exist
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Generate chunks
    print("\n=== Step 1: Generating chunks ===")
    chunks = chunk_web_page(TARGET_URLS)
    
    # Save chunks to file
    chunks_path = os.path.join(PROJECT_ROOT, 'data', 'chunks.json')
    print(f"\nSaving chunks to {chunks_path}...")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=4)
    print(f"Successfully saved {len(chunks)} chunks!")

    # Step 2: Create sparse vectors
    print("\n=== Step 2: Creating sparse vectors ===")
    create_sparse_vectors(chunks)

    print("\n=== Knowledge base preparation completed! ===")

if __name__ == "__main__":
    main()