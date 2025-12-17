"""
Hyperlink handler for fetching and processing linked content.
Extracts text from HTML pages and creates child knowledge units.
"""
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional
from pypdf import PdfReader
import io
import os
from datetime import datetime

from config import HYPERLINK_TIMEOUT, MAX_CONTENT_LENGTH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up file handler for hyperlink log
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "hyperlink_processing.log")

file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def fetch_url_content(url: str, timeout: int = HYPERLINK_TIMEOUT) -> Dict:
    """
    Fetch content from URL.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with:
        - url: Original URL
        - status: 'success' or 'error'
        - content_type: 'html', 'pdf', or 'unknown'
        - content: Fetched content (if successful)
        - error: Error message (if failed)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Determine content type
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'pdf' in content_type:
            return {
                'url': url,
                'status': 'success',
                'content_type': 'pdf',
                'content': response.content,
                'error': None
            }
        elif 'html' in content_type or 'text' in content_type:
            return {
                'url': url,
                'status': 'success',
                'content_type': 'html',
                'content': response.text,
                'error': None
            }
        else:
            # Try to parse as HTML anyway
            return {
                'url': url,
                'status': 'success',
                'content_type': 'html',
                'content': response.text,
                'error': None
            }
            
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching URL: {url}")
        return {
            'url': url,
            'status': 'error',
            'content_type': 'unknown',
            'content': None,
            'error': 'Timeout'
        }
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching URL {url}: {e}")
        return {
            'url': url,
            'status': 'error',
            'content_type': 'unknown',
            'content': None,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching URL {url}: {e}")
        return {
            'url': url,
            'status': 'error',
            'content_type': 'unknown',
            'content': None,
            'error': str(e)
        }


def parse_html_content(html_content: str) -> str:
    """
    Extract text from HTML using BeautifulSoup.
    
    Args:
        html_content: HTML content as string
        
    Returns:
        Extracted text content
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Limit content length
        if len(text) > MAX_CONTENT_LENGTH:
            text = text[:MAX_CONTENT_LENGTH] + "... [Content truncated]"
        
        return text
        
    except Exception as e:
        logger.error(f"Error parsing HTML content: {e}")
        return ""


def parse_pdf_content(pdf_content: bytes) -> str:
    """
    Extract text from PDF.
    
    Args:
        pdf_content: PDF file as bytes
        
    Returns:
        Extracted text content
    """
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        full_text = '\n\n'.join(text_parts)
        
        # Limit content length
        if len(full_text) > MAX_CONTENT_LENGTH:
            full_text = full_text[:MAX_CONTENT_LENGTH] + "... [Content truncated]"
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error parsing PDF content: {e}")
        return ""


def create_hyperlink_chunks(urls_with_context: List[Dict]) -> List[Dict]:
    """
    Create child knowledge units for hyperlinks.
    
    Args:
        urls_with_context: List of URL dictionaries from extract_urls_from_markdown
        
    Returns:
        List of chunk dictionaries for hyperlinks:
        - content: Extracted text content
        - content_type: "hyperlink"
        - parent_section: Originating section
        - source_url: Original URL
        - link_text: Display text
        - section_number: Parent section number
    """
    hyperlink_chunks = []
    processed_urls = set()  # Track URLs to avoid duplicates
    success_count = 0
    failure_count = 0
    failed_urls = []
    
    logger.info("=" * 80)
    logger.info(f"Starting hyperlink processing - {len(urls_with_context)} URLs to process")
    logger.info("=" * 80)
    
    for url_info in urls_with_context:
        url = url_info['url']
        
        # Skip duplicates
        if url in processed_urls:
            logger.info(f"Skipping duplicate URL: {url}")
            continue
        
        processed_urls.add(url)
        
        logger.info(f"Processing: {url}")
        logger.info(f"  Link text: {url_info.get('link_text', 'N/A')}")
        logger.info(f"  Parent section: {url_info.get('parent_section', 'N/A')}")
        
        # Fetch content
        fetched = fetch_url_content(url)
        
        if fetched['status'] != 'success':
            error_msg = fetched.get('error', 'Unknown error')
            logger.warning(f"FAILED - URL: {url} | Error: {error_msg}")
            failure_count += 1
            failed_urls.append({
                'url': url,
                'link_text': url_info.get('link_text', 'N/A'),
                'parent_section': url_info.get('parent_section', 'N/A'),
                'error': error_msg
            })
            continue
        
        # Parse content based on type
        if fetched['content_type'] == 'pdf':
            text_content = parse_pdf_content(fetched['content'])
        elif fetched['content_type'] == 'html':
            text_content = parse_html_content(fetched['content'])
        else:
            logger.warning(f"FAILED - Unknown content type for URL: {url}")
            failure_count += 1
            failed_urls.append({
                'url': url,
                'link_text': url_info.get('link_text', 'N/A'),
                'parent_section': url_info.get('parent_section', 'N/A'),
                'error': 'Unknown content type'
            })
            continue
        
        if not text_content or len(text_content.strip()) < 50:
            logger.warning(f"FAILED - Insufficient content extracted from URL: {url} (length: {len(text_content) if text_content else 0})")
            failure_count += 1
            failed_urls.append({
                'url': url,
                'link_text': url_info.get('link_text', 'N/A'),
                'parent_section': url_info.get('parent_section', 'N/A'),
                'error': 'Insufficient content extracted'
            })
            continue
        
        # Create chunk
        chunk = {
            'content': text_content,
            'content_type': 'hyperlink',
            'parent_section': url_info['parent_section'],
            'source_url': url,
            'link_text': url_info['link_text'],
            'section_number': url_info.get('section_number'),
            'section_path': f"Reference: {url_info['link_text']}",
            'level': 0,
            'line_start': url_info.get('line_number', 0),
            'line_end': url_info.get('line_number', 0),
            'header_text': url_info['link_text']
        }
        
        hyperlink_chunks.append(chunk)
        success_count += 1
        logger.info(f"SUCCESS - Processed: {url_info['link_text']} | Content length: {len(text_content)} chars")
    
    # Log summary
    logger.info("=" * 80)
    logger.info(f"Hyperlink processing complete - Success: {success_count}, Failed: {failure_count}, Total: {len(processed_urls)}")
    logger.info("=" * 80)
    
    if failed_urls:
        logger.info("\nFailed URLs Summary:")
        for failed in failed_urls:
            logger.info(f"  URL: {failed['url']}")
            logger.info(f"    Link text: {failed['link_text']}")
            logger.info(f"    Parent section: {failed['parent_section']}")
            logger.info(f"    Error: {failed['error']}")
            logger.info("")
    
    return hyperlink_chunks

