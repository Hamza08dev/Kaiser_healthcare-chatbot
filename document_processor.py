"""
Document processor for header-based markdown chunking.
Splits markdown documents by headers while preserving hierarchy.
"""
import re
from typing import List, Dict, Optional


def parse_markdown_file(file_path: str) -> str:
    """
    Read and parse markdown file.
    
    Args:
        file_path: Path to markdown file
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Document file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading document file: {e}")


def chunk_by_headers(markdown_text: str) -> List[Dict]:
    """
    Split markdown by headers, preserving hierarchy.
    Each chunk includes complete section content until next same-level header.
    
    Args:
        markdown_text: Full markdown content
        
    Returns:
        List of chunk dictionaries with:
        - content: Full section text
        - section_path: Hierarchical path (e.g., "7.2 > Strategic Pillar 1")
        - section_number: e.g., "7.2", "8.3"
        - level: Header level (1, 2, 3)
        - line_start, line_end: Line number range
        - header_text: Header title
    """
    lines = markdown_text.split('\n')
    chunks = []
    current_chunk = None
    current_path = []
    current_levels = []  # Track level hierarchy
    
    for line_num, line in enumerate(lines, start=1):
        # Match markdown headers (#, ##, ###)
        header_match = re.match(r'^(#{1,3})\s+(.+)$', line.strip())
        
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            
            # Save previous chunk if exists
            if current_chunk:
                current_chunk['line_end'] = line_num - 1
                current_chunk['content'] = '\n'.join(current_chunk['content_lines']).strip()
                if current_chunk['content']:  # Only add non-empty chunks
                    chunks.append(current_chunk)
            
            # Update hierarchy
            # Remove deeper levels when we encounter a same or higher level header
            while current_levels and current_levels[-1] >= level:
                current_levels.pop()
                if current_path:
                    current_path.pop()
            
            # Extract section number if present
            section_match = re.search(r'^(\d+\.?\d*\.?\d*)\s*[\.\-]?\s*(.+)$', header_text)
            if section_match:
                section_num = section_match.group(1)
                header_title = section_match.group(2).strip()
            else:
                # Try to find section number in header
                section_match = re.search(r'^(\d+\.?\d*\.?\d*)', header_text)
                if section_match:
                    section_num = section_match.group(1)
                    header_title = header_text.replace(section_num, '').strip()
                else:
                    section_num = None
                    header_title = header_text
            
            # Build section path
            current_path.append(header_title)
            section_path = " > ".join(current_path)
            
            # Start new chunk
            current_chunk = {
                'content': [],
                'content_lines': [line],  # Include header in content
                'section_path': section_path,
                'section_number': section_num,
                'level': level,
                'line_start': line_num,
                'line_end': line_num,
                'header_text': header_title
            }
            current_levels.append(level)
            
        else:
            # Add line to current chunk
            if current_chunk:
                current_chunk['content_lines'].append(line)
            else:
                # Content before first header - create a chunk for it
                if line.strip():
                    current_chunk = {
                        'content': [],
                        'content_lines': [line],
                        'section_path': 'Introduction',
                        'section_number': None,
                        'level': 0,
                        'line_start': line_num,
                        'line_end': line_num,
                        'header_text': 'Introduction'
                    }
                    current_levels.append(0)
                    current_path = ['Introduction']
    
    # Save last chunk
    if current_chunk:
        current_chunk['line_end'] = len(lines)
        current_chunk['content'] = '\n'.join(current_chunk['content_lines']).strip()
        if current_chunk['content']:
            chunks.append(current_chunk)
    
    return chunks


def clean_url(url: str) -> str:
    """
    Clean URL by removing trailing punctuation and validating it.
    
    Args:
        url: Raw URL string
        
    Returns:
        Cleaned URL or None if invalid
    """
    if not url:
        return None
    
    url = url.strip()
    
    # Skip anchor links (internal document links)
    if url.startswith('#'):
        return None
    
    # Skip mailto links
    if url.startswith('mailto:'):
        return None
    
    # Only process http/https URLs
    if not url.startswith('http://') and not url.startswith('https://'):
        return None
    
    # Remove trailing punctuation that might have been accidentally included
    # Remove trailing ), ., ;, :, !, ? if they're at the end
    while url and url[-1] in ').,;:!?':
        # But don't remove if it's part of the URL path (check if it's a valid URL character)
        # Simple check: if the character before is not alphanumeric or /, remove it
        if len(url) > 1 and url[-2] not in '/_':
            url = url[:-1]
        else:
            break
    
    return url


def extract_urls_from_markdown(markdown_text: str, chunks: List[Dict]) -> List[Dict]:
    """
    Extract all URLs from markdown and associate them with their parent sections.
    Filters out anchor links and cleans URLs.
    
    Args:
        markdown_text: Full markdown content
        chunks: List of chunks from chunk_by_headers
        
    Returns:
        List of URL dictionaries with:
        - url: The URL (cleaned)
        - parent_section: Section that contains the link
        - link_text: Display text of the link
        - section_number: Section where URL appears
    """
    urls = []
    seen_urls = set()  # Track to avoid duplicates
    
    # Pattern for markdown links: [text](url)
    markdown_link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    
    # Pattern for plain URLs (improved to better handle URLs)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    lines = markdown_text.split('\n')
    
    # Create a mapping of line number to chunk
    line_to_chunk = {}
    for chunk in chunks:
        for line_num in range(chunk['line_start'], chunk['line_end'] + 1):
            line_to_chunk[line_num] = chunk
    
    # Find all markdown links
    for line_num, line in enumerate(lines, start=1):
        # Find markdown links
        for match in re.finditer(markdown_link_pattern, line):
            link_text = match.group(1)
            raw_url = match.group(2)
            
            # Clean and validate URL
            url = clean_url(raw_url)
            if not url:
                continue  # Skip invalid URLs (anchor links, etc.)
            
            # Skip duplicates
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Get parent chunk
            parent_chunk = line_to_chunk.get(line_num, None)
            
            urls.append({
                'url': url,
                'parent_section': parent_chunk['section_path'] if parent_chunk else 'Unknown',
                'link_text': link_text,
                'section_number': parent_chunk['section_number'] if parent_chunk else None,
                'line_number': line_num
            })
        
        # Find plain URLs (not in markdown link format)
        for match in re.finditer(url_pattern, line):
            raw_url = match.group(0)
            
            # Clean and validate URL
            url = clean_url(raw_url)
            if not url:
                continue
            
            # Skip if already captured
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Skip if already captured as markdown link on same line
            if url in [u['url'] for u in urls if u.get('line_number') == line_num]:
                continue
            
            parent_chunk = line_to_chunk.get(line_num, None)
            
            urls.append({
                'url': url,
                'parent_section': parent_chunk['section_path'] if parent_chunk else 'Unknown',
                'link_text': url,  # Use URL as display text
                'section_number': parent_chunk['section_number'] if parent_chunk else None,
                'line_number': line_num
            })
    
    # Also check for URLs in angle brackets <url>
    angle_bracket_pattern = r'<([^>]+)>'
    for line_num, line in enumerate(lines, start=1):
        for match in re.finditer(angle_bracket_pattern, line):
            raw_url = match.group(1)
            
            # Clean and validate URL
            url = clean_url(raw_url)
            if not url:
                continue
            
            # Skip duplicates
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            # Check if not already captured on same line
            if url in [u['url'] for u in urls if u.get('line_number') == line_num]:
                continue
            
            parent_chunk = line_to_chunk.get(line_num, None)
            
            urls.append({
                'url': url,
                'parent_section': parent_chunk['section_path'] if parent_chunk else 'Unknown',
                'link_text': url,
                'section_number': parent_chunk['section_number'] if parent_chunk else None,
                'line_number': line_num
            })
    
    return urls

