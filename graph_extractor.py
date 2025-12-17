"""
Strategy graph extractor.
Extracts strategic pillars, initiatives, and KPIs, then generates Mermaid diagram.
"""
import google.generativeai as genai
import re
import json
import logging
from typing import Dict, Optional

from config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    PILLARS_SECTION,
    INITIATIVES_SECTION,
    KPIS_SECTION
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GenAI
genai.configure(api_key=GOOGLE_API_KEY)


def extract_pillars_section(markdown_text: str) -> Optional[str]:
    """
    Extract Section 7.2 (Five Strategic Pillars) content.
    
    Args:
        markdown_text: Full markdown content
        
    Returns:
        Section text or None if not found
    """
    lines = markdown_text.split('\n')
    in_section = False
    section_lines = []
    # More flexible pattern - handles various header formats
    section_start_pattern = re.compile(r'7\.2.*[Ff]ive.*[Ss]trategic.*[Pp]illar', re.IGNORECASE)
    section_end_pattern = re.compile(r'^#+\s*7\.3')
    
    for i, line in enumerate(lines):
        # Check if this line matches the section start (header or content line)
        if section_start_pattern.search(line):
            in_section = True
            # Include the previous line if it's a header marker
            if i > 0 and lines[i-1].strip().startswith('#'):
                section_lines.append(lines[i-1])
            section_lines.append(line)
        elif in_section:
            # Stop at next major section (7.3)
            if section_end_pattern.match(line.strip()):
                break
            section_lines.append(line)
    
    if section_lines:
        return '\n'.join(section_lines)
    return None


def extract_related_sections(markdown_text: str) -> Dict[str, str]:
    """
    Extract Section 7.3 (Initiatives) and 8.2 (KPIs) for mapping.
    
    Args:
        markdown_text: Full markdown content
        
    Returns:
        Dictionary with 'initiatives' and 'kpis' sections
    """
    lines = markdown_text.split('\n')
    sections = {'initiatives': None, 'kpis': None}
    
    # Extract Initiatives (7.3)
    in_initiatives = False
    initiatives_lines = []
    initiatives_start = re.compile(r'#+\s*7\.3.*[Ii]nitiatives')
    initiatives_end = re.compile(r'#+\s*8\.')
    
    for line in lines:
        if initiatives_start.search(line):
            in_initiatives = True
            initiatives_lines.append(line)
        elif in_initiatives:
            if initiatives_end.search(line):
                break
            initiatives_lines.append(line)
    
    if initiatives_lines:
        sections['initiatives'] = '\n'.join(initiatives_lines)
    
    # Extract KPIs (8.2)
    in_kpis = False
    kpis_lines = []
    kpis_start = re.compile(r'#+\s*8\.2.*[Kk]ey.*[Pp]erformance.*[Ii]ndicators')
    kpis_end = re.compile(r'#+\s*8\.3')
    
    for line in lines:
        if kpis_start.search(line):
            in_kpis = True
            kpis_lines.append(line)
        elif in_kpis:
            if kpis_end.search(line):
                break
            kpis_lines.append(line)
    
    if kpis_lines:
        sections['kpis'] = '\n'.join(kpis_lines)
    
    return sections


def parse_strategy_structure(sections_text: Dict[str, str], gemini_client=None) -> Dict:
    """
    Use Gemini to extract structured strategy data.
    
    Args:
        sections_text: Dictionary with 'pillars', 'initiatives', 'kpis' sections
        gemini_client: Optional Gemini client (not needed, we use global config)
        
    Returns:
        Structured dictionary with root, pillars, initiatives, KPIs
    """
    # Combine all sections
    combined_text = f"""
PILLARS SECTION (7.2):
{sections_text.get('pillars', '')}

INITIATIVES SECTION (7.3):
{sections_text.get('initiatives', '')}

KPIs SECTION (8.2):
{sections_text.get('kpis', '')}
"""
    
    prompt = f"""Extract the strategic structure from the following sections of a Kaiser Permanente strategy document.

Please extract:
1. The 5 Strategic Pillars (from Section 7.2)
2. The Strategic Initiatives (from Section 7.3) and map them to their corresponding pillars
3. The Key Performance Indicators (from Section 8.2) and link them to relevant pillars

Return a JSON structure in this exact format:
{{
    "root": "2026 Strategy",
    "pillars": [
        {{
            "id": 1,
            "name": "Full pillar name",
            "initiatives": ["Initiative name 1", "Initiative name 2"],
            "kpis": ["KPI name 1", "KPI name 2"]
        }}
    ]
}}

Ensure:
- All 5 pillars are included
- Initiatives are mapped to their corresponding pillars (matching by name or description)
- KPIs are linked to relevant pillars
- Use the exact names from the document

Document sections:
{combined_text}

Return only valid JSON, no additional text."""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        generation_config = {
            'temperature': 0.1,  # Very low temperature for structured extraction
            'max_output_tokens': 4096,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Parse JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        structure = json.loads(response_text)
        
        # Validate structure
        if 'pillars' not in structure:
            raise ValueError("Missing 'pillars' in response")
        
        logger.info(f"Successfully extracted {len(structure.get('pillars', []))} pillars")
        
        return structure
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from Gemini: {e}")
        logger.error(f"Response was: {response_text[:500]}")
        raise
    except Exception as e:
        logger.error(f"Error extracting strategy structure: {e}")
        raise


def clean_node_text(text: str, max_length: int = 40) -> str:
    """Clean and sanitize text for Mermaid nodes."""
    # Remove quotes, backticks, and other problematic chars
    cleaned = text.replace('"', "'").replace('`', '').replace('\n', ' ').replace('\r', ' ')
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    # Shorten if needed
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length-3] + "..."
    return cleaned


def generate_mermaid_diagram(strategy_structure: Dict) -> str:
    """
    Convert strategy structure to Mermaid graph format with radial layout (mindmap-like).
    Using graph TD format for better compatibility.
    
    Args:
        strategy_structure: Structured dictionary from parse_strategy_structure
        
    Returns:
        Mermaid diagram string
    """
    root = strategy_structure.get('root', '2026 Strategy')
    pillars = strategy_structure.get('pillars', [])
    
    # Use graph TD format for better compatibility across Mermaid versions
    mermaid_lines = ['graph TD']
    
    # Root node in center
    root_id = 'Root'
    root_clean = clean_node_text(root, 30)
    mermaid_lines.append(f'    {root_id}["{root_clean}"]')
    
    # Add pillars as first-level branches
    for pillar in pillars:
        pillar_name = pillar.get('name', 'Unknown Pillar')
        # Clean and simplify pillar name
        pillar_name_clean = pillar_name.replace('Strategic Pillar', 'Pillar').replace(':', '')
        pillar_name_clean = clean_node_text(pillar_name_clean, 50)
        
        pillar_id = f'P{pillar.get("id", 0)}'
        mermaid_lines.append(f'    {pillar_id}["{pillar_name_clean}"]')
        mermaid_lines.append(f'    {root_id} --> {pillar_id}')
        
        # Add initiatives as sub-branches (limit to 2-3)
        initiatives = pillar.get('initiatives', [])
        for i, initiative in enumerate(initiatives[:3]):
            init_name_clean = str(initiative)
            init_name_clean = re.sub(r'^(Initiative \d+:\s*"?\s*)?', '', init_name_clean, flags=re.IGNORECASE)
            init_name_clean = clean_node_text(init_name_clean, 40)
            
            init_id = f'I{pillar.get("id", 0)}{i}'
            mermaid_lines.append(f'    {init_id}["{init_name_clean}"]')
            mermaid_lines.append(f'    {pillar_id} --> {init_id}')
        
        # Add KPIs as sub-branches (limit to 2)
        kpis = pillar.get('kpis', [])
        for i, kpi in enumerate(kpis[:2]):
            kpi_name_clean = clean_node_text(str(kpi), 35)
            kpi_id = f'K{pillar.get("id", 0)}{i}'
            mermaid_lines.append(f'    {kpi_id}["{kpi_name_clean}"]')
            mermaid_lines.append(f'    {pillar_id} --> {kpi_id}')
    
    return '\n'.join(mermaid_lines)


def extract_strategy_structure_from_markdown(markdown_text: str, gemini_client=None) -> Dict:
    """
    Extract and parse the strategy structure (root, pillars, initiatives, KPIs)
    from the full strategy markdown.

    This helper is used both for Mermaid generation and for the interactive
    mind‑map style visualization in the Streamlit app.
    """
    try:
        # Extract pillars section
        pillars_text = extract_pillars_section(markdown_text)
        if not pillars_text:
            raise ValueError("Could not find Section 7.2 (Strategic Pillars)")

        # Extract related sections
        related_sections = extract_related_sections(markdown_text)

        # Combine sections
        sections_text = {
            'pillars': pillars_text,
            'initiatives': related_sections.get('initiatives', ''),
            'kpis': related_sections.get('kpis', '')
        }

        # Parse structure
        structure = parse_strategy_structure(sections_text, gemini_client)
        return structure

    except Exception as e:
        logger.error(f"Error extracting strategy structure: {e}")
        raise


def generate_strategy_graph(markdown_text: str, gemini_client=None) -> str:
    """
    Main function to extract sections and generate Mermaid diagram.

    Kept for compatibility; internally uses the shared structure extraction
    helper so the same data can power other visualizations.
    """
    try:
        structure = extract_strategy_structure_from_markdown(markdown_text, gemini_client)
        mermaid_diagram = generate_mermaid_diagram(structure)
        return mermaid_diagram

    except Exception as e:
        logger.error(f"Error generating strategy graph: {e}")
        raise

