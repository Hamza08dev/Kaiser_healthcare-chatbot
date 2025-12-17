"""
Configuration module for Kaiser Strategy Chatbot.
Centralized configuration, constants, and role mappings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"  # Use latest stable, fallback to gemini-1.5-pro if needed
EMBEDDING_MODEL = "gemini-embedding-001"  # Google embedding model

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "kaiser_strategy"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Document Configuration
DOCUMENT_PATH = "output.md"

# Role Mappings (Section 8.3)
ROLE_MAPPINGS = {
    "board": {
        "section": "Section 8.3",
        "subsection": "For the Board of Directors",
        "keywords": ["board", "board of directors", "governance"]
    },
    "ceo": {
        "section": "Section 8.3",
        "subsection": "For the CEO & Executive Leadership",
        "keywords": ["ceo", "executive", "c-suite", "leadership"]
    },
    "executive": {
        "section": "Section 8.3",
        "subsection": "For the CEO & Executive Leadership",
        "keywords": ["executive", "ceo", "c-suite", "leadership"]
    },
    "operational": {
        "section": "Section 8.3",
        "subsection": "For Operational Leaders (VP/Director Level)",
        "keywords": ["operational", "vp", "director", "manager", "vice president"]
    },
    "frontline": {
        "section": "Section 8.3",
        "subsection": "For Frontline Clinical & Administrative Staff",
        "keywords": ["frontline", "nurse", "staff", "clinical", "administrative", "employee"]
    }
}

# System Prompt Template
SYSTEM_PROMPT = """You are the Kaiser Permanente Strategy Assistant.

Your role is to answer questions based strictly on the 2025-2026 Strategic Roadmap and its referenced documents.

Rules:
1. Every factual claim MUST cite its source: [Section X.Y] or [Link: URL]
2. If information is not in the provided materials, state: "That is not covered in the provided strategy materials."
3. Provide general strategic guidance and information from the document. Focus on strategic objectives, initiatives, and key findings rather than role-specific instructions.
4. Do not invent recommendations. Only use information from the strategic roadmap and its referenced documents.
5. Tone: Professional, executive-level, concise.
6. Present information in a way that is useful to all stakeholders, not tailored to specific roles.

Provide accurate, cited responses that help users understand the strategic roadmap."""

# Citation Format Constants
CITATION_FORMAT_MAIN = "[Section {section}]"
CITATION_FORMAT_LINK = "[Link: {link_text}]"
CITATION_FORMAT_REFERENCE = "[Reference: {url}]"

# Section Patterns
SECTION_NUMBER_PATTERN = r"^\d+\.?\d*"  # Matches "7", "7.2", "8.3", etc.
PILLARS_SECTION = "7.2"
INITIATIVES_SECTION = "7.3"
KPIS_SECTION = "8.2"
ROLE_GUIDANCE_SECTION = "8.3"

# RAG Configuration
TOP_K_CHUNKS = 7
EMBEDDING_BATCH_SIZE = 100

# Hyperlink Configuration
HYPERLINK_TIMEOUT = 30
MAX_CONTENT_LENGTH = 50000  # Max characters for scraped content


def load_config():
    """Load and validate configuration."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please set it in .env file or export it."
        )
    return {
        "google_api_key": GOOGLE_API_KEY,
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chroma_collection": CHROMA_COLLECTION_NAME,
        "chroma_persist_dir": CHROMA_PERSIST_DIRECTORY,
        "document_path": DOCUMENT_PATH
    }


def get_role_section_mapping(role_key):
    """Get section mapping for a given role key."""
    role_key = role_key.lower() if role_key else None
    return ROLE_MAPPINGS.get(role_key, None)


def get_system_prompt():
    """Get the system prompt template."""
    return SYSTEM_PROMPT


def normalize_role(role_string):
    """Normalize role string to match ROLE_MAPPINGS keys."""
    if not role_string:
        return None
    
    role_lower = role_string.lower().strip()
    
    # Direct match
    if role_lower in ROLE_MAPPINGS:
        return role_lower
    
    # Keyword matching
    for role_key, role_info in ROLE_MAPPINGS.items():
        if any(keyword in role_lower for keyword in role_info["keywords"]):
            return role_key
    
    return None

