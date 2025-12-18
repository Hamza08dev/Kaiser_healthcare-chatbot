"""
RAG handler for query processing and response generation.
Handles role detection, prompt construction, and Gemini integration.
"""
import google.generativeai as genai
from typing import List, Dict, Optional
import logging

from config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    SYSTEM_PROMPT,
    get_role_section_mapping,
    normalize_role,
    CITATION_FORMAT_MAIN,
    CITATION_FORMAT_LINK
)
from vector_store import query_collection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GenAI
genai.configure(api_key=GOOGLE_API_KEY)


def detect_role_from_query(query: str, selected_role: Optional[str] = None) -> Optional[str]:
    """
    Detect user role from query or selected role.
    
    Args:
        query: User query text
        selected_role: Role selected from dropdown
        
    Returns:
        Normalized role string or None
    """
    if selected_role and selected_role.lower() != 'general':
        return normalize_role(selected_role)
    
    # Try to detect from query text
    return normalize_role(query)


def is_advice_request(query: str) -> bool:
    """
    Detect if the user is asking for advice vs just information.
    
    Args:
        query: User's query text
        
    Returns:
        True if user is requesting advice, False if just asking a question
    """
    advice_keywords = [
        'advice', 'what should', 'recommendation', 'recommend', 'guidance',
        'how should', 'what would you suggest', 'suggest', 'suggestion',
        'what do you advise', 'what action', 'what steps', 'should i',
        'what would be best', 'best approach', 'best way', 'guidance on', 'how can i'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in advice_keywords)


def build_rag_prompt(user_query: str, 
                     retrieved_chunks: List[Dict], 
                     user_role: Optional[str] = None,
                     is_advice: bool = False) -> str:
    """
    Construct RAG prompt with system prompt and retrieved chunks.
    
    Args:
        user_query: User's question
        retrieved_chunks: List of retrieved chunks from vector store
        user_role: Optional user role for context (not used for role-specific filtering)
        is_advice: Whether user is requesting advice (vs just information)
        
    Returns:
        Formatted prompt string
    """
    # Use different prompt based on whether user wants advice or information
    if is_advice:
        base_prompt = """You are the Kaiser Permanente Strategy Assistant.

Your role is to provide strategic advice and actionable guidance based on the 2025-2026 Strategic Roadmap and its referenced documents.

Rules:
1. Every factual claim MUST cite its source: [Section X.Y] or [Link: URL]
2. If information is not in the provided materials, state: "That is not covered in the provided strategy materials."
3. Provide actionable strategic advice and guidance based on the strategic roadmap. Focus on recommendations, best practices, and strategic actions that align with the document's objectives and initiatives.
4. Use information from "Calls to Action", strategic initiatives, and implementation guidance sections.
5. Do not invent recommendations. Only provide advice based on the strategic roadmap and its referenced documents.
6. Tone: Professional, executive-level, actionable, and consultative.
7. Present advice in a way that is useful to all stakeholders, providing strategic recommendations for implementation.

Provide strategic advice that helps users understand what actions to take based on the strategic roadmap."""
    else:
        base_prompt = SYSTEM_PROMPT
    
    prompt_parts = [base_prompt]
    
    # Add retrieved context with citations
    if retrieved_chunks:
        context_label = "Relevant context from the strategy document" if not is_advice else "Strategic guidance and recommendations from the strategy document"
        prompt_parts.append(f"\n\n{context_label}:")
        prompt_parts.append("=" * 80)
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            section_path = metadata.get('section_path', '')
            section_number = metadata.get('section_number', '')
            content_type = metadata.get('content_type', 'main_doc')
            
            # Format citation
            if content_type == 'hyperlink':
                link_text = metadata.get('link_text', metadata.get('source_url', 'Unknown'))
                citation = CITATION_FORMAT_LINK.format(link_text=link_text)
            elif section_number:
                citation = CITATION_FORMAT_MAIN.format(section=section_number)
                if section_path:
                    citation += f" ({section_path})"
            else:
                citation = f"[{section_path}]" if section_path else "[Unknown]"
            
            prompt_parts.append(f"\n[{i}] {citation}\n{content}\n")
        
        prompt_parts.append("=" * 80)
    
    # Add user query with appropriate instruction based on mode
    prompt_parts.append(f"\n\nUser Query: {user_query}")
    
    if is_advice:
        prompt_parts.append("\n\nPlease provide strategic advice and actionable recommendations based on the context above. "
                           "Focus on what actions should be taken, best practices, and implementation guidance. "
                           "Use information from 'Calls to Action', strategic initiatives, and implementation sections. "
                           "Remember to cite all recommendations using the format [Section X.Y] or [Link: URL]. "
                           "If the requested advice is not in the provided context, state that clearly.")
    else:
        prompt_parts.append("\n\nPlease provide a comprehensive answer based on the context above. "
                           "Remember to cite all facts using the format [Section X.Y] or [Link: URL]. "
                           "If information is not in the provided context, state that clearly.")
    
    return "\n".join(prompt_parts)


def format_citations(response_text: str, chunks: List[Dict]) -> str:
    """
    Post-process response to ensure citations are clear.
    
    Args:
        response_text: LLM response text
        chunks: Retrieved chunks for reference
        
    Returns:
        Response text with improved citation formatting
    """
    import re

    # Insert a space between numbers and letters when they run together
    # e.g., "3.3billioninQ2" -> "3.3 billionin Q2" (and then another pass)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', response_text)
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)

    return text


def query_rag(user_query: str, 
              collection, 
              user_role: Optional[str] = None, 
              top_k: int = 7,
              response_style: str = "Detailed") -> Dict:
    """
    Main RAG query function.
    Note: user_role parameter is kept for API compatibility but not used for filtering.
    Advice is now general and not role-specific.
    """
    """
    Main RAG query function.
    
    Args:
        user_query: User's question
        collection: ChromaDB collection
        user_role: Optional user role
        top_k: Number of chunks to retrieve
        response_style: Controls answer length / level of detail ("Concise" or "Detailed")
        
    Returns:
        Dictionary with:
        - response: LLM response text
        - sources: List of source citations
        - role_detected: Detected or provided role
    """
    try:
        # Detect role (for logging only, not used for filtering)
        detected_role = detect_role_from_query(user_query, user_role)
        
        # Detect if user is asking for advice vs information
        is_advice = is_advice_request(user_query)
        query_type = "advice" if is_advice else "information"
        logger.info(f"Query type: {query_type}, Role: {detected_role}")
        
        # Query vector store (no role filtering - provide general information/advice)
        retrieved_chunks = query_collection(
            collection=collection,
            query_text=user_query,
            top_k=top_k,
            role_filter=None  # No role filtering - general approach
        )
        
        if not retrieved_chunks:
            return {
                'response': "I couldn't find relevant information in the strategy documents to answer your question. "
                           "Please try rephrasing your query or asking about a different topic.",
                'sources': [],
                'role_detected': detected_role
            }
        
        # Build prompt with advice/information mode
        prompt = build_rag_prompt(user_query, retrieved_chunks, detected_role, is_advice=is_advice)

        # Optionally constrain length for concise answers
        if response_style.lower() == "concise":
            prompt += (
                "\n\nPlease keep your answer concise: no more than about 200 words and "
                "at most 3–5 bullet points."
            )
        
        # Call Gemini
        logger.info(f"Calling Gemini model: {GEMINI_MODEL}")
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Configure generation
        # Adjust max output tokens based on desired style
        max_tokens = 1024 if response_style.lower() == "concise" else 4096

        generation_config = {
            'temperature': 0.3,  # Lower temperature for more factual responses
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': max_tokens,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        response_text = response.text
        
        # Format citations
        formatted_response = format_citations(response_text, retrieved_chunks)
        
        # Extract sources
        sources = []
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            if metadata.get('content_type') == 'hyperlink':
                sources.append({
                    'type': 'link',
                    'text': metadata.get('link_text', ''),
                    'url': metadata.get('source_url', '')
                })
            else:
                sources.append({
                    'type': 'section',
                    'section': metadata.get('section_number', ''),
                    'path': metadata.get('section_path', '')
                })
        
        return {
            'response': formatted_response,
            'sources': sources,
            'role_detected': detected_role
        }
        
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        return {
            'response': f"I encountered an error while processing your query: {str(e)}. Please try again.",
            'sources': [],
            'role_detected': None
        }

