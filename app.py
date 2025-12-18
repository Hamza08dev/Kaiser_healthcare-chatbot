"""
Streamlit application for Kaiser Strategy Chatbot.
Provides chat interface and strategy graph visualization.
"""
import streamlit as st
import logging
from typing import Optional, Dict
import inspect

from config import (
    load_config,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    DOCUMENT_PATH
)
from vector_store import initialize_chroma_db, collection_exists
import rag_handler
from document_processor import parse_markdown_file
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Kaiser Strategy Assistant | Business Optima",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp li, .stApp span, .stApp div {
        color: #fafafa !important;
    }
    .citation {
        background-color: #1e293b;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.9em;
        color: #60a5fa;
    }
    .user-message {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: #fafafa;
    }
    .assistant-message {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_resources():
    """Initialize ChromaDB and GenAI clients (cached)."""
    try:
        config = load_config()
        
        # Initialize GenAI
        genai.configure(api_key=config['google_api_key'])
        
        # Initialize ChromaDB
        if not collection_exists(CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY):
            st.error(
                f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' not found. "
                "Please run `python ingest.py` first."
            )
            st.stop()
        
        client, collection = initialize_chroma_db(CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY)
        
        return {
            'collection': collection,
            'config': config,
            'initialized': True
        }
    except Exception as e:
        st.error(f"Error initializing resources: {str(e)}")
        st.stop()


def format_message_with_citations(text: str) -> str:
    """Format message text, optionally handling citations.

    We now hide inline citation markers like [Section X.Y] or [Link: ...]
    from the rendered chat, since the detailed citations are already
    available in the separate "View Sources" expander.
    """
    import re

    # Remove inline citation markers such as:
    # [Section 3.1], [Link: Q1 2025 Financial Update], [Reference: ...]
    text = re.sub(r'\[Section [^\]]+\]', '', text)
    text = re.sub(r'\[Link: [^\]]+\]', '', text)
    text = re.sub(r'\[Reference: [^\]]+\]', '', text)

    # Convert simple markdown bold (**text**) to <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Respect any newlines from the model
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    text = text.replace('\n\n', '<br/><br/>')
    text = text.replace('\n', '<br/>')

    # Add line breaks before numbered items like "1." "2." when they appear
    # in the middle of a paragraph, to avoid one huge block of text.
    text = re.sub(r'\s(\d+\.)\s', r'<br/><br/>\1 ', text)

    # Clean up any excessive spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text


def main():
    """Main Streamlit application."""
    st.title("🏥 Kaiser Permanente Strategy Assistant")
    st.markdown("Ask questions about the 2025-2026 Strategic Roadmap")
    
    # Initialize resources
    resources = initialize_resources()
    collection = resources['collection']
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        # Logo at the top-left of sidebar
        st.image("bo_logo.jpeg", width=150)
        st.markdown("---")
        
        st.header("⚙️ Configuration")
        
        # Response style selector (replaces deprecated role selector)
        response_style = st.selectbox(
            "Answer Length",
            ["Concise", "Detailed"],
            help="Choose how long you want the assistant's answers to be."
        )
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Info
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This assistant provides answers based on the Kaiser Permanente "
            "2025-2026 Strategic Roadmap. All responses are grounded in the "
            "source document with citations."
        )
    
    # Main area with separate tabs for Chat and Strategy Graph
    chat_tab, graph_tab = st.tabs(["💬 Chat", "📊 Strategy Graph"])

    with chat_tab:
        # Chat interface
        st.header("💬 Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                role = message['role']
                content = message['content']
                
                if role == 'user':
                    st.markdown(
                        f'<div class="user-message"><strong>You:</strong> {content}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    formatted_content = format_message_with_citations(content)
                    st.markdown(
                        f'<div class="assistant-message"><strong>Assistant:</strong> {formatted_content}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show sources if available
                    if 'sources' in message and message['sources']:
                        with st.expander("View Sources"):
                            for source in message['sources']:
                                if source['type'] == 'section':
                                    st.write(
                                        f"📄 Section {source.get('section', 'N/A')}: {source.get('path', '')}"
                                    )
                                elif source['type'] == 'link':
                                    st.write(
                                        f"🔗 {source.get('text', '')}: {source.get('url', '')}"
                                    )
        
        # Chat input
        user_input = st.chat_input("Ask a question about the strategy...")
        
        if user_input:
            # Add user message to history
            st.session_state.messages.append({'role': 'user', 'content': user_input})
            
            # Get assistant response
            with st.spinner("Thinking..."):
                try:
                    # Backwards-compatible call to query_rag:
                    # Only pass response_style if the deployed function supports it.
                    query_fn = rag_handler.query_rag
                    sig = inspect.signature(query_fn)
                    extra_kwargs = {}
                    if "response_style" in sig.parameters:
                        extra_kwargs["response_style"] = response_style

                    result = query_fn(
                        user_query=user_input,
                        collection=collection,
                        user_role=None,
                        top_k=5,
                        **extra_kwargs
                    )
                    
                    response = result['response']
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': response,
                        'sources': result.get('sources', [])
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    error_message = f"Error processing query: {str(e)}"
                    st.error(error_message)
                    logger.error(f"Query error: {e}", exc_info=True)
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': error_message
                    })
                    st.rerun()

    with graph_tab:
        # Strategy graph display
        st.header("📊 Strategy Graph")
        
        # Display React Flow interactive graph
        try:
            import streamlit.components.v1 as components
            import os
            
            # Read the React Flow HTML file
            html_file_path = os.path.join(os.path.dirname(__file__), 'reactflow_graph.html')
            with open(html_file_path, 'r', encoding='utf-8') as f:
                reactflow_html = f.read()
            
            components.html(reactflow_html, height=800, scrolling=False)
            
        except Exception as e:
            st.error(f"Error rendering interactive graph: {str(e)}")
            logger.error(f"Graph rendering error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

