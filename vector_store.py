"""
Vector store operations using ChromaDB and Google embeddings.
Handles embedding generation, chunk storage, and semantic search.
"""
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from typing import List, Dict, Optional, Tuple
import logging
import os

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    GOOGLE_API_KEY,
    ROLE_GUIDANCE_SECTION
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_genai():
    """Initialize Google GenAI client."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in config")
    genai.configure(api_key=GOOGLE_API_KEY)


def initialize_chroma_db(collection_name: str = CHROMA_COLLECTION_NAME, 
                         persist_directory: str = CHROMA_PERSIST_DIRECTORY) -> Tuple[chromadb.Client, chromadb.Collection]:
    """
    Create or connect to ChromaDB collection.
    
    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist ChromaDB data
        
    Returns:
        Tuple of (ChromaDB client, Collection)
    """
    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Loaded existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created new collection: {collection_name}")
    
    return client, collection


def generate_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Generate embeddings using Google GenAI.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model name
        
    Returns:
        List of embedding vectors
    """
    initialize_genai()
    
    embeddings = []
    
    # Process embeddings one at a time for reliability
    # (Google GenAI API batch behavior can be unpredictable)
    for i, text in enumerate(texts):
        try:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            # Handle response structure
            if isinstance(result, dict):
                if 'embedding' in result:
                    embedding = result['embedding']
                elif 'embeddings' in result:
                    # Shouldn't happen for single text, but handle it
                    emb = result['embeddings']
                    embedding = emb[0] if isinstance(emb, list) and len(emb) > 0 else emb
                else:
                    logger.error(f"Unexpected result structure. Keys: {list(result.keys())}")
                    raise ValueError(f"No embedding key found in result: {list(result.keys())}")
            elif isinstance(result, list):
                embedding = result[0] if len(result) > 0 else result
            else:
                # If result is directly the embedding vector
                embedding = result
            
            embeddings.append(embedding)
            
            # Log progress every 10 embeddings
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{len(texts)} embeddings")
            
        except Exception as e:
            logger.error(f"Error generating embedding for text {i+1}: {e}", exc_info=True)
            raise ValueError(f"Failed to generate embedding for chunk {i+1}. Error: {e}")
    
    return embeddings


def store_chunks(collection: chromadb.Collection, chunks: List[Dict], embeddings: List[List[float]]):
    """
    Store chunks with metadata in ChromaDB.
    
    Args:
        collection: ChromaDB collection
        chunks: List of chunk dictionaries
        embeddings: List of embedding vectors (one per chunk)
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
    
    ids = []
    documents = []
    metadatas = []
    embedding_list = []
    
    for i, chunk in enumerate(chunks):
        # Generate ID
        chunk_id = f"chunk_{i}_{hash(chunk.get('section_path', ''))}"
        ids.append(chunk_id)
        
        # Document content
        documents.append(chunk['content'])
        
        # Metadata - ChromaDB doesn't accept None values, convert to empty strings
        metadata = {
            'section_path': chunk.get('section_path') or '',
            'section_number': str(chunk.get('section_number')) if chunk.get('section_number') is not None else '',
            'content_type': chunk.get('content_type', 'main_doc'),
            'level': str(chunk.get('level', 0)),
            'header_text': chunk.get('header_text') or '',
            'line_start': str(chunk.get('line_start', 0)),
            'line_end': str(chunk.get('line_end', 0))
        }
        
        # Add hyperlink-specific metadata
        if chunk.get('content_type') == 'hyperlink':
            metadata['parent_section'] = chunk.get('parent_section') or ''
            metadata['source_url'] = chunk.get('source_url') or ''
            metadata['link_text'] = chunk.get('link_text') or ''
        
        # Add role context if from Section 8.3
        section_num = chunk.get('section_number')
        if section_num and str(section_num) == ROLE_GUIDANCE_SECTION:
            header_lower = chunk.get('header_text', '').lower()
            if 'board' in header_lower:
                metadata['role_context'] = 'board'
            elif 'ceo' in header_lower or 'executive' in header_lower:
                metadata['role_context'] = 'ceo'
            elif 'operational' in header_lower:
                metadata['role_context'] = 'operational'
            elif 'frontline' in header_lower:
                metadata['role_context'] = 'frontline'
        
        metadatas.append(metadata)
        embedding_list.append(embeddings[i])
    
    # Add to collection
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list
        )
        logger.info(f"Stored {len(chunks)} chunks in collection")
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        raise


def query_collection(collection: chromadb.Collection, 
                     query_text: str, 
                     top_k: int = 5,
                     role_filter: Optional[str] = None) -> List[Dict]:
    """
    Perform semantic search in ChromaDB collection.
    
    Args:
        collection: ChromaDB collection
        query_text: Query text
        top_k: Number of results to return
        role_filter: Optional role to filter/prioritize (e.g., 'frontline', 'board')
        
    Returns:
        List of retrieved chunks with metadata
    """
    initialize_genai()
    
    # Generate query embedding
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query_text,
            task_type="RETRIEVAL_QUERY"
        )
        
        # Handle different response structures from Google GenAI API
        # The API can return: dict with 'embedding' or 'embeddings' key, or direct list
        if isinstance(result, dict):
            if 'embedding' in result:
                query_embedding = result['embedding']
            elif 'embeddings' in result:
                # If embeddings is a list, take first element
                emb = result['embeddings']
                query_embedding = emb[0] if isinstance(emb, list) and len(emb) > 0 else emb
            else:
                # Debug: log available keys
                logger.error(f"Unexpected result structure. Keys: {list(result.keys())}")
                return []
        elif isinstance(result, list):
            # If result is directly a list, take first element
            query_embedding = result[0] if len(result) > 0 else result
        else:
            logger.error(f"Unexpected result type: {type(result)}")
            return []
            
        # Ensure query_embedding is a list/array
        if not isinstance(query_embedding, list):
            query_embedding = list(query_embedding) if hasattr(query_embedding, '__iter__') else [query_embedding]
                
    except KeyError as e:
        logger.error(f"Error accessing embedding from result. Result type: {type(result)}. Error: {e}")
        if isinstance(result, dict):
            logger.error(f"Available keys: {list(result.keys())}")
        return []
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        return []
    
    # Build where clause if role filter provided
    where_clause = None
    if role_filter:
        # Prioritize Section 8.3 chunks for the role
        where_clause = {
            "$or": [
                {"role_context": role_filter},
                {"section_number": ROLE_GUIDANCE_SECTION}
            ]
        }
    
    try:
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2 if role_filter else top_k,  # Get more if filtering
            where=where_clause
        )
        
        # If role filter, we got more results, now prioritize and limit
        retrieved_chunks = []
        
        # Process results
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk_dict = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_chunks.append(chunk_dict)
            
            # If role filter, prioritize role-specific chunks and limit to top_k
            if role_filter:
                role_chunks = [c for c in retrieved_chunks if c['metadata'].get('role_context') == role_filter]
                other_chunks = [c for c in retrieved_chunks if c['metadata'].get('role_context') != role_filter]
                retrieved_chunks = (role_chunks + other_chunks)[:top_k]
            else:
                retrieved_chunks = retrieved_chunks[:top_k]
        
        return retrieved_chunks
        
    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        return []


def collection_exists(collection_name: str = CHROMA_COLLECTION_NAME,
                      persist_directory: str = CHROMA_PERSIST_DIRECTORY) -> bool:
    """
    Check if ChromaDB collection already exists.
    
    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist ChromaDB data
        
    Returns:
        True if collection exists, False otherwise
    """
    try:
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)
        client.get_collection(name=collection_name)
        return True
    except Exception:
        return False

