"""
Main ingestion script for processing document and creating vector database.
Run this once to process output.md and create the ChromaDB collection.
"""
import argparse
import logging
import os
from typing import List, Dict

from config import (
    load_config,
    DOCUMENT_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY
)
from document_processor import parse_markdown_file, chunk_by_headers, extract_urls_from_markdown
from hyperlink_handler import create_hyperlink_chunks
from vector_store import (
    initialize_chroma_db,
    generate_embeddings,
    store_chunks,
    collection_exists
)

# Set up logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "ingestion.log"), mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def main(force: bool = False, skip_hyperlinks: bool = False):
    """
    Main ingestion workflow.
    
    Args:
        force: Force re-indexing even if collection exists
        skip_hyperlinks: Skip hyperlink fetching (for faster testing)
    """
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Check if collection exists
        if collection_exists(CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY):
            if not force:
                logger.warning(
                    f"Collection '{CHROMA_COLLECTION_NAME}' already exists. "
                    "Use --force to re-index, or delete the collection first."
                )
                response = input("Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("Ingestion cancelled.")
                    return
            else:
                logger.info(f"Force flag set, will re-index collection '{CHROMA_COLLECTION_NAME}'")
        
        # Parse markdown document
        logger.info(f"Parsing document: {DOCUMENT_PATH}")
        markdown_text = parse_markdown_file(DOCUMENT_PATH)
        logger.info(f"Document parsed successfully ({len(markdown_text)} characters)")
        
        # Header-based chunking
        logger.info("Performing header-based chunking...")
        main_chunks = chunk_by_headers(markdown_text)
        logger.info(f"Created {len(main_chunks)} chunks from main document")
        
        # Add content_type to main chunks
        for chunk in main_chunks:
            chunk['content_type'] = 'main_doc'
        
        # Extract URLs
        logger.info("Extracting URLs from document...")
        urls_with_context = extract_urls_from_markdown(markdown_text, main_chunks)
        logger.info(f"Found {len(urls_with_context)} URLs")
        
        # Process hyperlinks
        hyperlink_chunks = []
        if not skip_hyperlinks and urls_with_context:
            logger.info("Fetching and processing hyperlinks...")
            hyperlink_chunks = create_hyperlink_chunks(urls_with_context)
            logger.info(f"Successfully processed {len(hyperlink_chunks)} hyperlinks")
        elif skip_hyperlinks:
            logger.info("Skipping hyperlink processing (--skip-hyperlinks flag)")
        else:
            logger.info("No URLs found to process")
        
        # Combine all chunks
        all_chunks = main_chunks + hyperlink_chunks
        logger.info(f"Total chunks to index: {len(all_chunks)}")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        embeddings = generate_embeddings(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        if force and collection_exists(CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY):
            # Delete existing collection
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            try:
                client.delete_collection(name=CHROMA_COLLECTION_NAME)
                logger.info(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")
        
        client, collection = initialize_chroma_db(CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY)
        
        # Store chunks
        logger.info("Storing chunks in ChromaDB...")
        store_chunks(collection, all_chunks, embeddings)
        logger.info("Chunks stored successfully")
        
        # Print and log summary
        summary = f"""
{'=' * 80}
INGESTION COMPLETE
{'=' * 80}
Total chunks created: {len(all_chunks)}
  - Main document chunks: {len(main_chunks)}
  - Hyperlink chunks: {len(hyperlink_chunks)}
Collection name: {CHROMA_COLLECTION_NAME}
Collection location: {CHROMA_PERSIST_DIRECTORY}
{'=' * 80}

You can now run the Streamlit app with: streamlit run app.py
"""
        print(summary)
        logger.info(summary.strip())
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Kaiser Strategy document into vector database")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if collection exists"
    )
    parser.add_argument(
        "--skip-hyperlinks",
        action="store_true",
        help="Skip hyperlink fetching (for faster testing)"
    )
    
    args = parser.parse_args()
    
    main(force=args.force, skip_hyperlinks=args.skip_hyperlinks)

