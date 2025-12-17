# Kaiser Strategy Chatbot Demo

A high-precision RAG-based chatbot for querying the Kaiser Permanente 2025-2026 Strategic Roadmap. This system provides accurate, citation-backed answers with a visual strategy graph.

## Features

- **High-Precision Q&A**: Accurate answers with specific section citations
- **Visual Strategy Map**: Interactive graph visualization of strategic pillars and initiatives
- **Deep Knowledge Coverage**: Answers using both the main document and referenced hyperlinks
- **Role-Aware Responses**: Tailored guidance based on user role (Board, CEO, Frontline, etc.)
- **Header-Based Chunking**: Preserves complete strategic units for accurate retrieval

## Architecture

The system uses a simplified two-script workflow:

- **`ingest.py`**: One-time document processing and vector database creation
- **`app.py`**: Streamlit demo interface for chat and graph visualization

### Technology Stack

- **LLM**: Google GenAI Gemini 2.0 Flash (or latest stable)
- **Embeddings**: Google GenAI text-embedding-004
- **Vector Store**: ChromaDB (local persistent storage)
- **UI Framework**: Streamlit
- **Document Processing**: Python markdown parser + BeautifulSoup

## File Structure

```
kaiser_chatbot/
├── ingest.py                  # Main ingestion orchestration script
├── app.py                     # Main Streamlit application
├── document_processor.py      # Header-based chunking logic
├── hyperlink_handler.py       # URL extraction and content scraping
├── vector_store.py            # ChromaDB operations and embeddings
├── rag_handler.py             # RAG query logic and prompt construction
├── graph_extractor.py         # Strategy graph extraction
├── config.py                  # Configuration, constants, and role mappings
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create from .env.example)
├── .env.example               # Template for environment variables
├── README.md                  # This file
├── output.md                  # Source document
└── chroma_db/                 # ChromaDB persistent storage (created by ingest.py)
```

## Quick Start

Follow these steps in order to get the chatbot running:

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Up Your Gemini API Key

**⚠️ IMPORTANT: You must complete this step before running the application.**

1. Get your Google Gemini API key from: https://aistudio.google.com/app/apikey

2. Create a `.env` file in the project root (you can copy from the example):
   ```bash
   cp .env.example .env
   ```
   
   Or create it manually and add your API key:
   ```
   GOOGLE_API_KEY=your-google-api-key-here
   ```
   
   **Windows users:** You can create the `.env` file by:
   - Opening a text editor
   - Adding the line: `GOOGLE_API_KEY=your-actual-api-key-here`
   - Saving it as `.env` (make sure it's named exactly `.env`, not `.env.txt`)

3. Replace `your-google-api-key-here` with your actual API key from Google AI Studio.

### Step 3: Run Ingestion (One-Time Setup)

After setting up your API key, run the ingestion script to process the document and create the vector database:

```bash
python ingest.py
```

This will:
- Parse `output.md` using header-based chunking
- Extract and scrape hyperlinks
- Generate embeddings for all chunks
- Store everything in ChromaDB (creates the `chroma_db/` folder)

**Note:** This step only needs to be run once (or if you want to rebuild the database).

**Options:**
- `--force`: Force re-indexing even if collection exists
- `--skip-hyperlinks`: Skip hyperlink fetching (for faster testing)

Example:
```bash
python ingest.py --force  # Re-index everything
python ingest.py --skip-hyperlinks  # Skip hyperlink processing
```

### Step 4: Run the Streamlit App

Once ingestion is complete, start the chat interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Chat Interface

1. **Select Your Role**: Choose your role from the sidebar (Board, CEO/Executive, Operational Leaders, Frontline Staff, or General)
2. **Ask Questions**: Type your question in the chat input
3. **View Citations**: Responses include inline citations like `[Section 7.2]` or `[Link: URL]`
4. **View Sources**: Click "View Sources" to see all referenced sections and links

### Strategy Graph

1. Click "Generate Strategy Graph" in the sidebar
2. The graph shows:
   - Root: "2026 Strategy"
   - Level 1: 5 Strategic Pillars
   - Level 2: Strategic Initiatives mapped to pillars
   - Level 3: KPIs linked to relevant pillars

## How It Works

### Header-Based Chunking

The document is split strictly by markdown headers (`#`, `##`, `###`), preserving complete strategic units. Each chunk maintains:
- Full section hierarchy
- Source line numbers for citations
- Complete strategic content (no splitting mid-section)

### Hyperlink Handling

1. URLs are extracted from the document (Section 9 + inline links)
2. Content is fetched and parsed (HTML/PDF)
3. Child knowledge units are created with parent section references
4. Linked content supplements answers but doesn't override main document

### RAG Query Process

1. **Role Detection**: Detects user role from dropdown or query text
2. **Semantic Search**: Queries ChromaDB for relevant chunks (prioritizes Section 8.3 for roles)
3. **Prompt Construction**: Builds prompt with system instructions, retrieved chunks, and citations
4. **Response Generation**: Uses Gemini to generate grounded, cited responses
5. **Citation Enforcement**: All facts must cite `[Section X.Y]` or `[Link: URL]`

### Strategy Graph Generation

1. Extracts Section 7.2 (Strategic Pillars), Section 7.3 (Initiatives), and Section 8.2 (KPIs)
2. Uses Gemini to parse and structure the relationships
3. Generates Mermaid.js diagram for visualization

## Configuration

Key settings in `config.py`:

- `GEMINI_MODEL`: LLM model name (default: "gemini-2.0-flash-exp")
- `EMBEDDING_MODEL`: Embedding model (default: "models/text-embedding-004")
- `CHROMA_COLLECTION_NAME`: Vector database collection name
- `TOP_K_CHUNKS`: Number of chunks to retrieve (default: 7)
- `ROLE_MAPPINGS`: Role to section mappings for Section 8.3

## Troubleshooting

### "Collection not found" Error

Run `python ingest.py` first to create the vector database.

### "API Key not found" Error

Make sure your `.env` file contains `GOOGLE_API_KEY` or export it as an environment variable.

### Hyperlink Fetching Fails

Some URLs may be inaccessible or timeout. The system will skip failed URLs and continue. Check logs for details.

### Graph Generation Fails

Ensure Section 7.2 exists in `output.md`. The system expects specific section headers. Check logs for parsing errors.

## Citation Format

- **Main Document**: `[Section 7.2]` or `[Section 7.2 - Strategic Pillar 1]`
- **Hyperlinks**: `[Link: Q3 2025 Financial Update]` or `[Reference: https://...]`

## Role Mappings

- **Board** → Section 8.3 "For the Board of Directors"
- **CEO/Executive** → Section 8.3 "For the CEO & Executive Leadership"
- **Operational Leaders** → Section 8.3 "For Operational Leaders (VP/Director Level)"
- **Frontline Staff** → Section 8.3 "For Frontline Clinical & Administrative Staff"
- **General** → No role filtering

## Development

### Project Structure

The codebase is modular with clear separation of concerns:

- `config.py`: Centralized configuration
- `document_processor.py`: Markdown parsing and chunking
- `hyperlink_handler.py`: URL fetching and content extraction
- `vector_store.py`: ChromaDB and embedding operations
- `rag_handler.py`: Query processing and response generation
- `graph_extractor.py`: Strategy graph extraction
- `ingest.py`: Ingestion orchestration
- `app.py`: Streamlit UI

### Adding Features

1. **New Chunking Strategy**: Modify `document_processor.py`
2. **New Embedding Model**: Update `EMBEDDING_MODEL` in `config.py`
3. **Custom Prompts**: Modify `SYSTEM_PROMPT` in `config.py`
4. **UI Enhancements**: Edit `app.py` Streamlit components

## License

Internal use only - Kaiser Permanente Strategy Demo

## Support

For issues or questions, check the logs in the console output or Streamlit logs.

