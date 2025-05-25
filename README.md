# Simple RAG System with ChromaDB

A simple Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and ChromaDB vector database.

## Features

- **OpenAI Integration**: Uses OpenAI embeddings and chat models
- **ChromaDB Vector Storage**: Local vector database for document storage
- **Document Processing**: Support for text files and web pages
- **Environment Configuration**: Secure API key management with .env files
- **Interactive Queries**: Command-line interface for asking questions

## Project Structure

```
simple-rag/
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables (create from template)
├── rag_app.py             # Main RAG application
├── document_utils.py      # Document processing utilities
├── example.py             # Demo and example usage
├── README.md              # This file
└── chroma_db/            # ChromaDB storage (created automatically)
```

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd simple-rag
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.template .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

### 3. Get OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Add it to your `.env` file

## Usage

### Quick Start

Run the example script:

```bash
python example.py
```

This will:
- Load sample documents
- Demonstrate basic RAG queries
- Provide an interactive session

### Basic Usage

```python
from rag_app import SimpleRAG

# Initialize RAG system
rag = SimpleRAG()

# Add documents
documents = [
    {
        'text': "Your document content here...",
        'metadata': {'source': 'example'},
        'id': 'doc_1'
    }
]
rag.add_documents(documents)

# Query the system
result = rag.query("Your question here")
print(result['answer'])
```

### Document Processing

```python
from document_utils import DocumentProcessor

processor = DocumentProcessor()

# Process a text file
docs = processor.process_file('path/to/file.txt')
rag.add_documents(docs)

# Process a web page
docs = processor.process_url('https://example.com')
rag.add_documents(docs)

# Process raw text
docs = processor.process_text("Your text content", "source_name")
rag.add_documents(docs)
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `CHROMA_DB_PATH` | `./chroma_db` | Path to ChromaDB storage |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |

### Document Processing

```python
processor = DocumentProcessor(
    chunk_size=1000,      # Size of text chunks
    chunk_overlap=200     # Overlap between chunks
)
```

## API Reference

### SimpleRAG Class

#### Methods

- `add_documents(documents)` - Add documents to vector database
- `search_documents(query, n_results=3)` - Search for relevant documents
- `generate_response(query, context_docs)` - Generate AI response with context
- `query(question, n_results=3)` - Complete RAG pipeline
- `get_collection_info()` - Get collection statistics

#### Document Format

```python
{
    'text': str,           # Document content
    'metadata': dict,      # Optional metadata
    'id': str             # Unique identifier
}
```

### DocumentProcessor Class

#### Methods

- `process_file(file_path)` - Process text file into chunks
- `process_url(url)` - Process web page into chunks  
- `process_text(text, source_name)` - Process raw text into chunks
- `chunk_text(text, metadata)` - Split text into chunks

## Examples

### Interactive Session

```bash
python rag_app.py
```

### Web Page Processing

```python
from rag_app import SimpleRAG
from document_utils import DocumentProcessor

rag = SimpleRAG()
processor = DocumentProcessor()

# Load Wikipedia article
docs = processor.process_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
rag.add_documents(docs)

# Query the content
result = rag.query("What is artificial intelligence?")
print(result['answer'])
```

### File Processing

```python
# Process a text file
docs = processor.process_file("data/document.txt", "research_paper")
rag.add_documents(docs)

# Query with file context
result = rag.query("Summarize the main findings")
```

## Troubleshooting

### Common Issues

**OpenAI API Key Error**
- Ensure your API key is valid and has sufficient credits
- Check the `.env` file is in the project root
- Verify the key starts with `sk-`

**ChromaDB Permission Error**
- Ensure write permissions in the project directory
- Try changing `CHROMA_DB_PATH` to a different location

**Import Errors**
- Run `pip install -r requirements.txt`
- Ensure you're using Python 3.8+

### Performance Tips

- Use smaller chunk sizes for more precise retrieval
- Increase `n_results` for more context in responses
- Adjust `chunk_overlap` based on your document type
- Consider using `text-embedding-3-large` for better quality (higher cost)

## Dependencies

- `openai` - OpenAI API client
- `chromadb` - Vector database
- `python-dotenv` - Environment variable management
- `requests` - HTTP requests for web scraping
- `beautifulsoup4` - HTML parsing
- `langchain-text-splitters` - Text chunking utilities

## License

MIT License - feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

For more advanced RAG implementations, consider:
- Adding support for PDF documents
- Implementing hybrid search (dense + sparse)
- Adding conversation memory
- Using different embedding models
- Implementing document metadata filtering