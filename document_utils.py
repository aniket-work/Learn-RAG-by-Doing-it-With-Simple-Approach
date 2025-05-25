import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor with chunking parameters"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_text_file(self, file_path: str) -> str:
        """Load text from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def load_web_page(self, url: str) -> str:
        """Load text content from a web page"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error loading web page {url}: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks and prepare for vector storage"""
        try:
            if metadata is None:
                metadata = {}
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    'text': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_id': i,
                        'chunk_count': len(chunks)
                    },
                    'id': f"{metadata.get('source', 'doc')}_{i}"
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def process_file(self, file_path: str, source_name: str = None) -> List[Dict[str, Any]]:
        """Process a text file into chunks ready for vector storage"""
        if source_name is None:
            source_name = file_path
        
        text = self.load_text_file(file_path)
        metadata = {
            'source': source_name,
            'type': 'file',
            'file_path': file_path
        }
        
        return self.chunk_text(text, metadata)
    
    def process_url(self, url: str, source_name: str = None) -> List[Dict[str, Any]]:
        """Process a web page into chunks ready for vector storage"""
        if source_name is None:
            source_name = url
        
        text = self.load_web_page(url)
        metadata = {
            'source': source_name,
            'type': 'web',
            'url': url
        }
        
        return self.chunk_text(text, metadata)
    
    def process_text(self, text: str, source_name: str = "manual_input") -> List[Dict[str, Any]]:
        """Process raw text into chunks ready for vector storage"""
        metadata = {
            'source': source_name,
            'type': 'text'
        }
        
        return self.chunk_text(text, metadata)

def load_sample_documents() -> List[Dict[str, Any]]:
    """Load some sample documents for testing"""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    sample_texts = [
        {
            'text': """
            Artificial Intelligence (AI) is a broad field of computer science focused on creating systems 
            capable of performing tasks that typically require human intelligence. These tasks include 
            learning, reasoning, problem-solving, perception, and language understanding.
            
            Machine Learning is a subset of AI that focuses on algorithms that can learn from and make 
            predictions or decisions based on data. Instead of being explicitly programmed for every task, 
            these systems improve their performance through experience.
            
            Deep Learning is a subset of machine learning that uses neural networks with multiple layers 
            to model and understand complex patterns in data. It has been particularly successful in areas 
            like image recognition, natural language processing, and speech recognition.
            """,
            'source': 'ai_overview'
        },
        {
            'text': """
            Python is a high-level, interpreted programming language known for its simplicity and readability. 
            Created by Guido van Rossum and first released in 1991, Python has become one of the most popular 
            programming languages in the world.
            
            Python's design philosophy emphasizes code readability and a syntax that allows programmers to 
            express concepts in fewer lines of code. This makes it an excellent choice for beginners and 
            experienced developers alike.
            
            Python is widely used in various domains including web development, data science, artificial 
            intelligence, scientific computing, automation, and more. Its extensive library ecosystem and 
            active community make it a versatile tool for many applications.
            """,
            'source': 'python_overview'
        },
        {
            'text': """
            Vector databases are specialized databases designed to store, index, and query high-dimensional 
            vector data efficiently. They are essential components in modern AI applications, particularly 
            those involving machine learning and semantic search.
            
            Traditional databases store structured data in rows and columns, but vector databases store 
            mathematical representations of data as vectors in high-dimensional space. This allows for 
            similarity searches based on semantic meaning rather than exact matches.
            
            ChromaDB is an open-source vector database that makes it easy to build AI applications. It 
            provides a simple API for storing embeddings and performing similarity searches, making it 
            ideal for retrieval-augmented generation (RAG) systems.
            """,
            'source': 'vector_db_overview'
        }
    ]
    
    all_documents = []
    for item in sample_texts:
        docs = processor.process_text(item['text'], item['source'])
        all_documents.extend(docs)
    
    return all_documents