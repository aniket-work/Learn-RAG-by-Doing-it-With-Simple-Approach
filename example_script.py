#!/usr/bin/env python3
"""
Example script demonstrating how to use the Simple RAG system
"""

from rag_app import SimpleRAG
from document_utils import DocumentProcessor, load_sample_documents
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def demo_basic_usage():
    """Demonstrate basic RAG functionality"""
    print("=== Basic RAG Demo ===\n")
    
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Check current collection status
    info = rag.get_collection_info()
    print(f"Collection: {info['collection_name']}")
    print(f"Current documents: {info['document_count']}\n")
    
    # Load sample documents if collection is empty
    if info['document_count'] == 0:
        print("Loading sample documents...")
        sample_docs = load_sample_documents()
        rag.add_documents(sample_docs)
        print(f"Added {len(sample_docs)} document chunks\n")
    
    # Demo queries
    demo_questions = [
        "What is artificial intelligence?",
        "Tell me about Python programming language",
        "What are vector databases used for?",
        "How does machine learning work?"
    ]
    
    print("Running demo queries...\n")
    for question in demo_questions:
        print(f"Q: {question}")
        try:
            result = rag.query(question, n_results=2)
            print(f"A: {result['answer']}")
            print(f"Sources: {[s['id'] for s in result['sources']]}\n")
        except Exception as e:
            print(f"Error: {e}\n")

def demo_document_processing():
    """Demonstrate document processing capabilities"""
    print("=== Document Processing Demo ===\n")
    
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    
    # Example: Process a web page
    try:
        print("Processing web page...")
        url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        web_docs = processor.process_url(url, "AI_Wikipedia")
        print(f"Created {len(web_docs)} chunks from web page")
        
        # Show first chunk
        if web_docs:
            print(f"First chunk preview: {web_docs[0]['text'][:200]}...\n")
    
    except Exception as e:
        print(f"Web processing failed: {e}")
        print("This is normal if you don't have internet access\n")
    
    # Example: Process text file
    try:
        print("To process a text file, use:")
        print("docs = processor.process_file('path/to/your/file.txt')")
        print("rag.add_documents(docs)\n")
    except Exception as e:
        print(f"File processing example: {e}\n")

def interactive_demo():
    """Interactive query session"""
    print("=== Interactive RAG Session ===")
    print("Ask questions about the loaded documents")
    print("Commands: 'info' for collection info, 'quit' to exit\n")
    
    rag = SimpleRAG()
    
    # Ensure we have some documents
    info = rag.get_collection_info()
    if info['document_count'] == 0:
        print("Loading sample documents first...")
        sample_docs = load_sample_documents()
        rag.add_documents(sample_docs)
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'info':
                info = rag.get_collection_info()
                print(f"Collection: {info['collection_name']} ({info['document_count']} documents)")
                continue
            elif not query:
                continue
            
            result = rag.query(query)
            print(f"\nAnswer: {result['answer']}")
            
            print(f"\nSources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['id']} (similarity: {1-source['distance']:.3f})")
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    """Main demo function"""
    print("Simple RAG System Demo\n")
    
    demos = {
        '1': ('Basic Usage Demo', demo_basic_usage),
        '2': ('Document Processing Demo', demo_document_processing), 
        '3': ('Interactive Session', interactive_demo),
        '4': ('Run All Demos', lambda: [demo_basic_usage(), demo_document_processing()])
    }
    
    print("Choose a demo:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    
    choice = input("\nEnter choice (or press Enter for interactive): ").strip()
    
    if choice in demos:
        demos[choice][1]()
    else:
        interactive_demo()

if __name__ == "__main__":
    main()