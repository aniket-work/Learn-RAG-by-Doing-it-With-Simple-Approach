import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging
import colorama
from colorama import Fore, Style

# Load environment variables
load_dotenv()
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self):
        """Initialize the RAG system with OpenAI and ChromaDB"""
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
        
        # Get or create collection
        self.collection_name = "rag_documents"
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")
        
        # Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database
        
        Args:
            documents: List of dictionaries with 'text', 'metadata', and optional 'id'
        """
        try:
            texts = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                text = doc['text']
                metadata = doc.get('metadata', {})
                doc_id = doc.get('id', f"doc_{i}")
                
                # Create embedding
                embedding = self.create_embedding(text)
                
                texts.append(text)
                embeddings.append(embedding)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to the collection")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            search_results = []
            for i in range(len(results['documents'][0])):
                search_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using OpenAI with retrieved context"""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([doc['text'] for doc in context_docs])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain relevant information, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Complete RAG pipeline: search and generate"""
        try:
            # Search for relevant documents
            relevant_docs = self.search_documents(question, n_results)
            
            # Generate response
            response = self.generate_response(question, relevant_docs)
            
            return {
                'question': question,
                'answer': response,
                'sources': relevant_docs
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {'error': str(e)}

def main():
    """Example usage"""
    rag = SimpleRAG()
    
    # Example documents
    sample_docs = [
        {
            'text': "A Next.js and Tailwind CSS blogging starter template, forked from timlrx/tailwind-nextjs-starter-blog. Easily configurable and customizable for technical writing.",
            'metadata': {'topic': 'web_development', 'framework': 'Next.js', 'css': 'Tailwind CSS'},
            'id': 'tailwind-nextjs-blog-template'
        },
        {
            'text': "Personal portfolio website built with Gatsby, showcasing projects and skills.",
            'metadata': {'topic': 'portfolio', 'framework': 'Gatsby', 'language': 'JavaScript'},
            'id': 'personal-portfolio'
        },
        {
            'text': "A video conferencing application demonstrating real-time communication features.",
            'metadata': {'topic': 'real_time_communication', 'language': 'JavaScript'},
            'id': 'video-conferencing-app'
        },
        {
            'text': "Demo Angular project utilizing NgRx for state management.",
            'metadata': {'topic': 'web_development', 'framework': 'Angular', 'state_management': 'NgRx'},
            'id': 'angular-with-ngrx'
        },
        {
            'text': "Personal portfolio website built using React.js and TypeScript.",
            'metadata': {'topic': 'portfolio', 'framework': 'React', 'language': 'TypeScript'},
            'id': 'personal-portfolio-react'
        },
        {
            'text': "React web application for typing practice, enhancing typing speed and accuracy.",
            'metadata': {'topic': 'education', 'framework': 'React', 'language': 'JavaScript'},
            'id': 'typingpracticecenter.com'
        },
        {
            'text': "A simple to-do list application built with React and Redux for state management.",
            'metadata': {'topic': 'productivity', 'framework': 'React', 'state_management': 'Redux'},
            'id': 'todo-list-app'
        },
        {
            'text': "Weather forecasting application using OpenWeatherMap API, built with React.",
            'metadata': {'topic': 'weather', 'framework': 'React', 'api': 'OpenWeatherMap'},
            'id': 'weather-app'
        },
        {
            'text': "Chat application demonstrating real-time messaging using Socket.io and Node.js.",
            'metadata': {'topic': 'real_time_communication', 'framework': 'Node.js', 'library': 'Socket.io'},
            'id': 'chat-app'
        },
        {
            'text': "E-commerce website built with MERN stack, featuring product listings and shopping cart functionality.",
            'metadata': {'topic': 'e_commerce', 'stack': 'MERN', 'language': 'JavaScript'},
            'id': 'ecommerce-website'
        }
    ]

    # Add documents only if collection is empty
    info = rag.get_collection_info()
    if info['document_count'] != 0:
        print(f"{Fore.YELLOW}Clearing existing collection...{Style.RESET_ALL}")
        rag.chroma_client.delete_collection(name=rag.collection_name)
        rag.collection = rag.chroma_client.create_collection(name=rag.collection_name)
        info = rag.get_collection_info()
    print(f"{Fore.GREEN}Adding sample documents...{Style.RESET_ALL}")
    rag.add_documents(sample_docs)

    # Interactive query loop
    print(f"\n{Fore.CYAN}=== RAG System Ready ==={Style.RESET_ALL}")
    print(f"{Fore.CYAN}Ask questions about the documents (type 'quit' to exit){Style.RESET_ALL}")

    while True:
        question = input(f"\n{Fore.MAGENTA}Your question: {Style.RESET_ALL}").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            try:
                result = rag.query(question)
                print(f"\n{Fore.GREEN}Answer: {result['answer']}{Style.RESET_ALL}")
                print(f"\n{Fore.YELLOW}Sources used: {len(result['sources'])}{Style.RESET_ALL}")
                for i, source in enumerate(result['sources']):
                    print(f"  {Fore.BLUE}{i+1}. {source['id']} (distance: {source['distance']:.3f}){Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

