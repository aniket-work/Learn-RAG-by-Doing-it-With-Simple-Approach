import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'CHROMA_DB_PATH': os.getenv('CHROMA_DB_PATH', './chroma_db'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'CHAT_MODEL': os.getenv('CHAT_MODEL', 'gpt-4o-mini')
    }

