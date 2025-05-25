import chromadb
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, config):
        self.collection_name = "rag_documents"
        self.chroma_client = chromadb.PersistentClient(path=config.get('CHROMA_DB_PATH', './chroma_db'))
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)

    def add_documents(self, documents: List[Dict[str, Any]], llm, embedding_model):
        texts, embeddings, metadatas, ids = [], [], [], []
        for i, doc in enumerate(documents):
            text = doc['text']
            metadata = doc.get('metadata', {})
            doc_id = doc.get('id', f"doc_{i}")
            embedding = llm.create_embedding(text, embedding_model)
            texts.append(text)
            embeddings.append(embedding)
            metadatas.append(metadata)
            ids.append(doc_id)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int, llm, embedding_model) -> List[Dict[str, Any]]:
        query_embedding = llm.create_embedding(query, embedding_model)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            })
        return search_results

    def get_info(self) -> Dict[str, Any]:
        return {
            'collection_name': self.collection_name,
            'document_count': self.collection.count()
        }

    def clear(self):
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.create_collection(name=self.collection_name)

