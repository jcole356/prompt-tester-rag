# rag_service_client.py
import requests

class RAGServiceClient:
    def __init__(self, base_url="http://localhost:8000"):
        """Initialize the client with the RAG service base URL."""
        self.base_url = base_url

    def get_embedding(self, text):
        """Generate embedding for a given text."""
        response = requests.post(f"{self.base_url}/embed/", json={"text": text})
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Failed to get embedding: {response.text}")

    def store_embedding(self, text):
        """Store the embedding of the given text in the RAG service."""
        response = requests.post(f"{self.base_url}/store/", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to store embedding: {response.text}")

    def retrieve_similar(self, query):
        """Retrieve items similar to the given query text."""
        response = requests.post(f"{self.base_url}/retrieve/", json={"query": query})
        if response.status_code == 200:
            return response.json()["results"]
        else:
            raise Exception(f"Failed to retrieve similar items: {response.text}")

    def embed_all_documents(self):
        """Call the endpoint to embed all documents."""
        response = requests.post(f"{self.base_url}/embed_all_documents/")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to embed all documents: {response.text}")

    def clear_embeddings(self):
        """Call the endpoint to clear all embeddings."""
        response = requests.post(f"{self.base_url}/clear_embeddings/")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to clear embeddings: {response.text}")

    def check_embeddings(self):
        """Call the endpoint to check the number of embeddings."""
        response = requests.get(f"{self.base_url}/check_embeddings/")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to check embeddings: {response.text}")

    def enhanced_retrieve(self, query, api_key):
        """Retrieve enhanced prompt and related documents using RAG pipeline."""
        response = requests.post(
            f"{self.base_url}/enhanced_retrieve/",
            json={"query": query, "api_key": api_key}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to perform enhanced retrieval: {response.text}")
        
