# in app/vector_store.py

import faiss
import numpy as np
import pickle
from abc import ABC, abstractmethod

# --- Vertex AI Client ---
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint

# This is the generic interface
class VectorStore(ABC):
    @abstractmethod
    def find_neighbors(self, query_embedding: list, num_neighbors: int) -> list[str]:
        pass

# --- FAISS Implementation (for local development) ---
class FAISSVectorStore(VectorStore):
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.id_map = {}
        self.text_map = {}

    def load(self):
        """Loads the index and metadata from disk."""
        print(f"INFO: Loading local FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.id_map = metadata['id_map']
            self.text_map = metadata['text_map']
        print("INFO: Local FAISS index loaded successfully.")

    def find_neighbors(self, query_embedding: list, num_neighbors: int) -> list[str]:
        if self.index is None:
            raise RuntimeError("Index is not loaded. Please call load() first.")
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, num_neighbors)
        
        return [self.id_map[i] for i in indices[0]]

# --- Vertex AI Implementation (for cloud) ---
class VertexAIVectorStore(VectorStore):
    def __init__(self, endpoint_id: str, deployed_index_id: str):
        print(f"INFO: Connecting to Vertex AI Endpoint {endpoint_id}...")
        self.endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
        self.deployed_index_id = deployed_index_id
        print("INFO: Connection to Vertex AI successful.")

    def find_neighbors(self, query_embedding: list, num_neighbors: int) -> list[str]:
        neighbor_results = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=num_neighbors
        )
        return [neighbor.id for neighbor in neighbor_results[0]]