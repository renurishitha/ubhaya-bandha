"""
Loads pre-computed structural or GNN embeddings for genes, 
providing a vector representation for each target.
"""
import numpy as np

class StructuralEmbeddingStore:
    def __init__(self, embedding_dim: int = 1280):
        self.embedding_dim = embedding_dim
        self.embeddings = {}

    def load_embeddings(self, embedding_dict: dict[str, np.ndarray]):
        """Load pre-computed gene embeddings."""
        self.embeddings = {str(k).upper(): v for k, v in embedding_dict.items()}

    def get_embedding(self, gene_symbol: str) -> np.ndarray:
        """Returns the 3D structural embedding vector for a given gene symbol."""
        clean_gene = str(gene_symbol).strip().upper()
        if clean_gene in self.embeddings:
            return self.embeddings[clean_gene]
        # Fallback to zero vector if embedding is missing
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def get_pair_embedding(self, gene_a: str, gene_b: str) -> np.ndarray:
        """Combines two gene embeddings to represent the pair (e.g., concatenation)."""
        emb_a = self.get_embedding(gene_a)
        emb_b = self.get_embedding(gene_b)
        return np.concatenate([emb_a, emb_b])