r"""Loads pre-computed 3D-GNN structural embeddings (gnn_3d_embeddings.pt)."""
import os
import numpy as np
import torch


class StructuralEmbeddingStore:
  def __init__(self, emb_path: str | None = None, embedding_dim: int = 128):
    self.embedding_dim = embedding_dim
    self.embeddings: dict[str, np.ndarray] = {}
    if emb_path and os.path.exists(emb_path):
      self.load_from_pt(emb_path)

  def load_from_pt(self, emb_path: str):
    raw_dict = torch.load(emb_path, map_location="cpu", weights_only=False)
    self.embeddings = {
        str(k).strip().upper(): np.asarray(v, dtype=np.float32).ravel()
        for k, v in raw_dict.items()
    }
    if self.embeddings:
      first_vec = next(iter(self.embeddings.values()))
      self.embedding_dim = len(first_vec)
    print(
        f"[3D-GNN] Loaded {len(self.embeddings)} structural embeddings"
        f" (dim={self.embedding_dim})"
    )

  def get_embedding(self, gene_symbol: str) -> np.ndarray:
    clean_gene = str(gene_symbol).strip().upper()
    if clean_gene in self.embeddings:
      return self.embeddings[clean_gene]
    return np.zeros(self.embedding_dim, dtype=np.float32)

  def get_pair_metrics(self, gene_a: str, gene_b: str) -> dict:
    ga, gb = str(gene_a).strip().upper(), str(gene_b).strip().upper()
    if ga in self.embeddings and gb in self.embeddings:
      za, zb = self.embeddings[ga], self.embeddings[gb]
      dist = float(np.linalg.norm(za - zb))
      denom = (np.linalg.norm(za) * np.linalg.norm(zb)) + 1e-8
      cos = float(np.dot(za, zb) / denom)
      return {"gnn_3d_dist": round(dist, 4), "gnn_3d_cos": round(cos, 4)}
    return {"gnn_3d_dist": 1.0, "gnn_3d_cos": 0.0}