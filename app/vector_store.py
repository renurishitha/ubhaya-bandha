r"""Vector DB layer over the research paper PDF ONLY."""
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

try:
  from app.config import (
      CHROMA_COLLECTION_NAME,
      CHROMA_PERSIST_DIR,
      CHUNK_OVERLAP_CHARS,
      CHUNK_SIZE_CHARS,
      SIMILARITY_THRESHOLD,
      TOP_K_CHUNKS,
  )
except ImportError:
  from config import (
      CHROMA_COLLECTION_NAME,
      CHROMA_PERSIST_DIR,
      CHUNK_OVERLAP_CHARS,
      CHUNK_SIZE_CHARS,
      SIMILARITY_THRESHOLD,
      TOP_K_CHUNKS,
  )


def extract_pdf_text(pdf_path: str) -> str:
  if not os.path.exists(pdf_path):
    return ""
  reader = PdfReader(pdf_path)
  return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(
    text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
  chunks = []
  start = 0
  while start < len(text):
    end = start + size
    chunks.append(text[start:end])
    start = end - overlap
  return [c.strip() for c in chunks if c.strip()]


class PaperVectorStore:
  def __init__(self, pdf_path: str, rebuild: bool = False):
    self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

    existing = [c.name for c in self.client.list_collections()]
    if rebuild and CHROMA_COLLECTION_NAME in existing:
      self.client.delete_collection(CHROMA_COLLECTION_NAME)
      existing.remove(CHROMA_COLLECTION_NAME)

    if CHROMA_COLLECTION_NAME in existing:
      self.collection = self.client.get_collection(
          CHROMA_COLLECTION_NAME, embedding_function=self.embed_fn
      )
    else:
      self.collection = self.client.create_collection(
          CHROMA_COLLECTION_NAME, embedding_function=self.embed_fn
      )
      if os.path.exists(pdf_path):
        self._index_pdf(pdf_path)

  def _index_pdf(self, pdf_path: str):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    if chunks:
      self.collection.add(
          documents=chunks,
          ids=[f"chunk_{i}" for i in range(len(chunks))],
      )

  def retrieve(self, query: str, k: int = TOP_K_CHUNKS) -> list[str]:
    if self.collection.count() == 0:
      return []
    results = self.collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    kept = []
    for doc, dist in zip(docs, distances):
      if (1 - dist) >= SIMILARITY_THRESHOLD:
        kept.append(doc)
    return kept