r"""Conversational layer on top of the RAG report generator."""
import re

try:
  from app.bispec_features import GENE_ALIASES
  from app.model_service import ModelService
  from app.rag_report import generate_explanation
  from app.structured_retrieval import StructuredStore
  from app.vector_store import PaperVectorStore
except ImportError:
  from bispec_features import GENE_ALIASES
  from model_service import ModelService
  from rag_report import generate_explanation
  from structured_retrieval import StructuredStore
  from vector_store import PaperVectorStore

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{1,11}")
NOT_GENES = {
    "THE", "AND", "WHY", "HOW", "FOR", "WHAT", "DOES",
    "ARE", "WITH", "THIS", "THAT", "FROM", "ABOUT",
}


class ChatSession:
  def __init__(
      self,
      store: StructuredStore,
      vectors: PaperVectorStore,
      model: ModelService,
  ):
    self.store = store
    self.vectors = vectors
    self.model = model
    self.history: list[dict] = []
    self.last_pair: tuple[str, str] | None = None
    self.curated_vocab = set(store.known_genes) | set(GENE_ALIASES)
    self.model_vocab = set(model.var_upper_map)

  def _extract_pair(self, message: str) -> tuple[str, str] | None:
    found: list[str] = []
    for tok in TOKEN_PATTERN.findall(message):
      up = tok.upper()
      if up in NOT_GENES or up in found:
        continue
      typed_as_symbol = tok.isupper() and up in self.model_vocab
      if up in self.curated_vocab or typed_as_symbol:
        found.append(up)
    return (found[0], found[1]) if len(found) >= 2 else None

  def _explain_pair(self, gene_a: str, gene_b: str, message: str) -> str:
    structured_ctx = self.store.lookup(gene_a, gene_b)
    if structured_ctx.get("coverage") == "neither_known":
      return (
          f"I couldn't find {gene_a} or {gene_b} in the prior clinical pair data, "
          "so I can't score this pair reliably."
      )
    model_out = self.model.predict(gene_a, gene_b, structured_ctx)
    chunks = self.vectors.retrieve(message)
    return generate_explanation(structured_ctx, model_out, chunks)

  def ask(self, message: str) -> str:
    pair = self._extract_pair(message)
    if pair:
      self.last_pair = pair
      reply = self._explain_pair(pair[0], pair[1], message)
    elif self.last_pair:
      reply = self._explain_pair(self.last_pair[0], self.last_pair[1], message)
    else:
      chunks = self.vectors.retrieve(message)
      if not chunks:
        reply = (
            "I don't have paper context relevant to that question, and no gene "
            "pair has been specified yet. Ask about a specific pair (e.g. 'BCMA and CD3')."
        )
      else:
        reply = "\n".join(chunks)

    self.history.append({"role": "user", "content": message})
    self.history.append({"role": "assistant", "content": reply})
    return reply