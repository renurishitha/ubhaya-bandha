r"""FastAPI backend for Ubhaya-Bandha (BSPAI).

Run locally from the project root with:
    python -m uvicorn app.main:app --reload --port 8000
"""
import csv
import io
from pathlib import Path
import uuid
import zipfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "chroma_store"
CHROMA_ZIP = ROOT_DIR / "chroma_store.zip"

if not CHROMA_DIR.exists() and CHROMA_ZIP.exists():
  with zipfile.ZipFile(CHROMA_ZIP, "r") as zip_ref:
    zip_ref.extractall(ROOT_DIR)

try:
  from app.bispec_features import GENE_ALIASES
  from app.chatbot import ChatSession
  from app.config import (
      H5AD_PATH,
      MOESM3_PATH,
      MOESM5_PATH,
      PAPER_PDF_PATH,
      TRAINED_MODEL_PATH,
  )
  from app.model_service import ModelService
  from app.rag_report import generate_explanation
  from app.structured_retrieval import StructuredStore
  from app.vector_store import PaperVectorStore
except ImportError:
  from bispec_features import GENE_ALIASES
  from chatbot import ChatSession
  from config import (
      H5AD_PATH,
      MOESM3_PATH,
      MOESM5_PATH,
      PAPER_PDF_PATH,
      TRAINED_MODEL_PATH,
  )
  from model_service import ModelService
  from rag_report import generate_explanation
  from structured_retrieval import StructuredStore
  from vector_store import PaperVectorStore

app = FastAPI(title="Ubhaya-Bandha (BSPAI) Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for static_dir in [ROOT_DIR / "frontend", ROOT_DIR / "public"]:
  if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    break

store = StructuredStore(MOESM3_PATH, MOESM5_PATH)
vectors = PaperVectorStore(PAPER_PDF_PATH)
model = ModelService(TRAINED_MODEL_PATH, H5AD_PATH)

_sessions: dict[str, ChatSession] = {}


class PairRequest(BaseModel):
  gene_a: str
  gene_b: str


class ChatRequest(BaseModel):
  session_id: str | None = None
  message: str


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/predict")
@app.post("/api/predict")
def predict(req: PairRequest):
  ctx = store.lookup(req.gene_a, req.gene_b)
  model_out = model.predict(req.gene_a, req.gene_b, ctx)
  return {**ctx, **model_out}


@app.post("/report")
@app.post("/api/report")
def report(req: PairRequest):
  ctx = store.lookup(req.gene_a, req.gene_b)
  model_out = model.predict(req.gene_a, req.gene_b, ctx)
  chunks = vectors.retrieve(f"{req.gene_a} {req.gene_b} target combination")
  explanation = generate_explanation(ctx, model_out, chunks)
  return {
      **ctx,
      **model_out,
      "explanation": explanation,
      "n_chunks_used": len(chunks),
  }


@app.post("/chat")
@app.post("/api/chat")
def chat(req: ChatRequest):
  session_id = req.session_id or str(uuid.uuid4())
  if session_id not in _sessions:
    _sessions[session_id] = ChatSession(store, vectors, model)
  reply = _sessions[session_id].ask(req.message)
  return {"session_id": session_id, "reply": reply}


@app.get("/api/genes")
def get_genes():
  all_symbols = sorted(list(store.known_genes | set(GENE_ALIASES.keys())))
  return {
      "genes": [{"symbol": sym} for sym in all_symbols],
      "allGeneSymbols": all_symbols,
      "totalCandidatePairs": len(all_symbols) * (len(all_symbols) - 1) // 2,
  }


@app.get("/api/metrics")
def get_metrics():
  gene_count = len(store.known_genes)
  importances = [
      {"feature": col, "importance": round(float(imp), 4)}
      for col, imp in zip(
          model.feature_cols, model.model.feature_importances_
      )
  ]
  importances.sort(key=lambda x: x["importance"], reverse=True)
  return {
      "datasetId": "GSE131907_MOESM3_3DGNN",
      "geneCount": gene_count,
      "totalCandidatePairs": len(store.pair_table),
      "seedVersion": "42-xgb",
      "modelVersion": "XGBoost-scRNAseq-3DGNN-v1.0",
      "topFeatureImportances": importances[:6],
  }


@app.get("/api/leaderboard")
def get_leaderboard():
  rows = []
  if not store.benchmark_table.empty:
    for rank, (pair, row) in enumerate(store.benchmark_table.iterrows(), 1):
      g1, g2 = pair.split("+")
      rows.append({
          "rank": rank,
          "gene_a": g1,
          "gene_b": g2,
          "pair": pair,
          "predict_score": round(float(row.get("predict score", 0.0)), 4),
          "clinical_stage": int(row.get("clinical stage", 0)),
          "safe_avg": round(float(row.get("safe_avg", 0.0)), 5),
          "double_ratio_max": round(float(row.get("double_ratio_max", 0.0)), 4),
      })
  return {"items": rows}


@app.get("/api/export/leaderboard.csv")
def export_leaderboard():
  output = io.StringIO()
  writer = csv.writer(output)
  writer.writerow(
      ["Rank", "Gene A", "Gene B", "Predict Score", "Clinical Stage", "Safe Avg"]
  )
  if not store.benchmark_table.empty:
    for rank, (pair, row) in enumerate(store.benchmark_table.iterrows(), 1):
      g1, g2 = pair.split("+")
      writer.writerow([
          rank,
          g1,
          g2,
          round(float(row.get("predict score", 0.0)), 4),
          row.get("clinical stage", 0),
          round(float(row.get("safe_avg", 0.0)), 5),
      ])
  output.seek(0)
  return StreamingResponse(
      iter([output.getvalue()]),
      media_type="text/csv",
      headers={
          "Content-Disposition": "attachment; filename=bispec_leaderboard.csv"
      },
  )


# Serve built React frontend (dist/) in cloud/Docker deployment, or fallback locally
dist_dir = ROOT_DIR / "dist"
if dist_dir.exists():
  app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="spa")
else:

  @app.get("/", response_class=HTMLResponse)
  def serve_fallback():
    for candidate in [
        ROOT_DIR / "frontend" / "index.html",
        ROOT_DIR / "index.html",
    ]:
      if candidate.exists():
        return FileResponse(candidate)
    return HTMLResponse(
        "<h2>Backend API Running on :8000. Open React Frontend on :3000.</h2>"
    )