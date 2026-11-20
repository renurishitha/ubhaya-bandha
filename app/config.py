r"""Central configuration for Ubhaya-Bandha (BSPAI) Backend."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
# Checks ROOT_DIR/data first, then falls back to C:/Users/Rishi/Downloads
DEFAULT_DATA_DIR = (
    ROOT_DIR / "data"
    if (ROOT_DIR / "data").exists()
    else Path(r"C:/Users/Rishi/Downloads")
)
DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))

TRAINED_MODEL_PATH = os.getenv(
    "MODEL_PATH", str(DATA_DIR / "bspai_model.joblib")
)
H5AD_PATH = os.getenv(
    "H5AD_PATH", str(DATA_DIR / "GSE131907_Lung_Cancer_preprocessed.h5ad")
)
GNN_EMB_PATH = os.getenv(
    "GNN_EMB_PATH", str(DATA_DIR / "gnn_3d_embeddings.pt")
)
MOESM3_PATH = os.getenv(
    "MOESM3_PATH", str(DATA_DIR / "432_2024_5740_MOESM3_ESM.xlsx")
)
MOESM5_PATH = os.getenv(
    "MOESM5_PATH", str(DATA_DIR / "432_2024_5740_MOESM5_ESM.xlsx")
)
PAPER_PDF_PATH = os.getenv(
    "PAPER_PDF_PATH", str(DATA_DIR / "BiSpec_Pairwise.pdf")
)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_DIR", str(ROOT_DIR / "chroma_store"))
CHROMA_COLLECTION_NAME = "bispec_paper"
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
SIMILARITY_THRESHOLD = 0.35
TOP_K_CHUNKS = 3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")