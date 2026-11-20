"""FastAPI smoke tests for local Python execution via pytest."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_genes_and_predict_smoke() -> None:
    res = client.get("/genes?q=CD274")
    assert res.status_code == 200
    body = res.json()
    assert body["geneCount"] >= 300
    assert body["totalCandidatePairs"] == (body["geneCount"] * (body["geneCount"] - 1)) // 2

    pred = client.post("/predict", json={"gene1": "CD274", "gene2": "CTLA4"})
    assert pred.status_code == 200
    pbody = pred.json()
    assert pbody["gene1"] == "CD274"
    assert pbody["gene2"] == "CTLA4"
    assert "score_safe" in pbody["features"]

    rep = client.post("/report", json={"gene1": "CD274", "gene2": "CTLA4"})
    assert rep.status_code == 200
    rbody = rep.json()
    assert "provenance" in rbody
    assert rbody["provenance"]["seed"] == 42
