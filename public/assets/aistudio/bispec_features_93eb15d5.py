"""
Feature engineering, integrating StructuralEmbeddingStore from graph_embeddings.py
to compute physics-aware GNN features at inference time.
"""
import numpy as np
from scipy.stats import pearsonr
from app.graph_embeddings import StructuralEmbeddingStore

GENE_ALIASES = {
    "CD20": "MS4A1",
    "PD-1": "PDCD1",
    "PD1": "PDCD1",
    "PD-L1": "CD274",
    "PDL1": "CD274",
    "CTLA-4": "CTLA4",
    "BCMA": "TNFRSF17",
    "CD3": "CD3E",
    "C-MET": "MET",
    "HER2": "ERBB2",
    "HER3": "ERBB3",
    "4-1BB": "TNFRSF9",
}

KEY_SUBTYPES = [
    "T:Treg",
    "T:Exhausted CD8+ T",
    "Myeloid:Macrophage",
    "Epithelial cells",
]

# Instantiate the structural embedding store globally
struct_store = StructuralEmbeddingStore(embedding_dim=1280)


def build_var_upper_map(adata) -> dict:
    return {name.upper(): name for name in adata.var_names}


def resolve_gene_symbol(gene_symbol, var_upper_map: dict):
    s = str(gene_symbol).strip().upper()
    s = GENE_ALIASES.get(s, s)
    return var_upper_map.get(s, None)


def extract_gene_vector(adata_sub, resolved_gene):
    if resolved_gene is not None and resolved_gene in adata_sub.var_names:
        vec = adata_sub[:, resolved_gene].X
        if hasattr(vec, "toarray"):
            vec = vec.toarray().flatten()
        return np.asarray(vec).flatten()
    return np.zeros(adata_sub.n_obs, dtype=np.float32)


def compute_pair_features(adata_obj, gene_a_str, gene_b_str, var_upper_map, threshold=0.1) -> dict:
    """Computes scRNA-seq expression features combined with GNN structural pair embeddings."""
    feats = {}
    ga = resolve_gene_symbol(gene_a_str, var_upper_map)
    gb = resolve_gene_symbol(gene_b_str, var_upper_map)

    t_mask = adata_obj.obs["Cell_type.refined"].isin(["tLung", "Tumor"])
    n_mask = adata_obj.obs["Cell_type.refined"].isin(["nLung", "Normal"])

    va_t = extract_gene_vector(adata_obj[t_mask], ga)
    va_n = extract_gene_vector(adata_obj[n_mask], ga)
    vb_t = extract_gene_vector(adata_obj[t_mask], gb)
    vb_n = extract_gene_vector(adata_obj[n_mask], gb)

    d_a = float(np.mean(va_t) - np.mean(va_n))
    d_b = float(np.mean(vb_t) - np.mean(vb_n))

    feats["safe_avg"] = (d_a * d_b) / (d_a + d_b) if (d_a + d_b) != 0 else 0.0

    double_ratios, single_ratios = [], []

    for subtype in KEY_SUBTYPES:
        clean_key = subtype.replace(":", "_").replace("+", "plus").replace(" ", "_")
        sub_mask = (
            adata_obj.obs["Cell_subtype"] == subtype
            if "Cell_subtype" in adata_obj.obs
            else np.zeros(adata_obj.n_obs, dtype=bool)
        )

        if np.sum(sub_mask) < 15:
            feats[f"corrcoef_{clean_key}"] = 0.0
            feats[f"double_ratio_{clean_key}"] = 0.0
            feats[f"sum_single_exp_{clean_key}"] = 0.0
            continue

        sub_adata = adata_obj[sub_mask]
        va = extract_gene_vector(sub_adata, ga)
        vb = extract_gene_vector(sub_adata, gb)

        if np.std(va) > 1e-5 and np.std(vb) > 1e-5:
            r, _ = pearsonr(va, vb)
            feats[f"corrcoef_{clean_key}"] = 0.0 if np.isnan(r) else float(r)
        else:
            feats[f"corrcoef_{clean_key}"] = 0.0

        pos_a = va > threshold
        pos_b = vb > threshold
        p_double = float(np.mean(pos_a & pos_b))
        p_single = float(np.mean(pos_a) + np.mean(pos_b))

        double_ratios.append(p_double)
        single_ratios.append(p_single)

        feats[f"double_ratio_{clean_key}"] = p_double
        feats[f"sum_single_exp_{clean_key}"] = float(np.mean(va) + np.mean(vb))

    feats["double_ratio_max"] = max(double_ratios) if double_ratios else 0.0
    feats["sum_single_ratio_max"] = max(single_ratios) if single_ratios else 0.0

    # --- Inject GNN Structural Pair Embeddings from graph_embeddings.py ---
    pair_emb = struct_store.get_pair_embedding(gene_a_str, gene_b_str)
    for i, val in enumerate(pair_emb):
        feats[f"struct_emb_{i}"] = float(val)

    return feats