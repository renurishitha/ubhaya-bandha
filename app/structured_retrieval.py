r"""Structured retrieval layer for MOESM3 and MOESM5."""
import os
import numpy as np
import pandas as pd

try:
  from app.graph_features import build_gene_graph, pair_graph_features
except ImportError:
  from graph_features import build_gene_graph, pair_graph_features


def _split_sort(target_str) -> str | float:
  parts = [t.strip().upper() for t in str(target_str).split("+")]
  if len(parts) == 2:
    parts.sort()
    return "+".join(parts)
  return np.nan


def load_moesm3_pairs(path: str) -> pd.DataFrame:
  xl = pd.ExcelFile(path)
  sheet = "Results" if "Results" in xl.sheet_names else xl.sheet_names[0]
  raw = pd.read_excel(path, sheet_name=sheet)

  if "Target(Gene Name)" not in [str(c).strip() for c in raw.columns]:
    for hdr in range(4):
      temp = pd.read_excel(path, sheet_name=sheet, header=hdr)
      cols = [str(c).strip() for c in temp.columns]
      if "Target(Gene Name)" in cols or ("Gene1" in cols and "Gene2" in cols):
        raw = temp
        raw.columns = cols
        break

  if "Target(Gene Name)" in raw.columns:
    df = raw[
        raw["Target(Gene Name)"].astype(str).str.contains(r"\+", na=False)
    ].copy()
    df["pair"] = df["Target(Gene Name)"].apply(_split_sort)
    df = df.dropna(subset=["pair"])
    df["label"] = df["Drug Highest Phase"].apply(
        lambda x: 1 if pd.notna(x) and "Approved" in str(x) else 0
    )
    return df
  else:
    df = raw.dropna(subset=["Gene1", "Gene2"]).copy()
    df["pair"] = df.apply(
        lambda r: "+".join(
            sorted([
                str(r["Gene1"]).strip().upper(),
                str(r["Gene2"]).strip().upper(),
            ])
        ),
        axis=1,
    )
    df["label"] = (
        pd.to_numeric(df.get("clinical stage", 0), errors="coerce").fillna(0)
        > 0
    ).astype(int)
    return df


def fit_beta_prior(pair_rates: pd.Series) -> tuple[float, float]:
  p_bar = pair_rates.mean()
  var = pair_rates.var()
  if pd.isna(var) or var == 0:
    var = 1e-6
  common = max(p_bar * (1 - p_bar) / var - 1, 0.01)
  alpha = max(p_bar * common, 1e-3)
  beta = max((1 - p_bar) * common, 1e-3)
  return alpha, beta


class StructuredStore:
  def __init__(self, moesm3_path: str, moesm5_path: str):
    df3 = load_moesm3_pairs(moesm3_path)

    grp = df3.groupby("pair")["label"].agg(successes="sum", n="count")
    grp["raw_rate"] = grp["successes"] / grp["n"]
    alpha, beta = fit_beta_prior(grp["raw_rate"])
    grp["eb_score"] = (grp["successes"] + alpha) / (grp["n"] + alpha + beta)
    self.pair_table = grp
    self.alpha, self.beta = alpha, beta

    pairs_list = [tuple(p.split("+")) for p in grp.index]
    self.graph = build_gene_graph(pairs_list)
    self.known_genes = set(self.graph.nodes)

    bench_path = moesm5_path if os.path.exists(moesm5_path) else moesm3_path
    df5 = pd.read_excel(bench_path, header=1)
    if "Gene1" not in df5.columns:
      df5 = pd.read_excel(bench_path, header=0)
    if "Gene1" in df5.columns and "Gene2" in df5.columns:
      df5 = df5.dropna(subset=["Gene1", "Gene2"]).copy()
      df5["pair"] = df5.apply(
          lambda r: "+".join(
              sorted([
                  str(r["Gene1"]).strip().upper(),
                  str(r["Gene2"]).strip().upper(),
              ])
          ),
          axis=1,
      )
      self.benchmark_table = df5.drop_duplicates(subset=["pair"]).set_index(
          "pair"
      )
    else:
      self.benchmark_table = pd.DataFrame()

  def lookup(self, gene_a: str, gene_b: str) -> dict:
    pair = "+".join(sorted([gene_a.strip().upper(), gene_b.strip().upper()]))
    result = {"pair": pair}

    if pair in self.pair_table.index:
      row = self.pair_table.loc[pair]
      result["eb_score"] = round(float(row.eb_score), 4)
      result["n_prior_drugs"] = int(row.n)
      result["prior_successes"] = int(row.successes)
    else:
      result["eb_score"] = round(self.alpha / (self.alpha + self.beta), 4)
      result["n_prior_drugs"] = 0
      result["prior_successes"] = 0

    result.update(pair_graph_features(self.graph, gene_a, gene_b))

    if not self.benchmark_table.empty and pair in self.benchmark_table.index:
      bench = self.benchmark_table.loc[pair]
      result["benchmark_predict_score"] = float(
          bench.get("predict score", np.nan)
      )
      result["benchmark_clinical_stage"] = bench.get("clinical stage", None)
    else:
      result["benchmark_predict_score"] = None
      result["benchmark_clinical_stage"] = None

    return result