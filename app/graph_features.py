r"""Bipartite gene-pair graph + link-prediction features."""
import networkx as nx


def build_gene_graph(pairs: list[tuple[str, str]]) -> nx.Graph:
  G = nx.Graph()
  for a, b in pairs:
    G.add_edge(a, b)
  return G


def pair_graph_features(G: nx.Graph, gene_a: str, gene_b: str) -> dict:
  a, b = gene_a.upper().strip(), gene_b.upper().strip()
  a_known = G.has_node(a)
  b_known = G.has_node(b)

  features = {
      "degree_a": G.degree(a) if a_known else 0,
      "degree_b": G.degree(b) if b_known else 0,
      "common_neighbors": 0,
      "jaccard": 0.0,
      "adamic_adar": 0.0,
      "preferential_attachment": 0,
      "coverage": (
          "both_known"
          if (a_known and b_known)
          else ("one_known" if (a_known or b_known) else "neither_known")
      ),
  }

  if a_known and b_known:
    try:
      jac = list(nx.jaccard_coefficient(G, [(a, b)]))
      features["jaccard"] = jac[0][2] if jac else 0.0
    except Exception:
      pass
    try:
      aa = list(nx.adamic_adar_index(G, [(a, b)]))
      features["adamic_adar"] = aa[0][2] if aa else 0.0
    except Exception:
      pass
    try:
      pa = list(nx.preferential_attachment(G, [(a, b)]))
      features["preferential_attachment"] = pa[0][2] if pa else 0
    except Exception:
      pass
    features["common_neighbors"] = len(list(nx.common_neighbors(G, a, b)))

  return features