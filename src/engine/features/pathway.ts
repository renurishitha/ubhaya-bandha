import type { PathwayDatabase } from '../types.ts';

export interface PathwayIndex {
  keggSets: Map<string, Set<string>>;
  goSets: Map<string, Set<string>>;
}

export function buildPathwayIndex(db: PathwayDatabase): PathwayIndex {
  const keggSets = new Map<string, Set<string>>();
  for (const [pw, genes] of Object.entries(db.kegg)) {
    for (const g of genes) {
      let s = keggSets.get(g);
      if (!s) {
        s = new Set<string>();
        keggSets.set(g, s);
      }
      s.add(pw);
    }
  }

  const goSets = new Map<string, Set<string>>();
  for (const [term, genes] of Object.entries(db.go)) {
    for (const g of genes) {
      let s = goSets.get(g);
      if (!s) {
        s = new Set<string>();
        goSets.set(g, s);
      }
      s.add(term);
    }
  }

  return { keggSets, goSets };
}

function countIntersection(a?: Set<string>, b?: Set<string>): number {
  if (!a || !b || a.size === 0 || b.size === 0) return 0;
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let count = 0;
  for (const item of small) {
    if (large.has(item)) count++;
  }
  return count;
}

export interface MechanismFeatures {
  kegg_cooccur: number;
  go_cooccur: number;
  pathway_cooccur: number;
  same_pathway: number;
}

export function computeMechanismFeatures(
  gene1: string,
  gene2: string,
  index: PathwayIndex
): MechanismFeatures {
  const kegg_cooccur = countIntersection(
    index.keggSets.get(gene1),
    index.keggSets.get(gene2)
  );
  const go_cooccur = countIntersection(
    index.goSets.get(gene1),
    index.goSets.get(gene2)
  );
  const pathway_cooccur = kegg_cooccur + go_cooccur;
  const same_pathway = pathway_cooccur > 0 ? 1 : 0;
  return { kegg_cooccur, go_cooccur, pathway_cooccur, same_pathway };
}
