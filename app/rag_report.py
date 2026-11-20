r"""Generation layer: turns model score + 3D-GNN metrics + paper chunks into a report."""
import os

try:
  from app.config import ANTHROPIC_MODEL, LLM_PROVIDER, OPENAI_MODEL
except ImportError:
  from config import ANTHROPIC_MODEL, LLM_PROVIDER, OPENAI_MODEL

SYSTEM_PROMPT = """You are assisting a researcher evaluating a candidate bispecific \
antibody target pair. You will be given:
1. A machine-learning viability score, computed from REAL single-cell RNA-seq \
expression data (safety differential, cell-subtype co-expression, correlation) \
and 3D-GNN extracellular protein structure embeddings (3D distance & cosine similarity).
2. Supplementary context from prior clinical drug-pair history.
3. (Optionally) excerpts retrieved from a research paper about the underlying method.

Rules you MUST follow:
- Only make pair-specific factual claims using the numbers given in (1) and (2).
- Only make general-principle claims using the text given in (3).
- If (3) is empty, explicitly state no paper-specific context was found, and explain using (1) and (2).
- Do NOT invent biological facts from pretrained knowledge.
"""


def _format_context(structured_ctx: dict, model_out: dict, chunks: list[str]) -> str:
  lines = [
      f"Gene pair: {structured_ctx['pair']}",
      "--- Primary model output (live scRNA-seq + 3D-GNN features) ---",
      f"Model predicted viability probability: {model_out['model_score']}",
      f"Tumor-vs-normal safety differential (safe_avg): {model_out['safe_avg']}",
      f"Max double-positive expression ratio: {model_out['double_ratio_max']}",
      f"Max summed single-gene expression ratio: {model_out['sum_single_ratio_max']}",
      f"3D-GNN structural distance (gnn_3d_dist): {model_out.get('gnn_3d_dist', 1.0)}",
      f"3D-GNN cosine similarity (gnn_3d_cos): {model_out.get('gnn_3d_cos', 0.0)}",
      "--- Supplementary context (prior clinical-drug pair history) ---",
      (
          f"Empirical-Bayes prior score ({structured_ctx['n_prior_drugs']} prior"
          f" drugs, {structured_ctx['prior_successes']} approved):"
          f" {structured_ctx['eb_score']}"
      ),
      (
          f"Coverage: {structured_ctx['coverage']}, degree_a={structured_ctx['degree_a']}, "
          f"degree_b={structured_ctx['degree_b']}, "
          f"common_neighbors={structured_ctx['common_neighbors']}, "
          f"jaccard={structured_ctx['jaccard']:.4f}"
      ),
  ]
  if structured_ctx.get("benchmark_predict_score") is not None:
    lines.append(
        "External benchmark:"
        f" predict_score={structured_ctx['benchmark_predict_score']},"
        f" clinical_stage={structured_ctx['benchmark_clinical_stage']}"
    )
  if chunks:
    lines.append("\nRelevant excerpts from the source paper:")
    for c in chunks:
      lines.append(f"- {c[:500]}")
  else:
    lines.append("\nNo sufficiently relevant excerpts were found in the source paper.")
  return "\n".join(lines)


def generate_explanation(
    structured_ctx: dict, model_out: dict, chunks: list[str]
) -> str:
  context_block = _format_context(structured_ctx, model_out, chunks)
  user_prompt = (
      f"{context_block}\n\n"
      "Write a short scientific explanation (one to two paragraphs) of this "
      "prediction for a researcher, following the rules you were given."
  )

  if LLM_PROVIDER == "openai":
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
      return f"[Local Summary — Set OPENAI_API_KEY for LLM prose]\n{context_block}"
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        ),
    )
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "openrouter/free"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

  elif LLM_PROVIDER == "anthropic":
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text

  else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")