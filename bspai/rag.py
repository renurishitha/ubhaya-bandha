"""Provider-agnostic RAG report generator with validated GEMINI_MODEL and deterministic fallback."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

VALID_GEMINI_MODELS = (
    "gemini-4-hydrogen",
    "gemini-4-argon",
    "gemini-4-carbon",
    "gemini-3.1-flash-lite",
)

DEPRECATED_GEMINI_MODELS = {
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-2.0-flash-thinking",
}


def resolve_and_validate_gemini_model(raw_model: Optional[str] = None) -> str:
    candidate = (raw_model if raw_model is not None else os.getenv("GEMINI_MODEL", "gemini-4-hydrogen")).strip() or "gemini-4-hydrogen"
    if candidate in DEPRECATED_GEMINI_MODELS:
        raise ValueError(
            f'GEMINI_MODEL "{candidate}" is deprecated and unsupported. Valid models: {", ".join(VALID_GEMINI_MODELS)}.'
        )
    if candidate not in VALID_GEMINI_MODELS:
        raise ValueError(
            f'GEMINI_MODEL "{candidate}" is invalid. Valid models: {", ".join(VALID_GEMINI_MODELS)}.'
        )
    return candidate


def generate_rag_report(
    gene1: str,
    gene2: str,
    score: float,
    rank: int,
    total_pairs: int,
    mechanism_type: str,
    features: Dict[str, float],
    discretized: List[Dict[str, object]],
    dataset_id: str,
    seed: int,
    model_version: str,
    data_mode: str = "Preprocessed AnnData (.h5ad) + Clinical Pairs (MOESM3 / Table 1)",
    in_clinical_trial: bool = False,
    clinical_stage: Optional[int] = None,
    clinical_drugs: Optional[List[str]] = None,
    enable_web_search: bool = False,
) -> Dict[str, object]:
    validated_model = resolve_and_validate_gemini_model()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    percentile = (1.0 - (rank - 1) / max(1, total_pairs)) * 100.0
    now_iso = datetime.now(timezone.utc).isoformat()

    if api_key and api_key != "MY_GEMINI_API_KEY":
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=api_key)
            prompt = (
                f"Summarize BSPAI target pair {gene1} + {gene2}. "
                f"Score={score:.4f}, Rank=#{rank}/{total_pairs} ({percentile:.2f}%), "
                f"Mechanism={mechanism_type}, Discretized={discretized}, Features={features}. "
                "State that this is a research aid and never invent clinical trial results."
            )
            resp = client.models.generate_content(model=validated_model, contents=prompt)
            text = (resp.text or "").strip()
            if len(text) > 80:
                provenance = {
                    "dataset": dataset_id,
                    "seed": seed,
                    "model_version": model_version,
                    "data_mode": data_mode,
                    "generation_mode": "gemini",
                    "gemini_model": validated_model,
                    "web_search_enabled": enable_web_search,
                    "generated_at": now_iso,
                }
                header = (
                    f"> **Research Aid Notice**: Automated BSPAI computational target-selection synthesis.\n"
                    f"> **Provenance Stamp**: Dataset `{dataset_id}` | Mode `{data_mode}` | Seed `{seed}` | "
                    f"Model `{model_version}` | Synthesizer `gemini ({validated_model})` | Generated `{now_iso}`"
                )
                return {"report_markdown": f"{header}\n\n{text}", "provenance": provenance}
        except Exception:
            pass

    provenance = {
        "dataset": dataset_id,
        "seed": seed,
        "model_version": model_version,
        "data_mode": data_mode,
        "generation_mode": "deterministic_template",
        "gemini_model": validated_model,
        "web_search_enabled": False,
        "generated_at": now_iso,
    }
    clinical_text = (
        f"Stage {clinical_stage} ({', '.join(clinical_drugs or [])})"
        if in_clinical_trial
        else "No clinical bispecific recorded in seeded benchmark table"
    )
    bin_bullets = "\n".join(
        f"- **{d['feature']}**: `{float(d['value']):.4f}` -> **{d['bin_label']}**"
        for d in discretized
    )
    markdown = (
        f"> **Research Aid Notice**: Automated BSPAI computational target-selection synthesis. Not clinical advice.\n"
        f"> **Provenance Stamp**: Dataset `{dataset_id}` | Mode `{data_mode}` | Seed `{seed}` | "
        f"Model `{model_version}` | Synthesizer `deterministic_template` | Generated `{now_iso}`\n\n"
        f"### 1. Target & Clinical Background\n"
        f"- Pair **{gene1} + {gene2}** (`{mechanism_type}`): {clinical_text}.\n\n"
        f"### 2. Quantitative ML Feature Analysis\n"
        f"- **Model Score & Rank**: `{score:.4f}` (Rank **#{rank} / {total_pairs}**, top **{percentile:.2f}%**).\n"
        f"{bin_bullets}\n\n"
        f"### 3. Conclusion, Risks & Next Steps\n"
        f"- Validate co-expression at surface protein level and benchmark affinity/safety windows in primary assays.\n"
    )
    return {"report_markdown": markdown, "provenance": provenance}
