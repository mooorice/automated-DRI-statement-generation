"""Anchor generation (via LLM) and loading from files."""

import json
import logging
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .prompts import load_prompt_sections

logger = logging.getLogger("autoconcourse.anchors")


def load_anchors(path: str | Path) -> pd.DataFrame:
    """Load pre-generated anchors from CSV or JSON.

    Returns DataFrame with columns: category (int), text (str), language (str, optional).
    """
    path = Path(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported anchor file format: {path.suffix}")

    if "category" not in df.columns and "subcategory_id" in df.columns:
        df = df.rename(columns={"subcategory_id": "category"})

    logger.info(f"Loaded {len(df)} anchors from {path}")
    return df


def save_anchors(anchors_df: pd.DataFrame, path: str | Path) -> None:
    """Save anchors to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        anchors_df.to_csv(path, index=False)
    elif path.suffix == ".json":
        anchors_df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported anchor file format: {path.suffix}")

    logger.info(f"Saved {len(anchors_df)} anchors to {path}")


def validate_anchors(anchors_df: pd.DataFrame, subcategories: list[dict]) -> list[str]:
    """Validate anchors against subcategory definitions.

    Checks that every subcategory has at least one anchor and that all anchor
    category IDs correspond to valid subcategories.

    Returns a list of warning/error messages (empty if all valid).
    """
    errors = []
    valid_ids = {s["id"] for s in subcategories}
    anchor_cats = set(anchors_df["category"].unique())

    missing = valid_ids - anchor_cats
    if missing:
        names = {s["id"]: s["name"] for s in subcategories}
        for mid in sorted(missing):
            errors.append(f"Subcategory {mid} ({names.get(mid, '?')}) has no anchors")

    extra = anchor_cats - valid_ids
    if extra:
        errors.append(f"Anchors reference unknown subcategory IDs: {sorted(extra)}")

    if "text" in anchors_df.columns:
        empty = anchors_df["text"].str.strip().eq("").sum()
        if empty > 0:
            errors.append(f"{empty} anchor(s) have empty text")

    for msg in errors:
        logger.warning(f"Anchor validation: {msg}")

    if not errors:
        logger.info(f"Anchor validation passed: {len(anchors_df)} anchors for {len(subcategories)} subcategories")

    return errors


def generate_anchors(config: PipelineConfig, client=None) -> pd.DataFrame:
    """Generate anchor texts via LLM for each subcategory and language.

    Uses subcategories.json for prompts and optionally a context document (PDF).

    Args:
        config: Pipeline configuration
        client: OpenAI-compatible client (for OpenRouter). If None, creates one.

    Returns:
        DataFrame with columns: category, text, language
    """
    subcategories_path = config.resolve_path(config.anchors.subcategories_file)
    with open(subcategories_path, encoding="utf-8") as f:
        subcategories = json.load(f)

    if client is None:
        from .generation import create_llm_client
        client = create_llm_client(config)

    # Load context document if provided
    context_text = ""
    if config.anchors.context_document:
        context_path = config.resolve_path(config.anchors.context_document)
        context_text = _load_context_document(context_path)

    anchors = []
    languages = config.anchors.languages
    model = config.anchors.model

    sections = load_prompt_sections("anchor_generation")
    system_prompt = sections.get("system", "You are an expert journalist. Write factual, comprehensive summary paragraphs.")

    for subcat in subcategories:
        cat_id = subcat["id"]
        anchor_prompt = subcat.get("anchor_prompt", "")
        name = subcat["name"]

        for lang in languages:
            prompt = _build_anchor_prompt(name, anchor_prompt, lang, context_text)

            logger.info(f"Generating anchor: category={cat_id}, language={lang}")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            text = response.choices[0].message.content.strip()
            anchors.append({
                "category": cat_id,
                "text": text,
                "language": lang,
                "subcategory_name": name,
            })

    df = pd.DataFrame(anchors)
    logger.info(f"Generated {len(df)} anchors")
    return df


LANGUAGE_NAMES = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "en": "English",
}


def _build_anchor_prompt(
    subcategory_name: str,
    custom_prompt: str,
    language: str,
    context_text: str,
) -> str:
    """Build the user prompt for anchor generation from the template."""
    sections = load_prompt_sections("anchor_generation")
    user_template = sections.get("user", "")

    lang_name = LANGUAGE_NAMES.get(language, language)

    context_section = ""
    if context_text:
        context_section = f"\nReference text:\n\n{context_text[:8000]}\n"

    if custom_prompt:
        subcategory_name = f"{subcategory_name} ({custom_prompt})"

    return user_template.format(
        subcategory_name=subcategory_name,
        context_section=context_section,
        language_name=lang_name,
    )


def _load_context_document(path: Path) -> str:
    """Load text from a context document (PDF or text)."""
    if path.suffix == ".pdf":
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            logger.warning("pymupdf not installed; cannot read PDF context document. Install with: pip install pymupdf")
            return ""
    else:
        return path.read_text(encoding="utf-8")


def load_or_generate_anchors(config: PipelineConfig, client=None) -> pd.DataFrame:
    """Load anchors from file if available, otherwise generate them.

    Resolution order for from_file method:
      1. config.anchors.anchors_file (explicit path)
      2. cache/{topic_id}/anchors.csv

    For llm_generated method:
      1. cache/{topic_id}/anchors.csv (if exists, use cached)
      2. Generate via LLM and cache
    """
    cache_path = config.cache_dir / "anchors.csv"

    if config.anchors.method == "from_file":
        if config.anchors.anchors_file:
            anchors_path = config.resolve_path(config.anchors.anchors_file)
            if anchors_path.exists():
                logger.info(f"Loading anchors from {anchors_path}")
                return load_anchors(anchors_path)
        if cache_path.exists():
            logger.info(f"Loading cached anchors from {cache_path}")
            return load_anchors(cache_path)
        raise FileNotFoundError(
            f"Anchor method is 'from_file' but no anchors found at "
            f"{config.anchors.anchors_file or cache_path}. "
            "Provide an anchors file or set method to 'llm_generated'."
        )

    if cache_path.exists():
        logger.info(f"Loading cached anchors from {cache_path}")
        return load_anchors(cache_path)

    df = generate_anchors(config, client)
    save_anchors(df, cache_path)
    return df
