"""Anchor generation (via LLM) and loading from files."""

import json
import logging
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .keywords import _expand_context_folder, _format_context, _load_context_document
from .prompts import load_prompt_sections

logger = logging.getLogger("autoconcourse.anchors")


def resolve_anchors_path(config: PipelineConfig) -> Path:
    """Return the canonical on-disk path for anchor artifacts.

    If `anchors.anchors_file` is configured, use it for both loading and saving.
    Otherwise, fall back to the project-local artifacts location.
    """
    if config.anchors.anchors_file:
        return config.resolve_path(config.anchors.anchors_file)
    return config.anchors_cache_path


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


def generate_anchors(
    config: PipelineConfig,
    client=None,
    only_pairs: set[tuple[int, str]] | None = None,
) -> pd.DataFrame:
    """Generate anchor texts via LLM for each subcategory and language.

    Uses subcategories.json for prompts and optionally a context document (PDF).

    Args:
        config: Pipeline configuration.
        client: OpenAI-compatible client (for OpenRouter). If None, creates one.
        only_pairs: Optional set of (subcategory_id, language) tuples. When
            provided, only those pairs are generated; everything else is
            skipped. Use for incremental fills when a subcategory has been
            added to subcategories.json after the initial run.

    Returns:
        DataFrame with columns: category, text, language, subcategory_name
    """
    subcategories_path = config.resolve_path(config.anchors.subcategories_file)
    with open(subcategories_path, encoding="utf-8") as f:
        subcategories = json.load(f)

    if client is None:
        from .generation import create_llm_client
        client = create_llm_client(config)

    context_texts = _load_anchor_context(config)

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
            if only_pairs is not None and (cat_id, lang) not in only_pairs:
                continue
            prompt = _build_anchor_prompt(name, anchor_prompt, lang, context_texts)

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
    context_texts: list[str],
) -> str:
    """Build the user prompt for anchor generation from the template."""
    sections = load_prompt_sections("anchor_generation")
    user_template = sections.get("user", "")

    lang_name = LANGUAGE_NAMES.get(language, language)

    if custom_prompt:
        subcategory_name = f"{subcategory_name} ({custom_prompt})"

    return user_template.format(
        subcategory_name=subcategory_name,
        context_section=_format_context(context_texts),
        language_name=lang_name,
    )


def _load_anchor_context(config: PipelineConfig) -> list[str]:
    """Resolve and load every context file referenced by `anchors.context_*`.

    Strict mode: missing files raise FileNotFoundError. An empty list is
    returned only when the user set no context paths at all.
    """
    paths: list[Path] = []

    if config.anchors.context_folder:
        folder = config.resolve_path(config.anchors.context_folder)
        expanded = _expand_context_folder(folder)
        if not expanded:
            logger.warning(
                f"anchors.context_folder is empty (or no supported files): {folder}"
            )
        paths.extend(expanded)
        logger.info(f"anchors.context_folder: {len(expanded)} file(s) from {folder}")

    if config.anchors.context_files:
        for p in config.anchors.context_files:
            paths.append(config.resolve_path(p))
        logger.info(
            f"anchors.context_files: {len(config.anchors.context_files)} additional file(s)"
        )

    if not paths:
        return []

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "anchors context files not found (strict mode — fix anchors.context_*):\n  "
            + "\n  ".join(str(m) for m in missing)
        )

    texts: list[str] = []
    for p in paths:
        text = _load_context_document(p)
        if text:
            texts.append(text)
            logger.info(f"Loaded anchor context: {p.name} ({len(text)} chars)")
    return texts


def _expected_anchor_pairs(config: PipelineConfig) -> tuple[list[dict], set[tuple[int, str]]]:
    """Read subcategories.json and return (subcategories, expected (id, lang) pairs)."""
    subcategories_path = config.resolve_path(config.anchors.subcategories_file)
    with open(subcategories_path, encoding="utf-8") as f:
        subcategories = json.load(f)
    languages = config.anchors.languages or ["en"]
    pairs = {(int(s["id"]), lang) for s in subcategories for lang in languages}
    return subcategories, pairs


def _existing_anchor_pairs(df: pd.DataFrame) -> set[tuple[int, str]]:
    """Return the set of (category, language) pairs already covered in df."""
    if df is None or len(df) == 0:
        return set()
    if "language" not in df.columns:
        # Legacy monolingual file — treat as language='en' implicitly.
        cats = df["category"].astype(int).tolist()
        return {(c, "en") for c in cats}
    return {
        (int(c), str(l))
        for c, l in zip(df["category"].tolist(), df["language"].tolist())
    }


def load_or_generate_anchors(config: PipelineConfig, client=None) -> pd.DataFrame:
    """Load anchors from file if available, otherwise generate them.

    The function understands incremental updates: if `subcategories.json` has
    grown since the anchors file was written, the missing (subcategory ×
    language) pairs are detected and either:

      - `method: from_file`     → raise a helpful error telling the user how
                                  to fill the gap (switch to llm_generated
                                  for one run, or add the entries manually).
      - `method: llm_generated` → generate only the missing pairs, merge
                                  them into the existing file, and save.

    Resolution order for the on-disk anchors file:
      1. config.anchors.anchors_file (explicit path)
      2. project-local fallback (data/anchors/anchors.csv)
    """
    anchors_path = resolve_anchors_path(config)
    legacy_cache_path = config.anchors_cache_path

    def _read_existing() -> tuple[pd.DataFrame | None, Path | None]:
        for p in (anchors_path, legacy_cache_path):
            if p.exists():
                df = load_anchors(p)
                if len(df) > 0:
                    return df, p
        return None, None

    _, expected_pairs = _expected_anchor_pairs(config)
    existing_df, existing_path = _read_existing()
    existing_pairs = _existing_anchor_pairs(existing_df) if existing_df is not None else set()
    missing_pairs = expected_pairs - existing_pairs

    if config.anchors.method == "from_file":
        if existing_df is None:
            raise FileNotFoundError(
                f"Anchor method is 'from_file' but no anchors found at "
                f"{config.anchors.anchors_file or legacy_cache_path}. "
                "Provide an anchors file or set method to 'llm_generated'."
            )
        if missing_pairs:
            preview = sorted(missing_pairs)[:5]
            raise ValueError(
                f"anchors.method='from_file' but {len(missing_pairs)} "
                f"(subcategory_id, language) pair(s) are missing from "
                f"{existing_path} — e.g. {preview}. "
                "Switch anchors.method to 'llm_generated' for one run so the "
                "missing entries are generated and merged in, or add them to "
                "the anchors file manually."
            )
        return existing_df

    # method == "llm_generated" (any other value would have failed validation)
    if existing_df is not None and not missing_pairs:
        logger.info(
            f"Anchors file at {existing_path} already covers all "
            f"{len(expected_pairs)} expected (subcategory, language) pairs"
        )
        return existing_df

    if existing_df is not None and missing_pairs:
        logger.info(
            f"Anchors file at {existing_path} covers {len(existing_pairs)}"
            f"/{len(expected_pairs)} pairs; filling {len(missing_pairs)} "
            "missing entries via LLM"
        )
        new_df = generate_anchors(config, client, only_pairs=missing_pairs)
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.sort_values(["category", "language"]).reset_index(drop=True)
        save_anchors(merged, anchors_path)
        return merged

    new_df = generate_anchors(config, client)
    save_anchors(new_df, anchors_path)
    return new_df
