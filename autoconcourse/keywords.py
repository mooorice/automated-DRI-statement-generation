"""Generate a keywords.json file via an LLM call.

Used to populate `corpus.filtering.keywords_file` for the optional keyword
pre-filter. Can run in two modes:

  1. With a pipeline config — defaults are pulled from `topic`, `anchors`,
     `corpus.filtering`, and `llm` sections.
  2. Standalone — pass `--topic`, `--language`, `--output`, etc. directly.

Output is always a language-grouped JSON dict matching the format expected by
`autoconcourse.corpus.load_keywords()`:

    {"en": ["term1", "term2", ...], "de": ["..."]}

Examples
--------
With a config (uses topic.name, topic.languages, anchors.subcategories_file,
anchors.context_document, anchors.model, and writes to
corpus.filtering.keywords_file)::

    python -m autoconcourse.keywords --config projects/TA-Swiss/config.yaml

Standalone::

    python -m autoconcourse.keywords \\
        --topic "Participatory technology assessment in Switzerland" \\
        --context-file projects/TA-Swiss/scoping_note.md \\
        --language en \\
        --output projects/TA-Swiss/keywords.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

from .config import GenerationConfig, LLMConfig, PipelineConfig, TopicConfig, load_config
from .generation import create_llm_client
from .prompts import load_prompt_sections

logger = logging.getLogger("autoconcourse.keywords")

LANGUAGE_NAMES = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "rm": "Romansh",
}


def _format_subcategories(subcategories: list[dict] | None) -> str:
    """Render subcategories as a bulleted block, or empty string if none."""
    if not subcategories:
        return ""
    lines = ["Subcategories (use these to broaden vocabulary coverage):"]
    for s in subcategories:
        name = s.get("name", "")
        desc = s.get("description") or s.get("topic_focus") or ""
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    return "\n".join(lines) + "\n"


def _format_context(context_texts: list[str], max_chars: int = 16000) -> str:
    """Concatenate and cap context texts for inclusion in the prompt."""
    if not context_texts:
        return ""
    joined = "\n\n---\n\n".join(t for t in context_texts if t)
    if not joined.strip():
        return ""
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n[... truncated]"
    return f"\nReference texts:\n\n{joined}\n"


def _load_context_document(path: Path) -> str:
    """Load text from a context document (.pdf or any text format)."""
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf
        except ImportError:
            logger.warning(
                f"pymupdf not installed; skipping PDF context document {path}. "
                "Install with: pip install pymupdf"
            )
            return ""
        doc = pymupdf.open(str(path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    return path.read_text(encoding="utf-8")


def _build_keyword_prompt(
    topic: str,
    subcategories: list[dict] | None,
    context_texts: list[str],
    language: str,
    n: int,
) -> str:
    sections = load_prompt_sections("keyword_generation")
    user_template = sections.get("user", "")
    return user_template.format(
        topic=topic,
        subcategories_section=_format_subcategories(subcategories),
        context_section=_format_context(context_texts),
        language_name=LANGUAGE_NAMES.get(language, language),
        n=n,
    )


def _parse_keywords(raw: str) -> list[str]:
    """Extract a deduplicated keyword list from LLM JSON output, tolerantly."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse keywords from LLM output: {raw[:200]!r}")
        data = json.loads(match.group(0))

    if isinstance(data, dict):
        for key in ("keywords", "items", "list", "terms"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                raise ValueError(f"No keyword list found in JSON object: keys={list(data)}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of keywords, got {type(data).__name__}")

    seen: set[str] = set()
    out: list[str] = []
    for item in data:
        s = str(item).strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def generate_keywords(
    topic: str,
    languages: list[str],
    *,
    subcategories: list[dict] | None = None,
    context_texts: list[str] | None = None,
    per_language: int = 50,
    model: str = "openai/gpt-4o",
    client=None,
    config: PipelineConfig | None = None,
) -> dict[str, list[str]]:
    """Generate a keyword list per language via an LLM call.

    Returns a dict keyed by language code, e.g. ``{"en": ["..."], "de": ["..."]}``.
    """
    if client is None:
        if config is None:
            config = _minimal_config_for_llm(model)
        client = create_llm_client(config)

    sections = load_prompt_sections("keyword_generation")
    system_prompt = sections.get(
        "system",
        "You are an expert at building topical lexicons for corpus filtering.",
    )

    result: dict[str, list[str]] = {}
    for lang in languages:
        user_prompt = _build_keyword_prompt(
            topic=topic,
            subcategories=subcategories,
            context_texts=context_texts or [],
            language=lang,
            n=per_language,
        )
        logger.info(f"Generating keywords: language={lang}, target≈{per_language}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Some providers/models don't support response_format; retry plain.
            logger.debug(f"response_format unsupported ({exc}); retrying without")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

        raw = response.choices[0].message.content or ""
        keywords = _parse_keywords(raw)
        logger.info(f"  → {len(keywords)} keywords for {lang}")
        result[lang] = keywords

    return result


def save_keywords(keywords: dict[str, list[str]], path: str | Path) -> None:
    """Write the keyword dict to JSON in the format `corpus.load_keywords` expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)
    total = sum(len(v) for v in keywords.values())
    logger.info(f"Saved {total} keywords across {len(keywords)} language(s) to {path}")


def _minimal_config_for_llm(model: str) -> PipelineConfig:
    """Build a minimal PipelineConfig sufficient to construct an LLM client."""
    return PipelineConfig(
        topic=TopicConfig(name="keywords-cli", id="keywords-cli", languages=["en"]),
        llm=LLMConfig(default_model=model),
        corpus=None,
        anchors=None,
        embedding=None,
        retrieval=None,
        groups=None,
        generation=GenerationConfig(model=model),
    )


CONTEXT_FILE_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf"}


def _expand_context_folder(folder: Path) -> list[Path]:
    """Return all readable, non-hidden context files inside `folder`, sorted.

    Recurses into subdirectories so users can organise their context however
    they like. Skips dotfiles and unsupported extensions.
    """
    if not folder.exists():
        raise FileNotFoundError(f"keywords.context_folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"keywords.context_folder is not a directory: {folder}")

    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(folder).parts):
            continue
        if p.suffix.lower() not in CONTEXT_FILE_EXTENSIONS:
            continue
        out.append(p)
    return out


def _resolve_inputs(args, parser) -> dict:
    """Resolve effective inputs from CLI args + optional config.

    Strict resolution: no silent fallbacks to anchors.* or other unrelated
    sections. The keyword section is self-contained.
    """
    config: PipelineConfig | None = None
    if args.config:
        config = load_config(args.config)

    kw_cfg = config.keywords if config else None
    resolve = (lambda p: config.resolve_path(p)) if config else (lambda p: Path(p))

    # --- topic
    if args.topic:
        topic, topic_source = args.topic, "CLI --topic"
    elif kw_cfg and kw_cfg.topic:
        topic, topic_source = kw_cfg.topic, "keywords.topic"
    elif config and config.topic.description:
        topic, topic_source = config.topic.description, "topic.description"
    elif config:
        topic, topic_source = config.topic.name, "topic.name (no description set)"
    else:
        parser.error("--topic is required when --config is not provided")
    logger.info(f"topic: {topic!r} (from {topic_source})")

    # --- languages
    if args.language:
        languages, lang_source = args.language, "CLI --language"
    elif kw_cfg and kw_cfg.languages:
        languages, lang_source = list(kw_cfg.languages), "keywords.languages"
    elif config:
        languages, lang_source = list(config.topic.languages), "topic.languages"
    else:
        parser.error("--language is required when --config is not provided")
    if not languages:
        parser.error("No languages resolved. Set --language, keywords.languages, or topic.languages.")
    logger.info(f"languages: {languages} (from {lang_source})")

    # --- per_language
    if args.per_language is not None:
        per_language = args.per_language
    elif kw_cfg:
        per_language = kw_cfg.per_language
    else:
        per_language = 50
    logger.info(f"per_language: {per_language}")

    # --- model
    if args.model:
        model, model_source = args.model, "CLI --model"
    elif kw_cfg and kw_cfg.model:
        model, model_source = kw_cfg.model, "keywords.model"
    elif config and config.llm and config.llm.default_model:
        model, model_source = config.llm.default_model, "llm.default_model"
    else:
        model, model_source = "openai/gpt-4o", "built-in default"
    logger.info(f"model: {model} (from {model_source})")

    # --- subcategories — no fallback to anchors.subcategories_file
    sub_path: str | None = None
    if args.subcategories:
        sub_path = args.subcategories
    elif kw_cfg and kw_cfg.subcategories_file:
        sub_path = str(resolve(kw_cfg.subcategories_file))
    subcategories: list[dict] | None = None
    if sub_path:
        sp = Path(sub_path)
        if not sp.exists():
            parser.error(f"subcategories file not found: {sp}")
        with open(sp, encoding="utf-8") as f:
            subcategories = json.load(f)
        logger.info(f"subcategories: {len(subcategories)} entries from {sp}")
    else:
        logger.info("subcategories: none (no keywords.subcategories_file set — strict mode)")

    # --- context paths — no fallback to anchors.context_document
    context_paths: list[Path] = []
    if kw_cfg and kw_cfg.context_folder:
        folder = resolve(kw_cfg.context_folder)
        expanded = _expand_context_folder(folder)
        if not expanded:
            logger.warning(f"keywords.context_folder is empty (or no supported files): {folder}")
        context_paths.extend(expanded)
        logger.info(f"context_folder: {len(expanded)} file(s) loaded from {folder}")
    if kw_cfg and kw_cfg.context_files:
        for p in kw_cfg.context_files:
            context_paths.append(resolve(p))
        logger.info(f"context_files: {len(kw_cfg.context_files)} additional file(s) from keywords.context_files")
    if args.context_file:
        for cf in args.context_file:
            context_paths.append(resolve(cf))
        logger.info(f"CLI --context-file: {len(args.context_file)} additional file(s)")
    if args.context_folder:
        folder = resolve(args.context_folder)
        expanded = _expand_context_folder(folder)
        context_paths.extend(expanded)
        logger.info(f"CLI --context-folder: {len(expanded)} file(s) loaded from {folder}")

    if not context_paths:
        logger.warning(
            "No context provided. Keyword generation will rely on topic + subcategories only. "
            "Set keywords.context_folder or keywords.context_files for grounding."
        )

    # --- output
    if args.output:
        output = args.output
    elif kw_cfg and kw_cfg.output_file:
        output = str(resolve(kw_cfg.output_file))
    elif config and config.corpus and config.corpus.filtering and config.corpus.filtering.keywords_file:
        output = str(resolve(config.corpus.filtering.keywords_file))
    else:
        parser.error(
            "Cannot determine output path. Set --output, keywords.output_file, "
            "or corpus.filtering.keywords_file."
        )
    logger.info(f"output: {output}")

    return {
        "config": config,
        "topic": topic,
        "languages": languages,
        "per_language": per_language,
        "model": model,
        "subcategories": subcategories,
        "context_paths": context_paths,
        "output": output,
    }


def keywords_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m autoconcourse.keywords",
        description="Generate a keywords.json file via an LLM call.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full examples.",
    )
    parser.add_argument(
        "--config",
        help="Pipeline config YAML. Provides defaults for topic, languages, "
             "subcategories, context, model, and output path.",
    )
    parser.add_argument(
        "--topic",
        help="Topic description for keyword generation. "
             "Overrides keywords.topic (which itself overrides topic.name).",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="Language code, repeatable (e.g. --language en --language de). "
             "Overrides config.topic.languages.",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=[],
        help="Path to a context document (.txt, .md, .pdf), repeatable. "
             "Stacked on top of keywords.context_folder + keywords.context_files.",
    )
    parser.add_argument(
        "--context-folder",
        help="Path to a directory; all supported files inside (.txt/.md/.pdf) "
             "are loaded as context. Stacks with keywords.context_folder.",
    )
    parser.add_argument(
        "--subcategories",
        help="Path to subcategories JSON. Overrides keywords.subcategories_file. "
             "No fallback to anchors.subcategories_file.",
    )
    parser.add_argument(
        "--per-language",
        type=int,
        default=None,
        help="Target keywords per language. "
             "Overrides config.keywords.per_language (default: 50).",
    )
    parser.add_argument(
        "--model",
        help="LLM model slug (e.g. 'openai/gpt-4o'). "
             "Overrides keywords.model > llm.default_model.",
    )
    parser.add_argument(
        "--output",
        help="Output keywords.json path. Defaults to config.corpus.filtering.keywords_file.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    inputs = _resolve_inputs(args, parser)

    context_texts: list[str] = []
    missing: list[Path] = []
    for cp in inputs["context_paths"]:
        if not cp.exists():
            missing.append(cp)
            continue
        text = _load_context_document(cp)
        if text:
            context_texts.append(text)
            logger.info(f"Loaded context: {cp.name} ({len(text)} chars)")
    if missing:
        parser.error(
            "Context files not found (strict mode — fix the paths in keywords.context_*):\n  "
            + "\n  ".join(str(m) for m in missing)
        )

    config = inputs["config"] or _minimal_config_for_llm(inputs["model"])
    client = create_llm_client(config)

    keywords = generate_keywords(
        topic=inputs["topic"],
        languages=inputs["languages"],
        subcategories=inputs["subcategories"],
        context_texts=context_texts,
        per_language=inputs["per_language"],
        model=inputs["model"],
        client=client,
        config=config,
    )

    save_keywords(keywords, inputs["output"])
    total = sum(len(v) for v in keywords.values())
    print(f"Wrote {total} keywords across {len(keywords)} language(s) to {inputs['output']}")


if __name__ == "__main__":
    keywords_main()
