"""Generate a subcategories.json file via an LLM call.

Used to populate the file that `anchors.subcategories_file` points to. Two modes:

  1. **Seeded**: user provides rough cluster definitions in plain text
     (`subcategories.seeds` inline or `subcategories.seeds_file`). The LLM
     produces one structured subcategory per seed cluster, preserving order
     and intent. Forgiving about input format.

  2. **Unseeded**: LLM generates `subcategories.count` subcategories from
     topic + context.

Output is a JSON list matching the shape the pipeline expects::

    [
      {"id": 0, "name": "...", "description": "...",
       "topic_focus": "...", "anchor_prompt": "..."},
      ...
    ]

Examples
--------
With a config (recommended)::

    python -m autoconcourse.subcategories --config projects/TA-Swiss/config.yaml

Standalone with seeds::

    python -m autoconcourse.subcategories \\
        --topic "Participatory tech assessment of AI in education and work" \\
        --language en \\
        --seeds-file projects/TA-Swiss/cluster_draft.md \\
        --output projects/TA-Swiss/subcategories.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

from .config import GenerationConfig, LLMConfig, PipelineConfig, TopicConfig, load_config
from .generation import create_llm_client
from .keywords import _expand_context_folder, _load_context_document
from .prompts import load_prompt_sections

logger = logging.getLogger("autoconcourse.subcategories")

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

REQUIRED_FIELDS = ("id", "name", "description", "topic_focus", "anchor_prompt")


def _format_seeds(seeds_text: str | None) -> str:
    if not seeds_text or not seeds_text.strip():
        return ""
    return (
        "Seed clusters (user-defined, plain text — parse forgivingly):\n\n"
        f"{seeds_text.strip()}\n\n"
    )


def _format_context(context_texts: list[str], max_chars: int = 16000) -> str:
    if not context_texts:
        return ""
    joined = "\n\n---\n\n".join(t for t in context_texts if t)
    if not joined.strip():
        return ""
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n[... truncated]"
    return f"Reference texts:\n\n{joined}\n\n"


def _build_subcategory_prompt(
    topic: str,
    seeds_text: str | None,
    context_texts: list[str],
    language: str,
    count: int,
    seeded: bool,
) -> str:
    sections = load_prompt_sections("subcategory_generation")
    user_template = sections.get("user", "")

    if seeded:
        count_instruction = (
            "Use the seed clusters above to produce the structured subcategories. "
            "Produce exactly ONE subcategory per seed, in the same order, preserving "
            "their names and intent. Do not merge, split, reorder, or add seed clusters."
        )
    else:
        count_instruction = (
            f"Produce {count} subcategories that cover the topic's thematic space "
            "comprehensively and with minimal overlap."
        )

    return user_template.format(
        topic=topic,
        seeds_section=_format_seeds(seeds_text),
        context_section=_format_context(context_texts),
        language_name=LANGUAGE_NAMES.get(language, language),
        count_instruction=count_instruction,
    )


def _parse_subcategories(raw: str) -> list[dict]:
    """Parse and normalise the LLM's JSON output into a list of subcategory dicts."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse subcategories from LLM output: {raw[:200]!r}")
        data = json.loads(match.group(0))

    if isinstance(data, dict):
        for key in ("subcategories", "items", "list", "categories"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                raise ValueError(f"No subcategory list found in JSON object: keys={list(data)}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of subcategories, got {type(data).__name__}")

    out: list[dict] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Subcategory at index {i} is not an object: {entry!r}")
        normalized = {
            "id": int(entry.get("id", i)) if entry.get("id") is not None else i,
            "name": str(entry.get("name", "")).strip(),
            "description": str(entry.get("description", "")).strip(),
            "topic_focus": str(entry.get("topic_focus", "")).strip(),
            "anchor_prompt": str(entry.get("anchor_prompt", "")).strip(),
        }
        if not normalized["name"]:
            raise ValueError(f"Subcategory at index {i} has no name: {entry!r}")
        out.append(normalized)

    # Reassign sequential ids defensively so downstream code can index by id.
    for i, sc in enumerate(out):
        sc["id"] = i
    return out


def generate_subcategories(
    topic: str,
    *,
    language: str = "en",
    count: int = 8,
    seeds_text: str | None = None,
    context_texts: list[str] | None = None,
    model: str = "openai/gpt-4o",
    client=None,
    config: PipelineConfig | None = None,
) -> list[dict]:
    """Generate a structured subcategory list via an LLM call.

    Returns a list of dicts with id/name/description/topic_focus/anchor_prompt.
    """
    if client is None:
        if config is None:
            config = _minimal_config_for_llm(model)
        client = create_llm_client(config)

    sections = load_prompt_sections("subcategory_generation")
    system_prompt = sections.get(
        "system",
        "You are an expert at structuring topic taxonomies for discourse analysis.",
    )

    seeded = bool(seeds_text and seeds_text.strip())
    user_prompt = _build_subcategory_prompt(
        topic=topic,
        seeds_text=seeds_text,
        context_texts=context_texts or [],
        language=language,
        count=count,
        seeded=seeded,
    )

    logger.info(
        f"Generating subcategories: language={language}, "
        f"{'seeded' if seeded else f'target={count}'}"
    )

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
    subcategories = _parse_subcategories(raw)
    logger.info(f"  → {len(subcategories)} subcategories produced")
    return subcategories


def save_subcategories(subcategories: list[dict], path: str | Path) -> None:
    """Write the subcategories list to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(subcategories, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(subcategories)} subcategories to {path}")


def _minimal_config_for_llm(model: str) -> PipelineConfig:
    """Build a minimal PipelineConfig sufficient to construct an LLM client."""
    return PipelineConfig(
        topic=TopicConfig(name="subcategories-cli", id="subcategories-cli", languages=["en"]),
        llm=LLMConfig(default_model=model),
        corpus=None,
        anchors=None,
        embedding=None,
        retrieval=None,
        groups=None,
        generation=GenerationConfig(model=model),
    )


def _count_seed_clusters(seeds_text: str) -> int:
    """Best-effort count of how many seed clusters the user listed.

    Looks for markers like "Cluster N", numbered lines, or bulleted entries.
    Forgiving — used only to set a default `count`, not to enforce structure.
    """
    if not seeds_text:
        return 0
    cluster_hits = len(re.findall(r"(?im)^\s*[•\-\*●]?\s*cluster\s+\d+", seeds_text))
    if cluster_hits > 0:
        return cluster_hits
    numbered = len(re.findall(r"(?m)^\s*\d+[\.\):]", seeds_text))
    if numbered > 0:
        return numbered
    bulleted = len(re.findall(r"(?m)^\s*[•\-\*●]\s", seeds_text))
    return bulleted


def _resolve_inputs(args, parser) -> dict:
    """Resolve effective inputs from CLI args + optional config (strict mode)."""
    config: PipelineConfig | None = None
    if args.config:
        config = load_config(args.config)

    sc_cfg = config.subcategories if config else None
    resolve = (lambda p: config.resolve_path(p)) if config else (lambda p: Path(p))

    # --- topic
    if args.topic:
        topic, src = args.topic, "CLI --topic"
    elif sc_cfg and sc_cfg.topic:
        topic, src = sc_cfg.topic, "subcategories.topic"
    elif config and config.topic.description:
        topic, src = config.topic.description, "topic.description"
    elif config:
        topic, src = config.topic.name, "topic.name (no description set)"
    else:
        parser.error("--topic is required when --config is not provided")
    logger.info(f"topic: {topic!r} (from {src})")

    # --- language
    if args.language:
        language, src = args.language, "CLI --language"
    elif sc_cfg and sc_cfg.language:
        language, src = sc_cfg.language, "subcategories.language"
    elif config and config.topic.languages:
        language, src = config.topic.languages[0], "topic.languages[0]"
    else:
        language, src = "en", "built-in default"
    logger.info(f"language: {language} (from {src})")

    # --- seeds
    seeds_text: str | None = None
    seeds_source: str | None = None
    if args.seeds_file:
        seeds_text = Path(args.seeds_file).read_text(encoding="utf-8")
        seeds_source = f"CLI --seeds-file {args.seeds_file}"
    elif sc_cfg and sc_cfg.seeds_file:
        p = resolve(sc_cfg.seeds_file)
        if not p.exists():
            parser.error(f"subcategories.seeds_file not found: {p}")
        seeds_text = p.read_text(encoding="utf-8")
        seeds_source = f"subcategories.seeds_file ({p})"
    elif sc_cfg and sc_cfg.seeds:
        seeds_text = sc_cfg.seeds
        seeds_source = "subcategories.seeds (inline)"
    if seeds_text:
        logger.info(f"seeds: {len(seeds_text)} chars (from {seeds_source})")
    else:
        logger.info("seeds: none (unseeded generation)")

    # --- count
    if args.count is not None:
        count, src = args.count, "CLI --count"
    elif seeds_text:
        inferred = _count_seed_clusters(seeds_text)
        if inferred > 0:
            count, src = inferred, f"inferred from {inferred} seed clusters"
        else:
            count, src = (sc_cfg.count if sc_cfg else 8), "subcategories.count (fallback)"
    elif sc_cfg:
        count, src = sc_cfg.count, "subcategories.count"
    else:
        count, src = 8, "built-in default"
    logger.info(f"count: {count} (from {src})")

    # --- model
    if args.model:
        model, src = args.model, "CLI --model"
    elif sc_cfg and sc_cfg.model:
        model, src = sc_cfg.model, "subcategories.model"
    elif config and config.llm and config.llm.default_model:
        model, src = config.llm.default_model, "llm.default_model"
    else:
        model, src = "openai/gpt-4o", "built-in default"
    logger.info(f"model: {model} (from {src})")

    # --- context paths (no anchors fallback)
    context_paths: list[Path] = []
    if sc_cfg and sc_cfg.context_folder:
        folder = resolve(sc_cfg.context_folder)
        expanded = _expand_context_folder(folder)
        if not expanded:
            logger.warning(f"subcategories.context_folder is empty: {folder}")
        context_paths.extend(expanded)
        logger.info(f"context_folder: {len(expanded)} file(s) from {folder}")
    if sc_cfg and sc_cfg.context_files:
        for p in sc_cfg.context_files:
            context_paths.append(resolve(p))
        logger.info(f"context_files: {len(sc_cfg.context_files)} from subcategories.context_files")
    if args.context_file:
        for cf in args.context_file:
            context_paths.append(resolve(cf))
        logger.info(f"CLI --context-file: {len(args.context_file)} additional file(s)")
    if args.context_folder:
        folder = resolve(args.context_folder)
        expanded = _expand_context_folder(folder)
        context_paths.extend(expanded)
        logger.info(f"CLI --context-folder: {len(expanded)} file(s) from {folder}")

    # --- output
    if args.output:
        output = args.output
    elif sc_cfg and sc_cfg.output_file:
        output = str(resolve(sc_cfg.output_file))
    elif config and config.anchors and config.anchors.subcategories_file:
        output = str(resolve(config.anchors.subcategories_file))
    else:
        parser.error(
            "Cannot determine output path. Set --output, subcategories.output_file, "
            "or anchors.subcategories_file."
        )
    logger.info(f"output: {output}")

    return {
        "config": config,
        "topic": topic,
        "language": language,
        "count": count,
        "seeds_text": seeds_text,
        "model": model,
        "context_paths": context_paths,
        "output": output,
    }


def subcategories_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m autoconcourse.subcategories",
        description="Generate a subcategories.json file via an LLM call.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full examples.",
    )
    parser.add_argument("--config", help="Pipeline config YAML.")
    parser.add_argument(
        "--topic",
        help="Topic description. Overrides subcategories.topic > topic.name.",
    )
    parser.add_argument(
        "--language",
        help="Output language code (e.g. 'en'). "
             "Overrides subcategories.language > topic.languages[0].",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of subcategories to generate (ignored when seeds are present "
             "— count is inferred from the seed list).",
    )
    parser.add_argument(
        "--seeds-file",
        help="Path to a plain-text file with rough cluster definitions. "
             "Overrides subcategories.seeds_file > subcategories.seeds.",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=[],
        help="Path to a context document (.txt, .md, .pdf), repeatable. "
             "Stacked on top of subcategories.context_folder/files.",
    )
    parser.add_argument(
        "--context-folder",
        help="Path to a directory; all supported files inside are loaded as context.",
    )
    parser.add_argument(
        "--model",
        help="LLM model slug. Overrides subcategories.model > llm.default_model.",
    )
    parser.add_argument(
        "--output",
        help="Output subcategories.json path. "
             "Overrides subcategories.output_file > anchors.subcategories_file.",
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
            "Context files not found (strict mode):\n  "
            + "\n  ".join(str(m) for m in missing)
        )

    config = inputs["config"] or _minimal_config_for_llm(inputs["model"])
    client = create_llm_client(config)

    subcategories = generate_subcategories(
        topic=inputs["topic"],
        language=inputs["language"],
        count=inputs["count"],
        seeds_text=inputs["seeds_text"],
        context_texts=context_texts,
        model=inputs["model"],
        client=client,
        config=config,
    )

    save_subcategories(subcategories, inputs["output"])
    print(f"Wrote {len(subcategories)} subcategories to {inputs['output']}")


if __name__ == "__main__":
    subcategories_main()
