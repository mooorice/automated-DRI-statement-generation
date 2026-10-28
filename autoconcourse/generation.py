"""LLM statement generation via any OpenAI-compatible API."""

import json
import logging
import os
import re
import time

import pandas as pd
from openai import OpenAI

from .config import PipelineConfig
from .prompts import load_examples, load_prompt_sections

logger = logging.getLogger("autoconcourse.generation")

_PROMPT_CACHE: dict[str, dict[str, str]] = {}
_EXAMPLES_CACHE: dict[str, str] = {}


def _get_prompt_sections(template_name: str) -> dict[str, str]:
    if template_name not in _PROMPT_CACHE:
        _PROMPT_CACHE[template_name] = load_prompt_sections(template_name)
    return _PROMPT_CACHE[template_name]


def _resolve_dri_examples(
    config: PipelineConfig,
    statement_type: str,
) -> str:
    """Return the reference-examples block for a DRI statement type.

    Resolution:
      - `dri_<type>_examples_file` set to a path → load that file.
      - `dri_<type>_examples_file == ""`         → no examples (empty block).
      - `dri_<type>_examples_file is None`       → packaged default at
        `autoconcourse/prompts/examples/dri_<type>_examples.txt`.
    """
    field_name = f"dri_{statement_type}_examples_file"
    override = getattr(config.generation, field_name, None)
    max_examples = config.generation.dri_max_examples

    cache_key = f"{statement_type}|{override}|{max_examples}"
    if cache_key in _EXAMPLES_CACHE:
        return _EXAMPLES_CACHE[cache_key]

    if override == "":
        rendered = ""
    elif override is not None:
        rendered = load_examples(
            path=config.resolve_path(override),
            max_examples=max_examples,
        )
    else:
        rendered = load_examples(
            name=f"dri_{statement_type}_examples",
            max_examples=max_examples,
        )

    _EXAMPLES_CACHE[cache_key] = rendered
    return rendered


PROVIDER_DEFAULTS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url": None,                # OpenAI SDK uses api.openai.com by default
        "api_key_env": "OPENAI_API_KEY",
    },
}


def resolve_llm_settings(config: PipelineConfig) -> dict:
    """Resolve provider, base_url, api_key_env, and headers from llm config.

    Returns a dict with: provider, base_url, api_key_env, headers. Pure function
    (no env reads, no client construction) so it's straightforward to test.
    """
    provider = config.llm.provider
    if provider == "custom":
        if not config.llm.base_url:
            raise ValueError("llm.provider: 'custom' requires llm.base_url to be set")
        if not config.llm.api_key_env:
            raise ValueError("llm.provider: 'custom' requires llm.api_key_env to be set")
        base_url = config.llm.base_url
        api_key_env = config.llm.api_key_env
    elif provider in PROVIDER_DEFAULTS:
        defaults = PROVIDER_DEFAULTS[provider]
        base_url = config.llm.base_url or defaults["base_url"]
        api_key_env = config.llm.api_key_env or defaults["api_key_env"]
    else:
        raise ValueError(
            f"Unknown llm.provider: {provider!r}. "
            f"Use one of: openrouter, openai, custom."
        )

    headers = dict(config.llm.extra_headers or {})
    if provider == "openrouter":
        if config.llm.site_url and "HTTP-Referer" not in headers:
            headers["HTTP-Referer"] = config.llm.site_url
        if config.llm.site_name and "X-Title" not in headers:
            headers["X-Title"] = config.llm.site_name

    return {
        "provider": provider,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "headers": headers,
    }


def create_llm_client(config: PipelineConfig) -> OpenAI:
    """Create an OpenAI-compatible client from config.

    Mirrors the OpenAI / OpenRouter SDK pattern:

        OpenAI(api_key=..., base_url=..., default_headers=...)

    Provider selection is explicit via `llm.provider` (default: "openrouter").
    """
    from dotenv import load_dotenv
    load_dotenv()

    settings = resolve_llm_settings(config)

    api_key = os.getenv(settings["api_key_env"])
    if not api_key:
        raise ValueError(
            f"No API key found in env var {settings['api_key_env']!r}. "
            f"Set it in .env (llm.provider={settings['provider']!r})."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=settings["base_url"],
        default_headers=settings["headers"] or None,
    )
    logger.info(
        "LLM client ready: provider=%s base_url=%s model=%s",
        settings["provider"],
        settings["base_url"] or "sdk-default",
        config.generation.model,
    )
    return client


def _format_group_instruction(template: str | None, group: str | None) -> str:
    """Render the group_instruction text from config template + group label.

    Returns empty string if no group, or no template, or template is empty.
    """
    if not group or not template:
        return ""
    return "\n" + template.format(group=group).strip() + "\n"


def build_generation_prompt(
    template_name: str,
    subcategory: dict,
    retrieved_texts: list[str],
    n_statements: int,
    output_language: str,
    group: str | None = None,
    group_instruction_template: str | None = None,
    examples: str = "",
) -> str:
    """Render the user prompt for statement generation."""
    sections = _get_prompt_sections(template_name)
    user_template = sections.get("user", "")

    topic_focus = subcategory.get("topic_focus", subcategory["name"])
    group_instruction = _format_group_instruction(group_instruction_template, group)

    paragraph_lines = [f"[{i}] {text}" for i, text in enumerate(retrieved_texts[:100], 1)]

    format_kwargs = {
        "n_statements": n_statements,
        "topic_focus": topic_focus,
        "group_instruction": group_instruction,
        "output_language": output_language,
        "paragraphs": "\n".join(paragraph_lines),
    }
    # Only inject {examples} if the template actually uses it, so older
    # templates (q-concourse) keep working unchanged.
    if "{examples}" in user_template:
        format_kwargs["examples"] = examples or "(none provided)"

    return user_template.format(**format_kwargs)


def generate_statements(
    retrieved_texts: dict,
    subcategories: list[dict],
    config: PipelineConfig,
    client: OpenAI | None = None,
) -> list[dict]:
    """Generate statements for every (group, subcategory) bucket in `retrieved_texts`.

    Args:
        retrieved_texts: dict keyed by either subcategory_id (int) or
            (group_name, subcategory_id) tuple, mapping to a DataFrame of passages.
        subcategories: list of subcategory dicts (must have 'id' and 'name').
        config: pipeline config.
        client: LLM client; created if None.

    Returns:
        List of statement dicts with id, text, subcategory, subcategory_id,
        group, source_paragraphs.
    """
    if client is None:
        client = create_llm_client(config)

    default_n = config.generation.statements_per_group_per_subcategory
    if config.generation.process == "dri":
        n_cons = (
            config.generation.dri_considerations_per_group_per_subcategory
            if config.generation.dri_considerations_per_group_per_subcategory is not None
            else default_n
        )
        n_pref = (
            config.generation.dri_preferences_per_group_per_subcategory
            if config.generation.dri_preferences_per_group_per_subcategory is not None
            else default_n
        )
        generation_specs = [
            (
                "dri_consideration_generation",
                "consideration",
                n_cons,
                _resolve_dri_examples(config, "consideration"),
            ),
            (
                "dri_preference_generation",
                "preference",
                n_pref,
                _resolve_dri_examples(config, "preference"),
            ),
        ]
    else:
        generation_specs = [("q-concourse_generation", None, default_n, "")]

    all_statements = []
    counter = 1
    n_buckets = len(retrieved_texts) * len(generation_specs)
    bucket_idx = 0

    for key, texts_df in retrieved_texts.items():
        if isinstance(key, tuple):
            group, cat_id = key
        else:
            cat_id, group = key, None

        subcat = subcategories[cat_id]
        texts = texts_df["text"].tolist()

        source_ids = texts_df["id"].tolist()[:100]
        for template_name, statement_type, n_statements, examples in generation_specs:
            bucket_idx += 1
            label = statement_type or "statement"
            logger.info(
                "[%d/%d] Generating %d %s(s) for subcategory=%s%s",
                bucket_idx,
                n_buckets,
                n_statements,
                label,
                subcat.get("name", cat_id),
                f", group={group}" if group else "",
            )

            sections = _get_prompt_sections(template_name)
            system_prompt = sections.get("system", "")
            prompt = build_generation_prompt(
                template_name=template_name,
                subcategory=subcat,
                retrieved_texts=texts,
                n_statements=n_statements,
                output_language=config.generation.output_language,
                group=group,
                group_instruction_template=config.generation.group_instruction_template,
                examples=examples,
            )

            raw_statements = _call_llm_with_retry(
                client=client,
                model=config.generation.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=config.generation.temperature,
                max_retries=config.generation.max_retries,
                retry_delay=config.generation.retry_delay,
            )

            n_before = len(all_statements)
            for stmt in raw_statements:
                stmt_text = stmt.get("text", stmt) if isinstance(stmt, dict) else str(stmt)
                record = {
                    "id": f"{config.topic.id}_{counter:03d}",
                    "text": stmt_text,
                    "subcategory": subcat["name"],
                    "subcategory_id": cat_id,
                    "group": group,
                    "source_paragraphs": source_ids[:10],
                }
                if statement_type is not None:
                    record["statement_type"] = statement_type
                all_statements.append(record)
                counter += 1
            logger.info(
                "  -> got %d %s(s) (total so far: %d)",
                len(all_statements) - n_before,
                label,
                len(all_statements),
            )

    logger.info(f"Generated {len(all_statements)} statements")
    return all_statements


def _extract_message_content(response) -> str | None:
    """Pull text out of an OpenAI-style chat-completions response.

    Handles three quirks observed in the wild (OpenRouter especially):
      - `message.content` is None but `message.reasoning_content` carries the
        actual reply (DeepSeek-V4-Pro, R1-style models).
      - `message.refusal` is populated and `content` is None.
      - The response was truncated and only contains tool/finish metadata.
    """
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError):
        return None

    message = getattr(choice, "message", None)
    if message is None:
        return None

    content = getattr(message, "content", None)
    if content:
        return content

    for fallback_attr in ("reasoning_content", "reasoning", "refusal"):
        val = getattr(message, fallback_attr, None)
        if val:
            return val

    return None


def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[dict]:
    """Call the LLM up to `max_retries` times.

    Never raises: any unrecoverable error (network, parse, empty content) is
    logged and an empty list returned, so one bad bucket cannot kill an
    otherwise-successful pipeline run.
    """
    last_error: str | None = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2000,
            )

            content = _extract_message_content(response)
            if not content:
                finish_reason = None
                try:
                    finish_reason = response.choices[0].finish_reason
                except Exception:
                    pass
                last_error = (
                    f"empty message.content (finish_reason={finish_reason!r})"
                )
                logger.warning(
                    "LLM returned empty content on attempt %d (finish_reason=%s)",
                    attempt + 1,
                    finish_reason,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                continue

            return _parse_json_response(content.strip())

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"LLM call error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

    logger.error(
        "Giving up on this bucket after %d attempts (last error: %s)",
        max_retries,
        last_error,
    )
    return []


def _parse_json_response(content: str) -> list[dict]:
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()
    parsed = json.loads(content)

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("statements", "considerations", "data", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return [parsed]
    return [{"text": str(parsed)}]
