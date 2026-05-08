"""LLM statement generation via OpenRouter (or any OpenAI-compatible API)."""

import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from .config import PipelineConfig
from .prompts import load_prompt_sections

logger = logging.getLogger("autoconcourse.generation")

_PROMPT_CACHE: dict[str, dict[str, str]] = {}


def _get_prompt_sections(template_name: str) -> dict[str, str]:
    """Load and cache prompt template sections."""
    if template_name not in _PROMPT_CACHE:
        _PROMPT_CACHE[template_name] = load_prompt_sections(template_name)
    return _PROMPT_CACHE[template_name]


def create_llm_client(config: PipelineConfig) -> OpenAI:
    """Create an OpenAI-compatible client from config.

    Resolution order for API key:
      1. config.llm.api_key_env (name of env var)
      2. OPENROUTER_API_KEY env var
      3. OPENAI_API_KEY env var

    Resolution order for base_url:
      1. config.llm.base_url (explicit)
      2. Auto-detect from model name (provider/model -> OpenRouter)
      3. None (default OpenAI)
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key_env = config.llm.api_key_env
    if api_key_env:
        api_key = os.getenv(api_key_env)
    else:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "No API key found. Set api_key_env in config, or "
            "OPENROUTER_API_KEY / OPENAI_API_KEY in .env"
        )

    base_url = config.llm.base_url
    if base_url is None:
        model = config.generation.model
        if "/" in model and not model.startswith("gpt"):
            base_url = "https://openrouter.ai/api/v1"

    extra_headers = config.llm.extra_headers or {}

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=extra_headers or None,
    )

    logger.info(f"Created LLM client (base_url={base_url or 'default'})")
    return client


def build_generation_prompt(
    subcategory: dict,
    retrieved_texts: list[str],
    perspective_group: str | None = None,
    n_statements: int = 5,
    output_language: str = "en",
    statement_type: str = "consideration",
) -> str:
    """Build the user prompt for statement generation from template."""
    template_name = (
        "policy_option_generation" if statement_type == "policy_option"
        else "statement_generation"
    )
    sections = _get_prompt_sections(template_name)
    user_template = sections.get("user", "")

    topic_focus = subcategory.get("topic_focus", subcategory["name"])

    perspective_instruction = ""
    if perspective_group:
        perspective_instruction = (
            f"\nThe media paragraphs below come from {perspective_group}-leaning media outlets. "
            "Ensure the statements represent the perspectives found in these sources."
        )

    paragraph_lines = []
    for i, text in enumerate(retrieved_texts[:100], 1):
        paragraph_lines.append(f"[{i}] {text}")

    return user_template.format(
        n_statements=n_statements,
        topic_focus=topic_focus,
        perspective_instruction=perspective_instruction,
        output_language=output_language,
        paragraphs="\n".join(paragraph_lines),
    )


def generate_statements(
    retrieved_texts: dict[int, pd.DataFrame] | dict[tuple, pd.DataFrame],
    subcategories: list[dict],
    config: PipelineConfig,
    client: OpenAI | None = None,
    statement_type: str = "consideration",
) -> list[dict]:
    """Generate statements for all subcategories (and optionally perspective groups).

    Args:
        retrieved_texts: Output from retrieval step.
            - If dict[int, DataFrame]: keys are subcategory ids
            - If dict[tuple, DataFrame]: keys are (group_name, subcategory_id)
        subcategories: List of subcategory definitions
        config: Pipeline configuration
        client: LLM client (created if None)
        statement_type: "consideration" or "policy_option"

    Returns:
        List of statement dicts with id, text, subcategory, perspective_group, source_paragraphs
    """
    if client is None:
        client = create_llm_client(config)

    template_name = (
        "policy_option_generation" if statement_type == "policy_option"
        else "statement_generation"
    )
    sections = _get_prompt_sections(template_name)
    system_prompt = sections.get("system", "")

    all_statements = []
    statement_counter = 1

    for key, texts_df in retrieved_texts.items():
        if isinstance(key, tuple):
            group_name, cat_id = key
        else:
            cat_id = key
            group_name = None

        subcat = subcategories[cat_id]
        texts = texts_df["text"].tolist()

        prompt = build_generation_prompt(
            subcategory=subcat,
            retrieved_texts=texts,
            perspective_group=group_name,
            n_statements=config.generation.statements_per_group_per_subcategory,
            output_language=config.generation.output_language,
            statement_type=statement_type,
        )

        # Call LLM with retries
        raw_statements = _call_llm_with_retry(
            client=client,
            model=config.generation.model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=config.generation.temperature,
            max_retries=config.generation.max_retries,
            retry_delay=config.generation.retry_delay,
        )

        # Process response into standard format
        source_ids = texts_df["id"].tolist()[:100]
        for stmt in raw_statements:
            stmt_text = stmt.get("text", stmt) if isinstance(stmt, dict) else str(stmt)
            all_statements.append({
                "id": f"{config.topic.id}_{statement_counter:03d}",
                "text": stmt_text,
                "subcategory": subcat["name"],
                "subcategory_id": cat_id,
                "perspective_group": group_name,
                "source_paragraphs": source_ids[:10],
                "type": statement_type,
            })
            statement_counter += 1

    logger.info(f"Generated {len(all_statements)} {statement_type} statements total")
    return all_statements


def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[dict]:
    """Call the LLM and parse JSON response, with retries."""
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

            content = response.choices[0].message.content.strip()
            return _parse_json_response(content)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to parse JSON after {max_retries} attempts")
                return []

        except Exception as e:
            logger.warning(f"LLM call error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise

    return []


def _parse_json_response(content: str) -> list[dict]:
    """Parse LLM response as JSON, handling markdown code blocks."""
    # Strip markdown code blocks
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    parsed = json.loads(content)

    # Handle various response shapes
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict):
        # Try common keys
        for key in ("statements", "considerations", "policy_options", "data", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return [parsed]
    else:
        return [{"text": str(parsed)}]
