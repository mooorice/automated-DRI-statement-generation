"""Iterative LLM-driven prompt self-improvement for statement generation."""

import json
import logging
from pathlib import Path

from openai import OpenAI

from autoconcourse.config import PipelineConfig
from autoconcourse.generation import create_llm_client, _call_llm_with_retry
from autoconcourse.prompts import load_prompt_sections

logger = logging.getLogger("autoconcourse.prompt_optimization")


def optimize_prompt(
    current_template: str,
    sample_statements: list[dict],
    config: PipelineConfig,
    client: OpenAI | None = None,
    iterations: int = 3,
    optimization_model: str | None = None,
) -> dict:
    """Iteratively improve a prompt template using LLM self-critique.

    Each iteration:
      1. Critiques current outputs against quality criteria
      2. Generates an improved prompt template
      3. Optionally re-generates statements with the improved prompt

    Args:
        current_template: The starting prompt template text
        sample_statements: Sample outputs from the current template
        config: Pipeline configuration
        client: LLM client (created if None)
        iterations: Number of improvement iterations
        optimization_model: Model to use for optimization (defaults to config)

    Returns:
        Dict with keys: final_template, history (list of iteration results)
    """
    if client is None:
        client = create_llm_client(config)

    model = optimization_model or config.generation.model

    sections = load_prompt_sections("prompt_optimization")
    system_prompt = sections.get("system", "")
    user_template = sections.get("user", "")

    history = []
    template = current_template

    for i in range(iterations):
        logger.info(f"Prompt optimization iteration {i + 1}/{iterations}")

        statements_text = "\n".join(
            f"[{j+1}] {s['text']}" for j, s in enumerate(sample_statements[:30])
        )

        user_prompt = user_template.format(
            current_template=template,
            sample_statements=statements_text,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=3000,
            )

            content = response.choices[0].message.content.strip()
            result = _parse_optimization_response(content)

            history.append({
                "iteration": i + 1,
                "analysis": result.get("analysis", ""),
                "improved_template": result.get("improved_template", template),
            })

            if result.get("improved_template"):
                template = result["improved_template"]
                logger.info(f"  Analysis: {result.get('analysis', '')[:200]}")
            else:
                logger.warning(f"  No improved template in response, keeping current")

        except Exception as e:
            logger.error(f"  Optimization iteration {i + 1} failed: {e}")
            history.append({
                "iteration": i + 1,
                "error": str(e),
            })

    return {
        "final_template": template,
        "iterations": len(history),
        "history": history,
    }


def _parse_optimization_response(content: str) -> dict:
    """Parse the optimization LLM response, handling markdown code blocks."""
    import re
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"analysis": content, "improved_template": ""}


def run_optimization_pipeline(
    config: PipelineConfig,
    statements: list[dict],
    output_dir: str | Path | None = None,
) -> dict:
    """Run prompt optimization and save results.

    Args:
        config: Pipeline configuration
        statements: Existing generated statements to optimize against
        output_dir: Where to save optimization results (defaults to outputs/{topic}/optimization/)

    Returns:
        Optimization results dict
    """
    sections = load_prompt_sections("q-concourse_generation")
    current_template = sections.get("user", "")

    opt_cfg = config.generation.prompt_optimization
    iterations = opt_cfg.iterations
    opt_model = opt_cfg.optimization_model

    result = optimize_prompt(
        current_template=current_template,
        sample_statements=statements,
        config=config,
        iterations=iterations,
        optimization_model=opt_model,
    )

    if output_dir is None:
        output_dir = config.output_dir / "optimization"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "optimization_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    template_path = output_dir / "optimized_template.txt"
    template_path.write_text(result["final_template"], encoding="utf-8")

    logger.info(
        f"Optimization complete: {result['iterations']} iterations, "
        f"results saved to {output_dir}"
    )
    return result
