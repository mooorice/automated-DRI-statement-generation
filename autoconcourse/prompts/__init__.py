"""Prompt template loading utilities."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
EXAMPLES_DIR = PROMPTS_DIR / "examples"


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without extension)."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_examples(
    name: str | None = None,
    path: str | Path | None = None,
    max_examples: int | None = None,
) -> str:
    """Load reference examples for a prompt and format them as a bulleted list.

    Resolution order:
      1. Explicit `path` (absolute or relative to cwd) — used if provided.
      2. Packaged file at `prompts/examples/{name}.txt`.

    Lines are stripped; empty lines and an optional leading header line that
    does not look like an example are skipped. If `max_examples` is given, the
    list is truncated. Any literal `{` / `}` are escaped so the result can be
    safely passed through `str.format()`.
    """
    if path is not None:
        file_path = Path(path)
    elif name is not None:
        file_path = EXAMPLES_DIR / f"{name}.txt"
    else:
        raise ValueError("load_examples() requires either `name` or `path`.")

    if not file_path.exists():
        raise FileNotFoundError(f"Examples file not found: {file_path}")

    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    for i, line in enumerate(raw_lines):
        stripped = line.strip().strip('"').strip()
        if not stripped:
            continue
        # Drop a single leading header line that clearly is not an example
        # (e.g. "dri statements", "policy options").
        if i == 0 and len(stripped.split()) <= 3 and stripped.islower():
            continue
        cleaned.append(stripped)

    if max_examples is not None and max_examples >= 0:
        cleaned = cleaned[:max_examples]

    formatted = "\n".join(f"- {line}" for line in cleaned)
    return formatted.replace("{", "{{").replace("}", "}}")


def load_prompt_sections(name: str) -> dict[str, str]:
    """Load a prompt template and split into [SYSTEM] and [USER] sections.

    Returns dict with keys 'system' and/or 'user'.
    """
    raw = load_prompt(name)
    sections = {}
    current_key = None
    current_lines = []

    for line in raw.split("\n"):
        stripped = line.strip().upper()
        if stripped in ("[SYSTEM]", "[USER]"):
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = stripped.strip("[]").lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections
