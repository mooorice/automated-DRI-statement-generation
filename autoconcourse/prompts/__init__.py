"""Prompt template loading utilities."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without extension)."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


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
