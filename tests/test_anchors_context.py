"""Anchor context loading: context_folder + context_files, strict mode.

Mirrors the strict-mode design from test_keywords / test_subcategories.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoconcourse import anchors as ac
from autoconcourse.config import load_config


def _write_config(tmp_path: Path, *, anchors_block: str) -> Path:
    """Minimal config; caller provides the full anchors block."""
    sub_path = tmp_path / "subs.json"
    sub_path.write_text(json.dumps([
        {"id": 0, "name": "S1", "description": "d1", "anchor_prompt": "p1"},
    ]))

    cfg = f"""
topic:
  name: "Test topic"
  id: "test"
  languages: ["en"]
  description: "A rich description"

llm:
  provider: "openrouter"
  default_model: "stub-model"

corpus:
  adapter: "csv"
  path: "{tmp_path}/missing.csv"

{anchors_block.replace("__SUB__", str(sub_path))}
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg)
    return cfg_path


def _stub_client(captured: dict):
    """Capture every user prompt sent to chat.completions.create."""
    client = MagicMock()

    def _create(model, messages, **kwargs):
        captured.setdefault("user_prompts", []).append(
            next(m["content"] for m in messages if m["role"] == "user")
        )
        response = MagicMock()
        response.choices[0].message.content = "Generated anchor text."
        return response

    client.chat.completions.create = _create
    return client


def test_context_folder_auto_loads_all_files(tmp_path):
    folder = tmp_path / "ctx"
    folder.mkdir()
    (folder / "a.txt").write_text("Alpha reference notes.")
    (folder / "b.md").write_text("Bravo reference notes.")
    (folder / "ignored.bin").write_bytes(b"\x00\x01")  # unsupported ext

    cfg_path = _write_config(tmp_path, anchors_block=f"""\
anchors:
  enabled: true
  method: "llm_generated"
  subcategories_file: "__SUB__"
  context_folder: "{folder}"
  languages: ["en"]
""")
    config = load_config(cfg_path)
    assert config.anchors.context_folder == str(folder)

    captured = {}
    df = ac.generate_anchors(config, client=_stub_client(captured))

    assert len(df) == 1
    prompt = captured["user_prompts"][0]
    assert "Alpha reference notes" in prompt
    assert "Bravo reference notes" in prompt


def test_context_files_stack_on_context_folder(tmp_path):
    folder = tmp_path / "ctx"
    folder.mkdir()
    (folder / "a.txt").write_text("From folder.")
    extra = tmp_path / "extra.md"
    extra.write_text("From explicit file.")

    cfg_path = _write_config(tmp_path, anchors_block=f"""\
anchors:
  enabled: true
  method: "llm_generated"
  subcategories_file: "__SUB__"
  context_folder: "{folder}"
  context_files: ["{extra}"]
  languages: ["en"]
""")
    config = load_config(cfg_path)
    captured = {}
    ac.generate_anchors(config, client=_stub_client(captured))

    prompt = captured["user_prompts"][0]
    assert "From folder" in prompt
    assert "From explicit file" in prompt


def test_missing_context_file_is_hard_error(tmp_path):
    cfg_path = _write_config(tmp_path, anchors_block="""\
anchors:
  enabled: true
  method: "llm_generated"
  subcategories_file: "__SUB__"
  context_files: ["/nonexistent/path.txt"]
  languages: ["en"]
""")
    config = load_config(cfg_path)
    with pytest.raises(FileNotFoundError, match="anchors context files not found"):
        ac.generate_anchors(config, client=_stub_client({}))


def test_missing_context_folder_is_hard_error(tmp_path):
    cfg_path = _write_config(tmp_path, anchors_block="""\
anchors:
  enabled: true
  method: "llm_generated"
  subcategories_file: "__SUB__"
  context_folder: "/nonexistent/folder"
  languages: ["en"]
""")
    config = load_config(cfg_path)
    with pytest.raises(FileNotFoundError):
        ac.generate_anchors(config, client=_stub_client({}))


def test_no_context_yields_empty_section(tmp_path):
    """No context is allowed (anchors fall back to subcategory name only)."""
    cfg_path = _write_config(tmp_path, anchors_block="""\
anchors:
  enabled: true
  method: "llm_generated"
  subcategories_file: "__SUB__"
  languages: ["en"]
""")
    config = load_config(cfg_path)
    captured = {}
    df = ac.generate_anchors(config, client=_stub_client(captured))
    assert len(df) == 1
    prompt = captured["user_prompts"][0]
    assert "Reference texts" not in prompt


def test_unsupported_extensions_are_skipped(tmp_path):
    folder = tmp_path / "ctx"
    folder.mkdir()
    (folder / "keep.txt").write_text("KEEPME content.")
    (folder / "skip.png").write_bytes(b"PNGDATA")

    cfg_path = _write_config(tmp_path, anchors_block=f"""\
anchors:
  enabled: true
  method: "llm_generated"
  subcategories_file: "__SUB__"
  context_folder: "{folder}"
  languages: ["en"]
""")
    config = load_config(cfg_path)
    captured = {}
    ac.generate_anchors(config, client=_stub_client(captured))
    prompt = captured["user_prompts"][0]
    assert "KEEPME content" in prompt
    assert "PNGDATA" not in prompt


def test_llm_generated_saves_to_configured_anchors_file(tmp_path):
    anchors_file = tmp_path / "project" / "anchors.json"
    cfg_path = _write_config(tmp_path, anchors_block=f"""\
anchors:
  enabled: true
  method: "llm_generated"
  anchors_file: "{anchors_file}"
  subcategories_file: "__SUB__"
  languages: ["en"]
""")
    config = load_config(cfg_path)

    df = ac.load_or_generate_anchors(config, client=_stub_client({}))

    assert len(df) == 1
    assert anchors_file.exists()
    assert not (config.cache_dir / "anchors.csv").exists()


def test_llm_generated_reruns_when_cached_anchors_file_is_empty(tmp_path):
    anchors_file = tmp_path / "project" / "anchors.json"
    anchors_file.parent.mkdir(parents=True, exist_ok=True)
    anchors_file.write_text("[]")

    cfg_path = _write_config(tmp_path, anchors_block=f"""\
anchors:
  enabled: true
  method: "llm_generated"
  anchors_file: "{anchors_file}"
  subcategories_file: "__SUB__"
  languages: ["en"]
""")
    config = load_config(cfg_path)
    captured = {}

    df = ac.load_or_generate_anchors(config, client=_stub_client(captured))

    assert len(df) == 1
    assert len(captured.get("user_prompts", [])) == 1
    reloaded = ac.load_anchors(anchors_file)
    assert len(reloaded) == 1
