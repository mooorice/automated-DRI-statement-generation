"""Tests for the subcategory generation CLI / module. No network calls."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoconcourse import subcategories as sc


# ---------------------------------------------------------------- parser tests

_SAMPLE = json.dumps({
    "subcategories": [
        {"id": 0, "name": "School", "description": "Education.",
         "topic_focus": "school", "anchor_prompt": "Write about school."},
        {"id": 1, "name": "Health", "description": "Healthcare.",
         "topic_focus": "health", "anchor_prompt": "Write about health."},
    ]
})


def test_parse_subcategories_clean_object():
    out = sc._parse_subcategories(_SAMPLE)
    assert len(out) == 2
    assert out[0]["name"] == "School"
    assert out[1]["id"] == 1
    assert {"id", "name", "description", "topic_focus", "anchor_prompt"} <= out[0].keys()


def test_parse_subcategories_strips_markdown_fences():
    out = sc._parse_subcategories(f"```json\n{_SAMPLE}\n```")
    assert len(out) == 2


def test_parse_subcategories_plain_list():
    raw = json.dumps([
        {"name": "A", "description": "d", "topic_focus": "t", "anchor_prompt": "p"},
        {"name": "B", "description": "d", "topic_focus": "t", "anchor_prompt": "p"},
    ])
    out = sc._parse_subcategories(raw)
    assert out[0]["id"] == 0
    assert out[1]["id"] == 1


def test_parse_subcategories_reassigns_ids():
    """Whatever ids the LLM emits get replaced with sequential 0..N-1."""
    raw = json.dumps({
        "subcategories": [
            {"id": 42, "name": "A", "description": "d", "topic_focus": "t", "anchor_prompt": "p"},
            {"id": 99, "name": "B", "description": "d", "topic_focus": "t", "anchor_prompt": "p"},
        ]
    })
    out = sc._parse_subcategories(raw)
    assert [sc_["id"] for sc_ in out] == [0, 1]


def test_parse_subcategories_rejects_missing_name():
    raw = json.dumps({"subcategories": [{"description": "no name"}]})
    with pytest.raises(ValueError, match="no name"):
        sc._parse_subcategories(raw)


def test_parse_subcategories_raises_on_garbage():
    with pytest.raises(ValueError):
        sc._parse_subcategories("not json")


# --------------------------------------------------------- seed-counter heuristic

def test_count_seed_clusters_recognises_cluster_prefix():
    text = """
    Cluster 0, "School": ...
    Cluster 1, "Health": ...
    Cluster 2, "Cloud": ...
    """
    assert sc._count_seed_clusters(text) == 3


def test_count_seed_clusters_recognises_bulleted_clusters():
    text = """
    ● Cluster 0, "A": ...
    ● Cluster 1, "B": ...
    """
    assert sc._count_seed_clusters(text) == 2


def test_count_seed_clusters_recognises_numbered_lines():
    text = """
    1. School
    2. Health
    3. Cloud
    """
    assert sc._count_seed_clusters(text) == 3


def test_count_seed_clusters_handles_bullets():
    text = """
    - First topic
    - Second topic
    - Third topic
    """
    assert sc._count_seed_clusters(text) == 3


def test_count_seed_clusters_empty():
    assert sc._count_seed_clusters("") == 0
    assert sc._count_seed_clusters(None) == 0


# -------------------------------------------------- end-to-end with stub client

def _make_stub_client(canned: str):
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = canned
    client.chat.completions.create.return_value = response
    return client


def test_generate_subcategories_seeded():
    client = _make_stub_client(_SAMPLE)
    out = sc.generate_subcategories(
        topic="AI in education",
        language="en",
        seeds_text='Cluster 0, "School": ...\nCluster 1, "Health": ...',
        client=client,
        model="stub",
    )
    assert len(out) == 2
    # Verify the prompt sent to the LLM mentions seeds and forbids merging.
    args, kwargs = client.chat.completions.create.call_args
    user_msg = kwargs["messages"][1]["content"]
    assert "Seed clusters" in user_msg
    assert "ONE subcategory per seed" in user_msg or "one subcategory per seed" in user_msg.lower()


def test_generate_subcategories_unseeded():
    client = _make_stub_client(_SAMPLE)
    sc.generate_subcategories(
        topic="AI in education",
        language="en",
        count=5,
        client=client,
        model="stub",
    )
    args, kwargs = client.chat.completions.create.call_args
    user_msg = kwargs["messages"][1]["content"]
    assert "Seed clusters" not in user_msg
    assert "5 subcategories" in user_msg


def test_generate_subcategories_falls_back_when_response_format_unsupported():
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = _SAMPLE
    client.chat.completions.create.side_effect = [
        Exception("response_format not supported"),
        response,
    ]
    out = sc.generate_subcategories(topic="t", language="en", count=2, client=client, model="m")
    assert len(out) == 2
    assert client.chat.completions.create.call_count == 2


# -------------------------------------------------------- config + CLI dispatch

def _write_config(tmp_path, *, sc_block: str | None = None,
                  langs: str = '["en"]', anchors_sub_path: str = None) -> Path:
    """Minimal config writer for subcategory CLI tests."""
    sub = anchors_sub_path or str(tmp_path / "anchors_subs.json")
    Path(sub).write_text('[{"id": 0, "name": "Existing"}]')

    cfg = f"""
topic:
  name: "Test topic"
  id: "test"
  languages: {langs}

llm:
  provider: "openrouter"
  default_model: "stub-model"

corpus:
  adapter: "csv"
  path: "{tmp_path}/missing.csv"

anchors:
  enabled: true
  method: "from_file"
  subcategories_file: "{sub}"
  context_files: ["/should/be/ignored.txt"]
"""
    if sc_block:
        cfg += "\n" + sc_block + "\n"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg)
    return cfg_path


def test_subcategories_section_parses_from_yaml(tmp_path):
    """SubcategoriesConfig loads all fields from YAML."""
    from autoconcourse.config import load_config

    block = """\
subcategories:
  topic: "Custom topic for subcats"
  count: 4
  language: "de"
  seeds: |
    Cluster 0, "School": a, b
    Cluster 1, "Health": c, d
  model: "sc-model/x"
  output_file: "sc_out.json"
"""
    cfg_path = _write_config(tmp_path, sc_block=block)
    config = load_config(cfg_path)
    assert config.subcategories.topic == "Custom topic for subcats"
    assert config.subcategories.count == 4
    assert config.subcategories.language == "de"
    assert "Cluster 0" in config.subcategories.seeds
    assert config.subcategories.model == "sc-model/x"
    assert config.subcategories.output_file == "sc_out.json"


def test_subcategories_section_omitted_uses_defaults(tmp_path):
    """No subcategories block → SubcategoriesConfig with built-in defaults."""
    from autoconcourse.config import load_config

    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)
    assert config.subcategories.topic is None
    assert config.subcategories.count == 8
    assert config.subcategories.seeds is None
    assert config.subcategories.seeds_file is None


def _patch_pipeline(monkeypatch, *, capture: dict):
    """Common patching for CLI tests."""
    def fake_generate(**kwargs):
        capture.update(kwargs)
        return [{"id": 0, "name": "X", "description": "d", "topic_focus": "t", "anchor_prompt": "p"}]
    monkeypatch.setattr(sc, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(sc, "generate_subcategories", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")


def test_cli_seed_count_inferred(tmp_path, monkeypatch):
    """When seeds are provided inline, count is inferred from the seed list."""
    block = """\
subcategories:
  seeds: |
    Cluster 0, "School": a, b
    Cluster 1, "Health": c, d
    Cluster 2, "Cloud": e, f
"""
    cfg_path = _write_config(tmp_path, sc_block=block)
    capture = {}
    _patch_pipeline(monkeypatch, capture=capture)
    out = tmp_path / "out.json"
    sc.subcategories_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert capture["count"] == 3
    assert "Cluster 0" in capture["seeds_text"]


def test_cli_seeds_file_loaded(tmp_path, monkeypatch):
    """subcategories.seeds_file is read and passed through."""
    seeds = tmp_path / "seeds.md"
    seeds.write_text('Cluster 0, "A": x\nCluster 1, "B": y')

    block = f"""\
subcategories:
  seeds_file: "{seeds}"
"""
    cfg_path = _write_config(tmp_path, sc_block=block)
    capture = {}
    _patch_pipeline(monkeypatch, capture=capture)
    out = tmp_path / "out.json"
    sc.subcategories_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert "Cluster 0" in capture["seeds_text"]
    assert capture["count"] == 2


def test_cli_strict_mode_ignores_anchors_context(tmp_path, monkeypatch):
    """anchors.context_files MUST NOT leak into subcategory generation."""
    cfg_path = _write_config(tmp_path)  # no subcategories block
    capture = {}
    _patch_pipeline(monkeypatch, capture=capture)
    out = tmp_path / "out.json"
    sc.subcategories_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert capture["context_texts"] == []


def test_cli_context_folder_loads_files(tmp_path, monkeypatch):
    """subcategories.context_folder recursively loads supported files."""
    folder = tmp_path / "ctx"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("Doc A.")
    (folder / "sub" / "b.md").write_text("Doc B.")
    (folder / "ignore.png").write_bytes(b"\x89PNG")

    block = f"""\
subcategories:
  context_folder: "{folder}"
"""
    cfg_path = _write_config(tmp_path, sc_block=block)
    capture = {}
    _patch_pipeline(monkeypatch, capture=capture)
    out = tmp_path / "out.json"
    sc.subcategories_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    joined = "\n".join(capture["context_texts"])
    assert "Doc A." in joined
    assert "Doc B." in joined
    assert "ignore" not in joined.lower()


def test_cli_output_defaults_to_anchors_subcategories_file(tmp_path, monkeypatch):
    """When subcategories.output_file is unset, write to anchors.subcategories_file."""
    target = tmp_path / "target_subs.json"
    cfg_path = _write_config(tmp_path, anchors_sub_path=str(target))

    # Real client + real save (just stub the LLM call)
    captured_client = _make_stub_client(_SAMPLE)
    monkeypatch.setattr(sc, "create_llm_client", lambda config: captured_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    sc.subcategories_main(["--config", str(cfg_path), "--log-level", "WARNING"])
    assert target.exists()
    data = json.loads(target.read_text())
    assert len(data) == 2
    assert data[0]["id"] == 0


def test_cli_topic_override_chain(tmp_path, monkeypatch):
    """--topic > subcategories.topic > topic.description > topic.name."""
    cfg_path = _write_config(tmp_path, sc_block='subcategories:\n  topic: "from-config"')
    # Add a description, so we can verify subcategories.topic still wins
    cfg_text = cfg_path.read_text().replace(
        'name: "Test topic"',
        'name: "Test topic"\n  description: "from-description"',
    )
    cfg_path.write_text(cfg_text)

    capture = {}
    _patch_pipeline(monkeypatch, capture=capture)
    out = tmp_path / "out.json"

    # subcategories.topic wins over both topic.description and topic.name
    sc.subcategories_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert capture["topic"] == "from-config"

    capture.clear()
    # CLI --topic wins over everything
    sc.subcategories_main([
        "--config", str(cfg_path),
        "--topic", "from-cli",
        "--output", str(out),
        "--log-level", "WARNING",
    ])
    assert capture["topic"] == "from-cli"


def test_subcategories_topic_description_fallback(tmp_path, monkeypatch):
    """When neither --topic nor subcategories.topic is set, topic.description is used."""
    cfg_path = _write_config(tmp_path)  # no subcategories block
    cfg_text = cfg_path.read_text().replace(
        'name: "Test topic"',
        'name: "Test topic"\n  description: "Project-level description"',
    )
    cfg_path.write_text(cfg_text)

    capture = {}
    _patch_pipeline(monkeypatch, capture=capture)
    out = tmp_path / "out.json"
    sc.subcategories_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert capture["topic"] == "Project-level description"


def test_cli_missing_seeds_file_is_a_hard_error(tmp_path, monkeypatch):
    block = """\
subcategories:
  seeds_file: "/nonexistent.md"
"""
    cfg_path = _write_config(tmp_path, sc_block=block)
    capture = {}
    _patch_pipeline(monkeypatch, capture=capture)
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        sc.subcategories_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
