"""Tests for the keyword generation CLI / module. No network calls."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoconcourse import keywords as kw


# ---------------------------------------------------------------- parser tests

def test_parse_keywords_plain_json_object():
    raw = '{"keywords": ["alpha", "beta", "gamma"]}'
    assert kw._parse_keywords(raw) == ["alpha", "beta", "gamma"]


def test_parse_keywords_plain_json_list():
    raw = '["alpha", "beta"]'
    assert kw._parse_keywords(raw) == ["alpha", "beta"]


def test_parse_keywords_strips_markdown_fences():
    raw = '```json\n{"keywords": ["alpha", "beta"]}\n```'
    assert kw._parse_keywords(raw) == ["alpha", "beta"]


def test_parse_keywords_extracts_list_from_prose():
    raw = 'Sure! Here you go:\n["alpha", "beta", "gamma"]\nHope that helps.'
    assert kw._parse_keywords(raw) == ["alpha", "beta", "gamma"]


def test_parse_keywords_dedups_case_insensitively():
    raw = '{"keywords": ["Alpha", "alpha", "BETA", "beta", "Gamma"]}'
    assert kw._parse_keywords(raw) == ["Alpha", "BETA", "Gamma"]


def test_parse_keywords_picks_first_list_value_in_object():
    raw = '{"language": "en", "terms": ["alpha", "beta"]}'
    assert kw._parse_keywords(raw) == ["alpha", "beta"]


def test_parse_keywords_raises_on_garbage():
    with pytest.raises(ValueError):
        kw._parse_keywords("not json at all")


# ----------------------------------------------------------- prompt formatting

def test_format_subcategories_empty():
    assert kw._format_subcategories(None) == ""
    assert kw._format_subcategories([]) == ""


def test_format_subcategories_rendered():
    subcats = [
        {"name": "Climate science", "description": "scientific consensus"},
        {"name": "Energy transition", "topic_focus": "renewables"},
        {"name": "No-desc"},
    ]
    out = kw._format_subcategories(subcats)
    assert "Subcategories" in out
    assert "- Climate science: scientific consensus" in out
    assert "- Energy transition: renewables" in out
    assert "- No-desc" in out


def test_format_context_truncates():
    long_text = "x" * 50_000
    out = kw._format_context([long_text], max_chars=1000)
    assert "[... truncated]" in out
    assert len(out) < 2000


def test_build_keyword_prompt_renders_all_placeholders():
    prompt = kw._build_keyword_prompt(
        topic="Health policy",
        subcategories=[{"name": "Cost", "description": "Costs of care"}],
        context_texts=["Some reference text."],
        language="de",
        n=42,
    )
    assert "Health policy" in prompt
    assert "Cost: Costs of care" in prompt
    assert "Some reference text." in prompt
    assert "German" in prompt
    assert "42" in prompt


# -------------------------------------------------- end-to-end with stub client

def _make_stub_client(canned: str):
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = canned
    client.chat.completions.create.return_value = response
    return client


def test_generate_keywords_single_language():
    client = _make_stub_client('{"keywords": ["health", "insurance", "premium"]}')
    out = kw.generate_keywords(
        topic="Health costs",
        languages=["en"],
        client=client,
        model="stub-model",
    )
    assert out == {"en": ["health", "insurance", "premium"]}
    client.chat.completions.create.assert_called_once()


def test_generate_keywords_multilingual_calls_once_per_language():
    client = _make_stub_client('{"keywords": ["a", "b"]}')
    out = kw.generate_keywords(
        topic="Topic",
        languages=["en", "de", "fr"],
        client=client,
        model="stub-model",
    )
    assert set(out.keys()) == {"en", "de", "fr"}
    assert client.chat.completions.create.call_count == 3


def test_generate_keywords_falls_back_when_response_format_unsupported():
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = '{"keywords": ["x"]}'

    # First call (with response_format) raises; second call returns OK.
    client.chat.completions.create.side_effect = [
        Exception("response_format not supported by model"),
        response,
    ]

    out = kw.generate_keywords(topic="t", languages=["en"], client=client, model="m")
    assert out == {"en": ["x"]}
    assert client.chat.completions.create.call_count == 2


# ----------------------------------------------------------------- save format

def test_save_keywords_matches_corpus_load_keywords_format(tmp_path):
    """`save_keywords` output must round-trip through `corpus.load_keywords`."""
    from autoconcourse.corpus import load_keywords

    path = tmp_path / "keywords.json"
    kw.save_keywords({"en": ["alpha", "beta"], "de": ["gamma"]}, path)

    flat = load_keywords(path)
    assert set(flat) == {"alpha", "beta", "gamma"}

    # Also valid JSON shape on disk.
    with open(path) as f:
        data = json.load(f)
    assert data == {"en": ["alpha", "beta"], "de": ["gamma"]}


# ----------------------------------------------------- CLI dispatch (no network)

def test_cli_standalone_writes_keywords(tmp_path, monkeypatch):
    """Run `keywords_main` standalone (no --config) with a stub LLM client."""
    captured = _make_stub_client('{"keywords": ["alpha", "beta", "gamma"]}')

    monkeypatch.setattr(kw, "create_llm_client", lambda config: captured)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    output = tmp_path / "keywords.json"
    kw.keywords_main([
        "--topic", "Test topic",
        "--language", "en",
        "--output", str(output),
        "--per-language", "10",
        "--log-level", "WARNING",
    ])

    assert output.exists()
    data = json.loads(output.read_text())
    assert data == {"en": ["alpha", "beta", "gamma"]}


def test_cli_uses_config_defaults(tmp_path, monkeypatch):
    """When --config is given, languages/output should fall through from it."""
    # Minimal config pointing at an existing example.
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "projects" / "example" / "monolingual" / "config.yaml"
    assert config_path.exists(), "monolingual example config missing"

    captured = _make_stub_client('{"keywords": ["one", "two"]}')
    monkeypatch.setattr(kw, "create_llm_client", lambda config: captured)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    output = tmp_path / "out.json"
    kw.keywords_main([
        "--config", str(config_path),
        "--output", str(output),
        "--log-level", "WARNING",
    ])

    assert output.exists()
    data = json.loads(output.read_text())
    # monolingual example has topic.languages == ["en"]
    assert list(data.keys()) == ["en"]
    assert data["en"] == ["one", "two"]


# ---------------------------------------------------- keywords config section

def _write_config(tmp_path, *, keywords_block: str | None = None,
                  langs: str = '["en"]', sub_block: bool = True) -> Path:
    """Write a minimal config.yaml + supporting files into tmp_path."""
    sub = ""
    if sub_block:
        sub_path = tmp_path / "subs.json"
        sub_path.write_text('[{"id": 0, "name": "S1", "description": "d1"}]')
        sub = f'\n  subcategories_file: "{sub_path}"'

    ctx_path = tmp_path / "anchor_ctx.txt"
    ctx_path.write_text("Anchor-only reference document — climate/energy framing.")

    cfg = f"""
topic:
  name: "Test topic"
  id: "test"
  languages: {langs}

llm:
  provider: "openrouter"

corpus:
  adapter: "csv"
  path: "{tmp_path}/missing.csv"
  filtering:
    min_length: 20
    keywords_file: "{tmp_path}/from_corpus.json"

anchors:
  enabled: true
  method: "from_file"{sub}
  context_files: ["{ctx_path}"]
  model: "anchor-model/x"
"""
    if keywords_block:
        cfg += "\n" + keywords_block + "\n"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg)
    return cfg_path


def test_keywords_section_loaded_into_config(tmp_path):
    """KeywordsConfig parses from YAML with all fields populated."""
    from autoconcourse.config import load_config

    block = """\
keywords:
  per_language: 25
  languages: ["en", "de"]
  context_files: ["ctx.txt"]
  model: "kw-model/x"
  output_file: "kw.json"
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)
    config = load_config(cfg_path)
    assert config.keywords.per_language == 25
    assert config.keywords.languages == ["en", "de"]
    assert config.keywords.context_files == ["ctx.txt"]
    assert config.keywords.model == "kw-model/x"
    assert config.keywords.output_file == "kw.json"


def test_keywords_section_omitted_uses_defaults(tmp_path):
    """No keywords block in YAML → KeywordsConfig with built-in defaults."""
    from autoconcourse.config import load_config

    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)
    assert config.keywords.per_language == 50
    assert config.keywords.languages == []
    assert config.keywords.context_files == []
    assert config.keywords.output_file is None


def test_cli_resolves_per_language_from_keywords_section(tmp_path, monkeypatch):
    """config.keywords.per_language wins when --per-language is absent."""
    block = """\
keywords:
  per_language: 7
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert seen["per_language"] == 7


def test_cli_keywords_languages_override_topic_languages(tmp_path, monkeypatch):
    """config.keywords.languages overrides topic.languages when set."""
    block = """\
keywords:
  languages: ["fr", "it"]
"""
    cfg_path = _write_config(tmp_path, keywords_block=block, langs='["en"]')

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert seen["languages"] == ["fr", "it"]


def test_strict_mode_does_not_fall_back_to_anchors_context(tmp_path, monkeypatch):
    """anchors.context_files MUST NOT leak into keyword generation."""
    # _write_config sets anchors.context_files to climate-ish content.
    cfg_path = _write_config(tmp_path)  # no keywords block at all
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert seen["context_texts"] == []


def test_strict_mode_does_not_fall_back_to_anchors_subcategories(tmp_path, monkeypatch):
    """anchors.subcategories_file MUST NOT be used when keywords.subcategories_file is unset."""
    cfg_path = _write_config(tmp_path)  # has anchors.subcategories_file, no keywords block
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert seen["subcategories"] is None


def test_keywords_context_files_only_used_when_explicit(tmp_path, monkeypatch):
    """Only keywords.context_files content is loaded; anchors.context_files is ignored."""
    extra = tmp_path / "extra.txt"
    extra.write_text("Extra reference material.")

    block = f"""\
keywords:
  context_files: ["{extra}"]
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert len(seen["context_texts"]) == 1
    assert "Extra reference material" in seen["context_texts"][0]


# --------------------------------------------------- keywords.topic + context_folder

def test_topic_description_used_when_no_keywords_topic(tmp_path, monkeypatch):
    """topic.description is the project-level fallback before topic.name."""
    cfg_path = _write_config(tmp_path)  # no keywords block, no description
    cfg_text = cfg_path.read_text().replace(
        'name: "Test topic"',
        'name: "Test topic"\n  description: "Long rich description of the topic"',
    )
    cfg_path.write_text(cfg_text)

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert seen["topic"] == "Long rich description of the topic"


def test_keywords_topic_overrides_topic_description(tmp_path, monkeypatch):
    """keywords.topic wins over topic.description."""
    cfg_path = _write_config(tmp_path, keywords_block='keywords:\n  topic: "Section-specific"')
    cfg_text = cfg_path.read_text().replace(
        'name: "Test topic"',
        'name: "Test topic"\n  description: "Project-level description"',
    )
    cfg_path.write_text(cfg_text)

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert seen["topic"] == "Section-specific"


def test_keywords_topic_overrides_topic_name(tmp_path, monkeypatch):
    """keywords.topic is used in the prompt instead of topic.name."""
    block = """\
keywords:
  topic: "Participatory tech assessment of AI in education and work"
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])
    assert seen["topic"] == "Participatory tech assessment of AI in education and work"


def test_cli_topic_flag_beats_keywords_topic(tmp_path, monkeypatch):
    """--topic CLI flag wins over keywords.topic."""
    block = """\
keywords:
  topic: "Topic from config"
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main([
        "--config", str(cfg_path),
        "--topic", "Topic from CLI",
        "--output", str(out),
        "--log-level", "WARNING",
    ])
    assert seen["topic"] == "Topic from CLI"


def test_context_folder_loads_all_supported_files(tmp_path, monkeypatch):
    """keywords.context_folder auto-loads every supported file inside, recursively."""
    folder = tmp_path / "ctx_folder"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("First doc.")
    (folder / "b.md").write_text("Second doc.")
    (folder / "sub" / "c.txt").write_text("Nested doc.")
    (folder / "ignore.png").write_bytes(b"\x89PNG")           # unsupported ext
    (folder / ".hidden.txt").write_text("Should be skipped.")  # dotfile

    block = f"""\
keywords:
  context_folder: "{folder}"
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])

    texts = seen["context_texts"]
    assert len(texts) == 3
    joined = "\n".join(texts)
    assert "First doc." in joined
    assert "Second doc." in joined
    assert "Nested doc." in joined
    assert "ignore" not in joined.lower() or "png" not in joined.lower()
    assert "Should be skipped" not in joined


def test_context_folder_and_files_stack(tmp_path, monkeypatch):
    """keywords.context_folder + keywords.context_files are combined."""
    folder = tmp_path / "ctx_folder"
    folder.mkdir()
    (folder / "from_folder.txt").write_text("From folder.")
    extra = tmp_path / "extra.md"
    extra.write_text("From extra.")

    block = f"""\
keywords:
  context_folder: "{folder}"
  context_files: ["{extra}"]
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])

    joined = "\n".join(seen["context_texts"])
    assert "From folder." in joined
    assert "From extra." in joined


def test_missing_context_file_is_a_hard_error(tmp_path, monkeypatch):
    """Referenced-but-missing context files must fail loudly, not skip silently."""
    block = """\
keywords:
  context_files: ["/nonexistent/path.txt"]
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", lambda **kw_: {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])


def test_missing_context_folder_is_a_hard_error(tmp_path, monkeypatch):
    """Referenced-but-missing context folder must fail loudly."""
    block = """\
keywords:
  context_folder: "/nonexistent/folder"
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", lambda **kw_: {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    with pytest.raises((FileNotFoundError, SystemExit)):
        kw.keywords_main(["--config", str(cfg_path), "--output", str(out), "--log-level", "WARNING"])


def test_cli_per_language_flag_beats_config(tmp_path, monkeypatch):
    """--per-language overrides config.keywords.per_language."""
    block = """\
keywords:
  per_language: 7
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)

    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return {lang: ["x"] for lang in kwargs["languages"]}

    monkeypatch.setattr(kw, "create_llm_client", lambda config: MagicMock())
    monkeypatch.setattr(kw, "generate_keywords", fake_generate)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    out = tmp_path / "out.json"
    kw.keywords_main([
        "--config", str(cfg_path),
        "--per-language", "99",
        "--output", str(out),
        "--log-level", "WARNING",
    ])
    assert seen["per_language"] == 99


def test_cli_keywords_output_file_used_when_corpus_filter_missing(tmp_path, monkeypatch):
    """config.keywords.output_file takes precedence over corpus.filtering.keywords_file."""
    explicit = tmp_path / "explicit.json"
    block = f"""\
keywords:
  output_file: "{explicit}"
"""
    cfg_path = _write_config(tmp_path, keywords_block=block)

    captured = _make_stub_client('{"keywords": ["a", "b"]}')
    monkeypatch.setattr(kw, "create_llm_client", lambda config: captured)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    # No --output flag → keywords.output_file should be used
    kw.keywords_main(["--config", str(cfg_path), "--log-level", "WARNING"])
    assert explicit.exists()
    data = json.loads(explicit.read_text())
    assert data == {"en": ["a", "b"]}
