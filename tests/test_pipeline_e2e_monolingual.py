"""End-to-end smoke test: monolingual English corpus, anchors-from-file, mocked LLM."""

import json
from pathlib import Path

import yaml


def _write_synthetic_topic(root: Path) -> Path:
    """Drop a self-contained tiny topic into `root`. Returns the config path."""
    topic = root / "synthetic_mono"
    topic.mkdir(parents=True)

    corpus = topic / "corpus.csv"
    corpus.write_text(
        "id,text,language,source\n"
        "p1,Renewable energy investment hit record highs in 2025 with solar leading the way.,en,A\n"
        "p2,Critics warn carbon taxes hurt low-income households without offsetting refunds.,en,B\n"
        "p3,Community-led adaptation projects show local solutions complement global policy.,en,A\n"
        "p4,Scientists confirm the planet has warmed by 1.1 degrees since pre-industrial times.,en,B\n"
        "p5,Renewables now outpace fossil fuels in new capacity additions in major markets.,en,A\n"
        "p6,Some economists argue aggressive climate policies threaten energy security.,en,B\n",
        encoding="utf-8",
    )

    (topic / "subcategories.json").write_text(json.dumps([
        {
            "id": 0,
            "name": "Climate science",
            "description": "Scientific evidence and warming trajectory.",
            "topic_focus": "climate science and warming",
            "anchor_prompt": "Write a paragraph on climate science evidence.",
        },
        {
            "id": 1,
            "name": "Energy transition",
            "description": "Renewables vs fossil fuels and policy tradeoffs.",
            "topic_focus": "the energy transition and renewables",
            "anchor_prompt": "Write a paragraph on the energy transition.",
        },
    ]), encoding="utf-8")

    (topic / "anchors.json").write_text(json.dumps([
        {"category": 0, "language": "en", "subcategory_name": "Climate science",
         "text": "Scientists have established that global temperatures rose by 1.1 degrees Celsius since pre-industrial times due to greenhouse gas emissions, and warming threatens 1.5 degrees within a decade."},
        {"category": 1, "language": "en", "subcategory_name": "Energy transition",
         "text": "The shift from fossil fuels to renewable energy is accelerating, with solar and wind investment outpacing coal, oil, and gas, though grid infrastructure and transition costs remain debated."},
    ]), encoding="utf-8")

    cfg = {
        "topic": {"name": "Synthetic mono", "id": "synthetic_mono", "languages": ["en"]},
        "llm": {"default_model": "stub/model"},
        "corpus": {
            "adapter": "csv",
            "path": str(corpus),
            "text_column": "text",
            "id_column": "id",
            "language_column": "language",
            "group_column": "source",
            "filtering": {"min_length": 10},
        },
        "anchors": {
            "enabled": True,
            "method": "from_file",
            "anchors_file": str(topic / "anchors.json"),
            "subcategories_file": str(topic / "subcategories.json"),
            "languages": ["en"],
        },
        "embedding": {"model": "stub-st-model", "device": "cpu"},
        "retrieval": {"top_k": 3, "aggregation": "max"},
        "generation": {
            "model": "stub/model",
            "statements_per_group_per_subcategory": 3,
            "output_language": "en",
            "max_retries": 1,
            "retry_delay": 0,
        },
    }
    config_path = topic / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def test_monolingual_pipeline_e2e(stub_embedding_model, stub_llm_client, tmp_path):
    config_path = _write_synthetic_topic(tmp_path)

    from autoconcourse.config import load_config
    from autoconcourse.cli import run_pipeline, PIPELINE_STEPS

    config = load_config(config_path)
    # Pin the project-root-derived dirs to tmp_path so caches/outputs go there.
    config._base_dir = tmp_path

    statements = run_pipeline(config, steps=PIPELINE_STEPS)

    # Generation produced 3 statements per subcategory × 2 subcategories = 6
    assert len(statements) == 6
    subcategory_ids = {s["subcategory_id"] for s in statements}
    assert subcategory_ids == {0, 1}

    # Output JSON exists and round-trips
    out = tmp_path / "synthetic_mono" / "outputs" / "anchor_guided" / "concourse.json"
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["topic"] == "synthetic_mono"
    assert saved["system"] == "anchor_guided"
    assert len(saved["statements"]) == 6
    assert all("subcategory" in s for s in saved["statements"])
    assert all("source_paragraphs" in s for s in saved["statements"])

    # The Excel-friendly CSV is written alongside the JSON.
    csv_out = out.with_suffix(".csv")
    assert csv_out.exists()
    # UTF-8 BOM prefix for Excel compatibility.
    assert csv_out.read_bytes().startswith(b"\xef\xbb\xbf")
    csv_text = csv_out.read_text(encoding="utf-8-sig")
    header = csv_text.splitlines()[0].split(",")
    for col in ("id", "subcategory", "subcategory_id", "group", "text", "source_paragraphs"):
        assert col in header, f"missing column {col!r} in CSV header: {header}"
    # 6 data rows + 1 header row
    assert len(csv_text.strip().splitlines()) == 7


def test_monolingual_pipeline_with_groups(stub_embedding_model, stub_llm_client, tmp_path):
    config_path = _write_synthetic_topic(tmp_path)

    # Add a groups file and enable groups stratification.
    groups_file = config_path.parent / "groups.json"
    # Note: corpus.group_column: "source" causes the adapter to rename source -> group.
    # Downstream (groups stratification) refers to the canonical column name.
    groups_file.write_text(json.dumps({
        "column": "group",
        "groups": {
            "left": {"label": "Left-leaning", "values": ["A"]},
            "right": {"label": "Right-leaning", "values": ["B"]},
        },
    }), encoding="utf-8")

    cfg = yaml.safe_load(config_path.read_text())
    cfg["groups"] = {"enabled": True, "groups_file": str(groups_file), "column": "group"}
    cfg["generation"]["group_instruction_template"] = (
        "The passages below are from {group}-aligned sources."
    )
    config_path.write_text(yaml.safe_dump(cfg))

    from autoconcourse.config import load_config
    from autoconcourse.cli import run_pipeline, PIPELINE_STEPS

    config = load_config(config_path)
    config._base_dir = tmp_path

    statements = run_pipeline(config, steps=PIPELINE_STEPS)

    # 2 groups × 2 subcategories × 3 statements = 12
    assert len(statements) == 12
    groups_seen = {s["group"] for s in statements}
    assert groups_seen == {"left", "right"}
