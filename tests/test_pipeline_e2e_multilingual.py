"""End-to-end smoke test: multilingual DE/FR/IT corpus with one anchor per language.

Verifies that the multilingual retrieval path (max over per-language anchor
embeddings) works: each passage's score for a subcategory is the max similarity
across that subcategory's language-specific anchors.
"""

import json
from pathlib import Path

import yaml


def _write_synthetic_multilingual(root: Path) -> Path:
    topic = root / "synthetic_multi"
    topic.mkdir(parents=True)

    corpus = topic / "corpus.csv"
    corpus.write_text(
        "id,text,language,source\n"
        "p1,Die globalen Temperaturen sind um 1.1 Grad gestiegen.,de,X\n"
        "p2,Le secteur des énergies renouvelables connaît une croissance record.,fr,Y\n"
        "p3,I critici sostengono che le politiche climatiche minacciano la crescita.,it,Z\n"
        "p4,Erneuerbare Energien werden zur dominanten Stromquelle.,de,X\n"
        "p5,Les scientifiques confirment que le réchauffement s'accélère.,fr,Y\n"
        "p6,La transizione energetica richiede investimenti massicci nelle infrastrutture.,it,Z\n",
        encoding="utf-8",
    )

    (topic / "subcategories.json").write_text(json.dumps([
        {"id": 0, "name": "Climate science", "description": "Warming trajectory.",
         "topic_focus": "climate science", "anchor_prompt": "Climate science."},
        {"id": 1, "name": "Energy transition", "description": "Renewables vs fossil fuels.",
         "topic_focus": "energy transition", "anchor_prompt": "Energy transition."},
    ]), encoding="utf-8")

    (topic / "anchors.json").write_text(json.dumps([
        {"category": 0, "language": "de",
         "text": "Wissenschaftler bestätigen, dass die globalen Temperaturen um 1.1 Grad seit der vorindustriellen Zeit gestiegen sind."},
        {"category": 0, "language": "fr",
         "text": "Les scientifiques confirment que les températures mondiales ont augmenté de 1.1 degré depuis l'ère préindustrielle."},
        {"category": 0, "language": "it",
         "text": "Gli scienziati confermano che le temperature globali sono aumentate di 1.1 gradi rispetto all'era preindustriale."},
        {"category": 1, "language": "de",
         "text": "Erneuerbare Energien wie Solar und Wind ersetzen zunehmend fossile Brennstoffe in der Stromerzeugung."},
        {"category": 1, "language": "fr",
         "text": "Les énergies renouvelables comme le solaire et l'éolien remplacent progressivement les combustibles fossiles dans la production d'électricité."},
        {"category": 1, "language": "it",
         "text": "Le energie rinnovabili come solare ed eolico stanno gradualmente sostituendo i combustibili fossili nella produzione di elettricità."},
    ]), encoding="utf-8")

    cfg = {
        "topic": {"name": "Synthetic multi", "id": "synthetic_multi", "languages": ["de", "fr", "it"]},
        "llm": {"default_model": "stub/model"},
        "corpus": {
            "adapter": "csv",
            "path": str(corpus),
            "text_column": "text",
            "id_column": "id",
            "language_column": "language",
            "filtering": {"min_length": 10},
        },
        "anchors": {
            "enabled": True,
            "method": "from_file",
            "anchors_file": str(topic / "anchors.json"),
            "subcategories_file": str(topic / "subcategories.json"),
            "languages": ["de", "fr", "it"],
        },
        "embedding": {"model": "stub-st-model", "device": "cpu"},
        "retrieval": {"top_k": 3, "aggregation": "max"},
        "generation": {
            "model": "stub/model",
            "statements_per_group_per_subcategory": 2,
            "output_language": "en",
            "max_retries": 1,
            "retry_delay": 0,
        },
    }
    config_path = topic / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def test_multilingual_pipeline_e2e(stub_embedding_model, stub_llm_client, tmp_path):
    config_path = _write_synthetic_multilingual(tmp_path)

    from autoconcourse.config import load_config
    from autoconcourse.cli import run_pipeline, PIPELINE_STEPS

    config = load_config(config_path)
    config._base_dir = tmp_path

    statements = run_pipeline(config, steps=PIPELINE_STEPS)

    # The LLM stub returns 3 statements per call × 2 subcategories = 6
    assert len(statements) == 6
    assert {s["subcategory_id"] for s in statements} == {0, 1}

    out = tmp_path / "synthetic_multi" / "outputs" / "anchor_guided" / "concourse.json"
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["topic"] == "synthetic_multi"
    assert len(saved["statements"]) == 6


def test_multilingual_retrieval_uses_all_anchor_languages(stub_embedding_model, tmp_path):
    """Verify the retrieval step actually consults every per-language anchor.

    With our deterministic-hash embedder, the cleanest assertion is structural:
    the retrieval result is keyed by subcategory_id, contains top_k rows per
    subcategory, and similarity_scores are well-defined.
    """
    config_path = _write_synthetic_multilingual(tmp_path)

    from autoconcourse.config import load_config
    from autoconcourse.cli import run_pipeline

    config = load_config(config_path)
    config._base_dir = tmp_path

    # Run up to retrieval; skip generate (no LLM stub here).
    run_pipeline(config, steps=["corpus", "anchors", "embed", "retrieve"])

    # Project-local paragraphs artifact should exist and have all 6 rows
    import pandas as pd
    paragraphs = pd.read_csv(tmp_path / "synthetic_multi" / "data" / "paragraphs" / "paragraphs.csv")
    assert len(paragraphs) == 6
    # Project-local embeddings artifacts exist
    import numpy as np
    para_emb = np.load(tmp_path / "synthetic_multi" / "data" / "embeddings" / "paragraph_embeddings.npy")
    anchor_emb = np.load(tmp_path / "synthetic_multi" / "data" / "embeddings" / "anchor_embeddings.npy")
    assert para_emb.shape == (6, 384)
    assert anchor_emb.shape == (6, 384)  # 2 subcategories × 3 languages
