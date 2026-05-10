# AutoConcourse

Automated generation of Q-concourse statements from text corpora. Given a corpus and a list of subcategories, AutoConcourse retrieves topically relevant passages via anchor-guided dense retrieval and uses an LLM to synthesise representative consideration statements suitable for Q-methodology surveys.

The method is corpus- and language-agnostic: any text input that can be loaded into the canonical shape works, and the default embedding model is multilingual.

## Install

```bash
pip install -e .
echo "OPENROUTER_API_KEY=your-key-here" > .env       # or OPENAI_API_KEY
```

Python 3.11+. CPU is fine; GPU is faster for the embedding step on large corpora (set `embedding.device: cuda` in your config).

## Run

```bash
# Built-in monolingual example (English, 6 paragraphs)
python -m autoconcourse.cli --config data/example/monolingual/config.yaml

# Multilingual example (DE / FR / IT, one anchor per language per subcategory)
python -m autoconcourse.cli --config data/example/multilingual/config.yaml

# Equivalent invocations
python run_pipeline.py --config <path>
autoconcourse --config <path>          # console script, after pip install -e
```

Output lands at `outputs/<topic.id>/anchor_guided/concourse.json`. Intermediate state (corpus, embeddings, anchors) caches under `cache/<topic.id>/`. Re-runs skip cached steps unless `--force` is passed; `--steps corpus,embed,retrieve,generate` runs only listed steps.

## How it works

```
raw input (CSV, JSONL, Swissdox TSV.xz, .txt directory)
  → adapter → canonical corpus DataFrame
  → SBERT embed
  → anchor-guided cosine retrieval (multilingual: max over per-language anchors)
  → LLM generate N statements per subcategory
  → concourse.json
```

Six core modules in `autoconcourse/`:

| Module | Role |
|---|---|
| `corpus.py` | Adapter dispatch + canonical-shape filters (keyword, min_length, language) |
| `adapters/` | csv, jsonl, swissdox, text_dir — convert raw input into canonical shape |
| `anchors.py` | Load or LLM-generate anchors; multilingual (one per language per subcategory) |
| `embedding.py` | Sentence-Transformers wrapper with `.npy` I/O |
| `retrieval.py` | Per-anchor cosine, aggregated (max or mean) per subcategory |
| `generation.py` | LLM statement generation; OpenAI-compatible (OpenRouter, OpenAI, Azure, local vLLM) |

### Canonical corpus shape

| Column | Type | Required | Notes |
|---|---|---|---|
| `id` | str/int | yes | unique passage id |
| `text` | str | yes | the passage |
| `language` | str | no | ISO code; multilingual retrieval uses this |
| `group` | str | no | stratification column for optional groups split |
| `source_id` | str | no | original document id, for tracing |
| `*` | any | no | extra metadata kept as-is |

Format-specific work lives in adapters, never in core. Add a new format by writing one `load(path, **opts) -> DataFrame` function under `autoconcourse/adapters/`.

## Adding a new topic

1. Copy an example: `cp -r data/example/monolingual data/<your_topic>` (or the multilingual one).
2. Replace `corpus.csv` with your data (canonical columns: `id`, `text`, plus optional language/group/source_id).
3. Edit `subcategories.json` — define the thematic dimensions you want statements organised along.
4. Choose an anchor strategy:
   - `from_file`: hand-write `anchors.json` (one entry per subcategory × language).
   - `llm_generated`: omit the file, set `anchors.method: llm_generated` — anchors are generated on first run from each subcategory's `anchor_prompt` (and optional `context_document`).
5. Edit `config.yaml`: `topic.id`, `corpus.path`, anchor/subcategory paths.
6. Run: `python -m autoconcourse.cli --config data/<your_topic>/config.yaml`.

For non-CSV inputs, change `corpus.adapter` to `jsonl`, `swissdox`, or `text_dir`. See `configs/schema.yaml` for the full set of options.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

The smoke tests stub the embedding model and LLM client, so they run offline in a few seconds — useful as a sanity check before any real pipeline run.

## Layout

```
autoconcourse/        # Bare method (≈900 LOC)
├── adapters/         # csv, jsonl, swissdox, text_dir
├── prompts/          # anchor_generation.txt, statement_generation.txt
├── config.py         # Typed config + project-root resolution
├── corpus.py         # Adapter dispatch + canonical-shape filters
├── anchors.py        # Load / generate / save anchors
├── embedding.py      # SBERT + .npy I/O
├── retrieval.py      # Anchor-guided cosine, max-or-mean over per-language anchors
├── generation.py     # LLM client + statement generation
├── cli.py            # Single entry point: pipeline_main
└── __main__.py       # `python -m autoconcourse`

configs/              # Reference schema (configs/schema.yaml)
data/example/         # Two runnable example topics
├── monolingual/      # English
└── multilingual/     # DE / FR / IT
tests/                # Offline pytest smoke tests
notebooks/            # Optional helpers (anchor_workshop, exploration)

paper/                # Parked EMNLP validation work — see paper/README.md
├── baselines.py      # B1–B6
├── evaluation.py     # coverage, groundedness, distinctiveness, balance, novelty
├── prompt_optimization.py
├── data/             # healthcare_ch / food_policy_ch
└── configs/

legacy/               # Original thesis notebooks (untouched)
texts/                # Planning docs (bare_method_plan.md, EMNLP_plan.md, …)
```

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
