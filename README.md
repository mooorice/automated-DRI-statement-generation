# AutoConcourse

Automated generation of Q-concourse statements from text corpora. You give it a corpus and a list of subcategories; it retrieves topically relevant passages via anchor-guided dense retrieval and uses an LLM to synthesise representative consideration statements suitable for Q-methodology surveys.

The method is corpus- and language-agnostic: any text input that can be loaded into the canonical shape works, and the default embedding model is multilingual.

---

## Install

```bash
pip install -e .
echo "OPENROUTER_API_KEY=your-key-here" > .env       # or OPENAI_API_KEY
```

Python 3.11+. CPU works; for large corpora set `embedding.device: cuda` in your config.

---

## Quick start

```bash
# English-only example: 6 paragraphs, 2 subcategories, 2 hand-written anchors
python -m autoconcourse.cli --config data/example/monolingual/config.yaml

# Multilingual example: DE/FR/IT, one anchor per language per subcategory
python -m autoconcourse.cli --config data/example/multilingual/config.yaml
```

After a successful run you'll see:

```
outputs/example_monolingual/anchor_guided/concourse.json    ← what you came for
cache/example_monolingual/paragraphs.csv                    ← canonical corpus
cache/example_monolingual/anchors.csv                       ← anchors
cache/example_monolingual/paragraph_embeddings.npy          ← (n_passages, 384) float
cache/example_monolingual/anchor_embeddings.npy             ← (n_anchors, 384) float
```

Equivalent invocations:

```bash
python run_pipeline.py --config <path>
autoconcourse --config <path>          # console script after pip install -e
```

---

## Inputs

A topic is a directory with up to four files plus one YAML config. The pipeline only requires **corpus + subcategories**. Anchors, keywords, and groups are optional.

### 1. Corpus (required)

The thing you want to mine for statements. Default adapter expects a CSV with at minimum a text column.

**File:** `corpus.csv` (or any path you point `corpus.path` at)

| Column | Type | Required | Notes |
|---|---|---|---|
| `id` | str / int | no¹ | unique passage id; auto-generated if absent |
| `text` | str | yes | the passage |
| `language` | str | no | ISO code (`en`, `de`, …); used by multilingual retrieval |
| `source` (or any name → `group`) | str | no | stratification column; map via `corpus.group_column` |
| `*` | any | no | extra metadata kept as-is |

¹ Required for *deterministic* output IDs across runs. If your CSV's id column has a different name, set `corpus.id_column` in the config.

Other input formats are supported via adapters — set `corpus.adapter` to one of:

| Adapter | Path expects | Notes |
|---|---|---|
| `csv` | a `.csv` file | column-mapping via `text_column`, `id_column`, `language_column`, `group_column`, `source_id_column` |
| `jsonl` | a `.jsonl` file | one JSON object per line; same column-mapping |
| `swissdox` | a `.tsv` or `.tsv.xz` | Swissdox dump: dedupes by `content_id`, splits HTML `<p>` paragraphs, maps `medium_code` → `group` |
| `text_dir` | a directory of `.txt` | each file = one document, double-newline-split into paragraphs |

To add a new format: drop a new module in `autoconcourse/adapters/` exposing `load(path, **opts) -> pd.DataFrame` and register it in `autoconcourse/adapters/__init__.py:ADAPTERS`.

### 2. Subcategories (required)

The thematic dimensions you want statements organised along. The pipeline runs retrieval and generation once per subcategory.

**File:** `subcategories.json` — JSON array of objects.

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | int | yes | 0-indexed, unique |
| `name` | str | yes | short label |
| `description` | str | yes | what this subtopic covers |
| `topic_focus` | str | yes | short focus phrase, injected into the generation prompt |
| `anchor_prompt` | str | only if `anchors.method: llm_generated` | prompt fragment for LLM-based anchor generation |

Example: `data/example/monolingual/subcategories.json`.

### 3. Anchors (optional — auto-generated if omitted)

Anchor texts are short paragraphs that semantically represent each subcategory. They drive retrieval: passages are scored by cosine similarity to anchors.

Two ways to provide them:

**(a) Hand-written.** Set `anchors.method: from_file` and provide:

**File:** `anchors.json` — JSON array of objects.

| Field | Type | Required | Notes |
|---|---|---|---|
| `category` | int | yes | matches a subcategory's `id` |
| `text` | str | yes | the anchor passage |
| `language` | str | no | ISO code; for multilingual corpora, write one anchor per language per subcategory |
| `subcategory_name` | str | no | for human readability only |

**(b) LLM-generated.** Set `anchors.method: llm_generated`. The pipeline calls the LLM once per (subcategory, language) on first run and caches results to `cache/<topic_id>/anchors.csv`. Optionally pass `anchors.context_document: path/to/source.pdf` and the LLM grounds the anchors on its content.

### 4. Keywords (optional)

For corpora where most passages are off-topic. The pipeline drops passages that contain none of these keywords.

**File:** `keywords.json` — either a flat list or a `{lang_code: [...]}` dict.

```json
{ "en": ["climate", "carbon"], "de": ["Klima", "Kohlenstoff"] }
```

Enable in config: `corpus.filtering.keywords_file: path/to/keywords.json`, `corpus.filtering.require_keyword: true`.

### 5. Groups (optional)

Stratify retrieval by a corpus column (e.g., political leaning, source type, demographic). When enabled, the pipeline runs retrieval once per group, producing one set of statements per (group, subcategory) pair.

**File:** `groups.json`

```json
{
  "column": "group",
  "groups": {
    "left":  { "label": "Left-leaning",  "values": ["GUARDIAN", "NYT"] },
    "right": { "label": "Right-leaning", "values": ["WSJ", "TELEGRAPH"] }
  }
}
```

Enable in config: `groups.enabled: true`, `groups.groups_file: path/to/groups.json`. Optionally set `generation.group_instruction_template: "The passages below are from {group}-aligned sources."` to surface the group label to the LLM.

### 6. Config (required)

A single YAML file ties it all together. See `configs/schema.yaml` for the full reference. The minimal config:

```yaml
topic:
  name: "My topic"
  id: "my_topic"
  languages: ["en"]

corpus:
  adapter: "csv"
  path: "data/my_topic/corpus.csv"
  text_column: "text"
  language_column: "language"

anchors:
  enabled: true
  method: "from_file"
  anchors_file: "data/my_topic/anchors.json"
  subcategories_file: "data/my_topic/subcategories.json"

embedding:
  model: "paraphrase-multilingual-MiniLM-L12-v2"
  device: "cpu"

retrieval:
  top_k: 500
  aggregation: "max"   # max-over-anchors per subcategory; "mean" also valid

generation:
  model: "openai/gpt-4o"
  statements_per_group_per_subcategory: 5
  output_language: "en"
```

---

## Outputs

### Final output: `outputs/<topic.id>/anchor_guided/concourse.json`

This is the deliverable.

```json
{
  "system": "anchor_guided",
  "topic": "my_topic",
  "model": "openai/gpt-4o",
  "timestamp": "2026-05-10T14:32:01+00:00",
  "config_hash": "a3f8c1e9b2d4",
  "statements": [
    {
      "id": "my_topic_001",
      "text": "Aggressive climate policy threatens energy security and economic growth.",
      "subcategory": "Energy transition",
      "subcategory_id": 1,
      "group": null,
      "source_paragraphs": [42, 117, 203, 389, 412, 88, 251, 305, 174, 9]
    },
    ...
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `system` | str | always `"anchor_guided"` for the bare method |
| `topic` | str | matches `topic.id` from your config |
| `model` | str | the LLM that generated the statements |
| `timestamp` | ISO 8601 | UTC, when the file was written |
| `config_hash` | str | first 12 chars of SHA-256 over the raw config dict; lets you tell two runs apart |
| `statements[].id` | str | `<topic_id>_<seq>`; sequential within a run |
| `statements[].text` | str | the generated statement |
| `statements[].subcategory` | str | human-readable name from your subcategories.json |
| `statements[].subcategory_id` | int | the matching `id` |
| `statements[].group` | str / null | non-null only when groups stratification was enabled |
| `statements[].source_paragraphs` | list[int/str] | the top 10 retrieved passage ids that informed this statement (for grounding/audit) |

**Statement count.** Without groups: `n_subcategories × statements_per_group_per_subcategory`. With groups: `n_groups × n_subcategories × statements_per_group_per_subcategory`.

### Cache: `cache/<topic.id>/`

Intermediate artifacts. Re-runs skip steps whose outputs already exist (use `--force` to override).

| File | Shape / content | Produced by |
|---|---|---|
| `paragraphs.csv` | canonical-shape DataFrame after adapter + filters | corpus step |
| `anchors.csv` | anchors loaded or LLM-generated | anchors step |
| `paragraph_embeddings.npy` | `(n_passages, embedding_dim)` float32 | embed step |
| `anchor_embeddings.npy` | `(n_anchors, embedding_dim)` float32 | embed step |

You can safely delete the cache to force a rebuild from scratch.

---

## Pipeline stages

```
raw input → adapter → canonical corpus
                         ↓
                       embed (SBERT) ────┐
                                         │
   anchors.json (or LLM-generated) → embed
                                         ↓
                              anchor-guided cosine retrieval
                              (max-or-mean over per-language anchors)
                                         ↓
                           top-k passages per subcategory
                                         ↓
                                  LLM generation
                                         ↓
                                  concourse.json
```

Run a subset with `--steps`:

```bash
# Just prepare the corpus, anchors, and embeddings without calling the LLM
python -m autoconcourse.cli --config <path> --steps corpus,anchors,embed,retrieve

# Re-run only generation (re-uses cached embeddings + retrieval state)
python -m autoconcourse.cli --config <path> --steps retrieve,generate
```

Steps: `corpus`, `anchors`, `embed`, `retrieve`, `generate`.

---

## Adding a new topic (walkthrough)

```bash
# 1. Copy an example as a starting point
cp -r data/example/monolingual data/my_topic

# 2. Replace the corpus
#    Drop your CSV at data/my_topic/corpus.csv (must have at least 'text';
#    optionally 'id', 'language', and any column you want to stratify on)

# 3. Edit subcategories
$EDITOR data/my_topic/subcategories.json

# 4. Either hand-write data/my_topic/anchors.json (one entry per subcategory ×
#    language) OR delete it and set anchors.method: llm_generated in config

# 5. Update the config: topic.id, corpus.path, anchor/subcategory paths
$EDITOR data/my_topic/config.yaml

# 6. Run
python -m autoconcourse.cli --config data/my_topic/config.yaml

# 7. Inspect output
cat outputs/my_topic/anchor_guided/concourse.json
```

For non-CSV input, change `corpus.adapter` and follow the table under [Inputs § 1](#1-corpus-required).

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Six offline smoke tests stub the embedder + LLM and run the full pipeline against synthetic data in ~3.5s. No API key, no model downloads. Good as a sanity check before any real run.

---

## Layout

```
autoconcourse/        # Bare method (~1,600 LOC)
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

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
