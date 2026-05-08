# Strip-Back Plan: AutoConcourse, Bare Method

**Goal.** Reduce the codebase to the smallest pipeline that takes *any topic* (corpus + subcategory list) and produces a concourse. The pipeline operates on a single canonical corpus shape; format-specific input (Swissdox TSV, raw HTML, JSONL, etc.) is handled by adapters that convert *into* the canonical shape. Multilingual is first-class — supported by the canonical shape, the anchor format, and the default embedding model — not stripped.

**Sequencing.** Bare method first, EMNLP validation second. Validation is not abandoned — it's the inevitable next phase. The strip must therefore leave **clean seams** so baselines (B1–B6), evaluation metrics, ablations (A1–A8), and prompt optimization reattach as plug-ins, *not* as a rewrite. Paper code moves to `paper/` as a *parked module*, not an exile zone.

**Three design constraints this implies.**
1. **Stable interfaces.** Retrieval and generation are strategy interfaces from day one — baselines later become alternative implementations of the same shape, not forks of the core. Keep them small and explicit.
2. **Stable output schema.** `concourse.json` is the contract baselines and evaluation will depend on. Lock it now, don't churn it during the strip.
3. **Canonical corpus shape, format adapters at the edge.** The pipeline never sees Swissdox columns or raw HTML. Adapters convert into the canonical shape; the rest of the pipeline is format-agnostic.

---

## Bare method (what stays)

```
raw input (Swissdox TSV.xz, HTML, CSV, JSONL, …)
  → adapter → canonical corpus
  → embed (multilingual SBERT)
  → for each subcategory:
      anchor text(s) (one per language) → embed → cosine retrieve top-k passages
  → LLM generate N statements per subcategory from retrieved passages
  → concourse.json
```

That is it. Six core modules + adapters, one driver, one config.

### Canonical corpus shape

A pandas DataFrame (or equivalent) with these columns:

| Column | Type | Required | Notes |
|---|---|---|---|
| `id` | str/int | yes | unique passage id |
| `text` | str | yes | the passage |
| `language` | str | no | ISO code (`de`, `fr`, `it`, `en`, …); used by retrieval/generation when present |
| `group` | str | no | stratification column (replaces `perspective_group`); arbitrary user-chosen grouping |
| `source_id` | str | no | original document id, for tracing |
| `*` | any | no | extra metadata kept as-is, ignored by core |

Embeddings ship alongside as a parallel `.npy`, indexed by `id` order.

### Adapters

Live in `autoconcourse/adapters/`. Each adapter is a function `load(path, **opts) -> canonical DataFrame`.

| Adapter | Handles |
|---|---|
| `swissdox.py` | TSV.xz, `content_id` dedup, HTML `<p>` chunking, language column from Swissdox metadata |
| `csv.py` | Generic CSV with column-mapping config (`text_column`, `language_column`, `group_column`) |
| `jsonl.py` | Line-delimited JSON |
| `text_dir.py` | Directory of `.txt` files (each file = one document, double-newline split) |

Adding a new input format = writing a new adapter. The pipeline never changes.

---

## Phase 1 — Remove paper scaffolding

Delete or move out of `autoconcourse/`:

| File | Action | Why |
|---|---|---|
| `autoconcourse/baselines.py` | move → `paper/baselines.py` | B1–B6 only matter for comparison tables |
| `autoconcourse/evaluation.py` | move → `paper/evaluation.py` | needs reference concourse; not part of generation |
| `autoconcourse/prompt_optimization.py` | move → `paper/` (or delete) | premature; no signal yet that base prompts fail |
| `autoconcourse/reduction.py` | delete | UMAP for cosine retrieval is unmotivated; default was `none` anyway |
| `run_baselines.py`, `run_evaluation.py`, `run_experiment.py` | move → `paper/` | paper-only entry points |
| `autoconcourse/cli.py` ablation runner (`experiment_main`, `baselines_main`, `evaluation_main`) | strip; keep only `pipeline_main` | cli should run *the method*, nothing else |
| `autoconcourse/prompts/policy_option_generation.txt` | move → `paper/prompts/` | DRI-specific second deliverable |
| `autoconcourse/prompts/prompt_optimization.txt` | move → `paper/prompts/` | follows `prompt_optimization.py` |
| `pyproject.toml` console scripts | drop `autoconcourse-baselines`, `-evaluate`, `-experiment` | one entry point: `autoconcourse-pipeline` |
| `data/example/example_reference_concourse.json` | move → `paper/` | only used by evaluation |
| `data/example/example_perspective_groups.json` | delete or rename to `example_groups.json` | rename matters — see Phase 2 |
| `evaluation/`, `outputs/` empty dirs | delete | clutter |

**Result:** `autoconcourse/` shrinks from 13 modules / ~2,700 LOC to 7 modules / ~1,000 LOC.

---

## Phase 2 — Decouple from Swiss/news/multilingual specifics

These are surgical edits inside the modules that survive Phase 1.

### 2.1 `corpus.py` + new `autoconcourse/adapters/`

- `corpus.py` becomes thin: dispatch to the right adapter (based on `corpus.adapter` in config), then apply optional steps that operate on canonical shape (keyword filter, min-length filter, language filter).
- Move all Swissdox-specific logic — `content_id` dedup, HTML `<p>` extraction, TSV.xz reading — into `autoconcourse/adapters/swissdox.py`. The Swissdox path keeps working; it just stops polluting the core.
- `csv.py`, `jsonl.py`, `text_dir.py` adapters as siblings. Each takes a column-mapping in config so any CSV layout converts cleanly.
- HTML chunking only runs in adapters where it makes sense (Swissdox, raw-HTML loader). `corpus.py` itself never calls BeautifulSoup.
- Keyword filter accepts the language-grouped `{"de": [...], "fr": [...]}` shape *and* a flat list; both are valid because multilingual is first-class.

### 2.2 `generation.py`

- Replace `perspective_group` with a generic `group` parameter.
- Rip out the hardcoded `"The media paragraphs below come from {group}-leaning media outlets…"` string. Move that wording into the prompt template, parameterized as `{group_instruction}` that the user supplies via config (defaults to empty).
- `output_language` stays as a config knob. Source language is *not* assumed — the prompt can mention "the passages are in multiple languages" or be silent, depending on what the corpus actually contains. Don't bake DE/FR/IT into the prompt.
- Drop the second prompt path (`statement_type="policy_option"`) — DRI-specific, moves to `paper/prompts/`.

### 2.3 `anchors.py`

- Keep three modes: `from_file`, `interactive` (notebook), `llm_generated`.
- **Anchors stay multilingual.** Format: each subcategory has a list of `{language, text}` entries (one per language present in the corpus). A monolingual topic just has one entry per anchor. Retrieval handles both cleanly — see 2.4.
- Drop the policy-PDF context-document hardcode. Make `context_document` optional and generic (any text/PDF), not a Swiss-policy default.
- The default embedding model stays `paraphrase-multilingual-MiniLM-L12-v2` because multilingual is a first-class case.

### 2.4 Retrieval against multilingual anchors

- An anchor for subcategory `c` is a list of texts, one per language. Each gets embedded; we keep them as a `(n_languages, dim)` matrix per subcategory.
- For each corpus passage, similarity to subcategory `c` = max similarity to any of its language-specific anchor embeddings (or mean — make it a config knob; default `max`).
- This way: a German passage matches the German anchor, a French passage matches the French anchor, and you never lose recall by averaging across languages. Monolingual case: list of length 1, behaviour identical to current.

### 2.5 Configs and data

- Rename `perspective_groups` → `groups` everywhere (config schema, code, JSON files). It's any stratification column the user wants, not specifically political leaning.
- `data/healthcare_ch/` and `data/food_policy_ch/` → move to `paper/data/` as worked examples. They aren't part of the generic method but are the *primary validation cases* for the EMNLP work.
- Keep `data/example/` as the canonical "this is what a topic looks like." Two flavours: `example_monolingual/` (English) and `example_multilingual/` (DE/FR/IT) so the multilingual case is a documented first-class path, not an afterthought.
- `configs/schema.yaml` → trim. Remove `evaluation:`, `reduction:`, `generation.prompt_optimization:`. Add `corpus.adapter` and adapter-specific option blocks. Languages stay configurable but with no hardcoded defaults.

### 2.6 Prompts

- One generation prompt template. Variables: `topic`, `subcategory_name`, `subcategory_description`, `n_statements`, `output_language`, `group_instruction` (optional, blank by default), `paragraphs`.
- One anchor-generation prompt template. Variables: `subcategory_name`, `subcategory_description`, `context` (optional), `language` (so the same prompt produces DE/FR/IT/EN versions of the anchor).

---

## Phase 3 — Final shape

```
autoconcourse/
├── __init__.py
├── config.py            # YAML load + LLM client factory
├── corpus.py            # adapter dispatch + canonical-shape filters
├── adapters/            # raw input → canonical corpus
│   ├── __init__.py
│   ├── swissdox.py      # TSV.xz, content_id dedup, HTML <p> chunking
│   ├── csv.py           # generic CSV with column mapping
│   ├── jsonl.py
│   └── text_dir.py
├── anchors.py           # load / generate / save anchors (multilingual)
├── embedding.py         # SBERT wrapper, .npy I/O
├── retrieval.py         # cosine top-k per subcategory anchor (multi-lang aware)
├── generation.py        # LLM → statements
├── utils.py             # logging, JSON I/O
├── cli.py               # one entry point: pipeline_main
└── prompts/
    ├── anchor_generation.txt
    └── statement_generation.txt

configs/
├── schema.yaml
└── example.yaml         # points at data/example/

data/
└── example/             # minimal generic topic, runnable end-to-end

notebooks/
└── anchor_workshop.ipynb

paper/                   # everything for the EMNLP submission, parked
├── baselines.py
├── evaluation.py
├── prompt_optimization.py
├── run_baselines.py
├── run_evaluation.py
├── run_experiment.py
├── prompts/
│   ├── policy_option_generation.txt
│   └── prompt_optimization.txt
└── data/
    ├── healthcare_ch/
    └── food_policy_ch/

legacy/                  # untouched

run_pipeline.py          # the only top-level entry point
README.md                # rewritten around the bare method
pyproject.toml           # one console script
```

**Lines of code budget:** ~1,000 for `autoconcourse/`, down from ~2,700.

---

## Phase 4 — Validate the strip

The whole point of stripping is to run the method on something arbitrary, while still working on what we already have. Three tests, in order:

1. **Monolingual synthetic smoke test.** 50 paragraphs of English Wikipedia text, 3 hand-written subcategories, 3 anchor sentences. End-to-end run produces a concourse JSON in <1 minute on CPU. Add as `tests/test_pipeline_e2e_monolingual.py`.
2. **Multilingual smoke test.** Same harness, but corpus mixes DE/FR/IT passages and anchors are trilingual. Verifies the multilingual retrieval path (max-over-language-anchors).
3. **Swissdox regression.** Run the Swissdox adapter on a healthcare-CH sample → canonical corpus → full pipeline → concourse. The output should be qualitatively similar to the thesis output. This guarantees the format-adapter refactor didn't break the original use case.
4. **One real non-Swiss, non-news topic.** Any open English corpus (HuggingFace `cc_news` filtered, or a single book split into paragraphs). 5 subcategories. If it requires *any* code change beyond writing a config + adapter selection, the strip isn't done.

Only after all four pass do you re-attach `paper/` for the EMNLP work.

---

## Sequencing

1. Branch: `strip/bare-method`.
2. Phase 1 (move/delete) in one commit — pure file moves, no code edits. Pipeline still runs.
3. Phase 2.1–2.3 (decouple `corpus.py`, `generation.py`, `anchors.py`) — one commit per module.
4. Phase 2.4–2.5 (configs, data, prompts) — one commit.
5. Phase 4 smoke test — add `tests/`, make it pass.
6. Phase 4 real-topic test — add `data/<topic>/`, `configs/<topic>.yaml`, run, commit outputs to `outputs/<topic>/`.
7. Rewrite `README.md` around the bare method (the current one still describes baselines/ablations).

Each phase ends with a runnable pipeline. No long-lived broken state.

---

## Decisions to make before starting

- **Keep `groups` (renamed perspective stratification) or drop it entirely?** It's ~30 LOC across `retrieval.py` and `generation.py`. Keep if you think most topics will want stratification; drop if you want the simplest possible core and reintroduce later.
- **Keep `interactive` anchor notebook?** It's not strictly part of the method but it is part of the *workflow* and is useful for new topics. Recommendation: keep, but move out of `notebooks/` into `tools/anchor_workshop.ipynb` to signal it's a helper, not exploration.
- **Where does `paper/` live?** Same repo (subdir) or sibling repo? Same repo is simpler; sibling repo makes the bare-method package cleaner to publish. Recommendation: same repo, `paper/` subdir, until you actually publish.
