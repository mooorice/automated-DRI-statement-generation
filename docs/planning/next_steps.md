# Next Steps — AutoConcourse

**Date:** 2026-02-20
**Status:** Codebase refactoring complete. Pipeline infrastructure is in place. No experiments have been run yet.

---

## What's Done

The entire plan.md Phases 1–5 (code infrastructure) have been implemented:

- **Package:** All 13 modules in `autoconcourse/` are fully implemented (config, corpus, anchors, embedding, reduction, retrieval, generation, baselines B1–B6, evaluation with 5 metrics, prompt optimization, utils, CLI).
- **Prompts:** 4 prompt templates (anchor generation, statement generation, policy option generation, prompt optimization).
- **Entry points:** `run_pipeline.py`, `run_baselines.py`, `run_evaluation.py`, `run_experiment.py` — all wired up with argument parsing.
- **Configs:** `schema.yaml` (reference), `healthcare_ch.yaml`, `food_policy_ch.yaml`.
- **Data files:** `data/example/` (8 files with README), `data/healthcare_ch/` (keywords, subcategories, perspective_groups), `data/food_policy_ch/` (same three files).
- **Ablations:** A1–A8 defined in `cli.py` with experiment runner.
- **Legacy:** Original notebooks moved to `legacy/`.
- **README:** Accurate project documentation with setup, usage, and new-topic instructions.

---

## What's Needed Before Running

The pipeline code is ready but cannot run end-to-end yet because it needs **data files** that are external to the repository:

### 1. Corpus files (blocking)

The configs expect pre-processed paragraph CSVs at:
- `data/healthcare_ch/corpus/paragraphs.csv`
- `data/food_policy_ch/corpus/paragraphs.csv`

**Action:** Either:
- (a) Run the legacy notebooks (`legacy/01a_get_articles.ipynb` → `legacy/02_chunk_into_paragraphs.ipynb`) to regenerate the paragraph CSVs from Swissdox, then copy the output to the expected paths. OR
- (b) If you already have the thesis pipeline's output CSVs (from `data/01_text_data/paragraphs.csv` or equivalent), copy/symlink them to the new locations. Make sure columns include at minimum: `text`, `id`, `medium_code`, `language`.

### 2. Context documents for anchor generation (non-blocking if using `from_file` method)

- `data/99_anchor_data/B_Politikfeldanalyse_Gesundheitskosten_2024_08_02_INTERFACE.pdf` (healthcare)
- `data/food_policy_ch/anchor_background.pdf` (food policy)

If these PDFs aren't available, change config to `anchors.method: "from_file"` and provide pre-generated anchor files instead (see item 3).

### 3. Anchor files (optional — pipeline can generate these)

Plan item 4.2 calls for extracting anchor texts from `legacy/01b_content_anchor_generation.ipynb` into `data/healthcare_ch/anchors.json`. This wasn't done because the pipeline supports LLM-generated anchors (the default). However, for reproducibility with the thesis results, you may want to:
- Extract the manually crafted anchors from the legacy notebook
- Save as `data/healthcare_ch/anchors.json` in the expected format (see `data/example/example_anchors.json`)
- Change config to `anchors.method: "from_file"` and set `anchors.anchors_file`

### 4. Reference concourses (needed for evaluation only)

- `data/healthcare_ch/reference_concourse.json` — Bevölkerungsrat 2025 DRI survey (28 statements + 8 policy options)
- `data/food_policy_ch/reference_concourse.json` — Mathys (2024) SCAFP DRI survey (41 statements + 8 policy options)

**Action:** Transcribe the reference statements into JSON format matching `data/example/example_reference_concourse.json`.

### 5. API key

Set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) in `.env` for LLM-dependent steps (anchor generation, statement generation, baselines B1–B4/B6).

---

## Immediate Next Steps (Priority Order)

### Step 1: Prepare healthcare corpus and do a first full run

1. Get `paragraphs.csv` into `data/healthcare_ch/corpus/` (see above)
2. Run: `python run_pipeline.py --config configs/healthcare_ch.yaml`
3. Inspect the output at `outputs/healthcare_ch/anchor_guided/concourse.json`
4. Iterate on prompt quality if needed

This is the critical validation that the pipeline works end-to-end.

### Step 2: Create reference concourses

Transcribe the human-authored DRI statements (Bevölkerungsrat 2025 for healthcare, Mathys 2024 for food policy) into the JSON format. These are needed before any evaluation can happen.

### Step 3: Run baselines on healthcare

```bash
python run_baselines.py --config configs/healthcare_ch.yaml --baselines all
```

This produces one `concourse.json` per baseline under `outputs/healthcare_ch/`.

### Step 4: First evaluation round

```bash
python run_evaluation.py --config configs/healthcare_ch.yaml --systems all
```

Review the metrics. This is the first empirical signal for the paper.

### Step 5: Repeat for food policy

Same pipeline + baselines + evaluation with `configs/food_policy_ch.yaml`.

### Step 6: Third topic — selection and setup

Per the EMNLP plan, an English-language topic is needed for generalizability. Climate change policy is the recommended choice (abundant open data, existing Q-studies for reference). Steps:

1. **Decide the topic** (climate, immigration, AI regulation, or housing)
2. **Assemble corpus** — use an open dataset (HuggingFace, All The News, MediaCloud) to avoid Swissdox redistribution issues
3. **Define subcategories** — draw from existing discourse frameworks (e.g. IPCC categories for climate)
4. **Create config** at `configs/topic3.yaml`
5. **Find or create reference concourse** — ideally from a published Q-study

### Step 7: Ablation experiments

```bash
python run_experiment.py --config configs/healthcare_ch.yaml --ablations all --dry-run  # preview
python run_experiment.py --config configs/healthcare_ch.yaml --ablations all
```

Key research questions to answer:
- A1 vs A3: Does UMAP help or hurt retrieval quality?
- A8: How much do anchors contribute over unguided retrieval?
- A5: Sensitivity to top-k parameter
- A7: Does generator model matter less than retrieval quality?

### Step 8: Prompt optimization (after initial results)

If the initial statement quality is decent but has identifiable weaknesses:

```bash
# Enable in config: generation.prompt_optimization.enabled: true
python run_pipeline.py --config configs/healthcare_ch.yaml --steps generate --force
```

This iteratively improves the generation prompt using LLM self-critique.

---

## Paper Timeline (from EMNLP_plan.md)

| Target | Milestone |
|---|---|
| Week 1–2 | First full pipeline run + baselines on healthcare |
| Week 2–4 | All systems running on all 3 topics |
| Week 4–6 | Evaluation complete (automatic + human) |
| Week 7–9 | Paper writing (sections 1–8) |
| Week 10–11 | External feedback + revision |
| Week 12 | Submit to EMNLP |

CompText (mid-April) comes first — focus on having healthcare + food policy results ready and the method clearly presentable.

---

## Known Technical Notes

- The pipeline defaults to `reduction.method: "umap"` in both topic configs. The plan recommends defaulting to `"none"` and treating UMAP/PCA as ablations. Consider changing the default in configs.
- Embedding and reduction of large corpora (700k+ paragraphs) will take significant time on CPU. Set `embedding.device: "cuda"` if a GPU is available.
- `baselines_main()` loads `paragraph_embeddings_reduced.npy` — this requires running the full pipeline (including reduce step) before baselines. If `reduction.method` is `"none"`, the reduced file won't exist. Consider falling back to raw embeddings.
- The `experiment_main()` ablation runner reruns the full pipeline per experiment. For expensive steps (embedding), consider sharing cached embeddings across ablations that don't change the embedding model.
