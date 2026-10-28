# paper/ — parked validation work

This directory holds the EMNLP-paper-specific scaffolding (baselines B1–B6, evaluation metrics, ablations A1–A8, prompt optimization) that was lifted out of the bare-method package during the strip-back.

## Why it's here

The strip-back's goal was a small, generalizable bare method (`autoconcourse/`). Everything in `paper/` exists only to validate that method against a reference concourse and compete with baselines — work that only matters once the bare method is solid and we have data to evaluate against.

The plan: build the bare method, validate it on real data, *then* reattach this code on top. None of it is deleted.

## Reattachment notes

When restarting paper work:

1. **Schema gaps.** Several modules reference config sections that no longer exist in the bare-method config schema:
   - `evaluation.reference_file`, `evaluation.coverage_thresholds` (used by `evaluation.py` and `cli.evaluation_main`)
   - `reduction.method`, `reduction.n_components`, `reduction.umap_params` (used by `cli.ABLATIONS` A1–A3)
   - `perspective_groups` (renamed to `groups` in the bare method — `cli.ABLATIONS` A4 needs updating)

   The case configs in `projects/<case>/config.yaml` already carry an `evaluation:` block (ignored by the bare-method loader, available via the raw YAML).
   - `generation.prompt_optimization` (used by `prompt_optimization.py`)

   Either restore these sections in `autoconcourse/config.py`, or extend the config in a paper-specific overlay.

2. **`reduction.py` is gone.** The dimensionality-reduction module was deleted because the bare method doesn't use it. Any baseline that relied on reduced embeddings (e.g. B4 BERTopic) needs a re-implementation that operates on raw embeddings, or `reduction.py` needs to be reintroduced.

3. **Multilingual retrieval changed.** The bare method scores passages as `max` (or `mean`) over per-language anchor embeddings. Baselines that aggregate anchor embeddings differently (`aggregate_anchor_embeddings` in `autoconcourse.retrieval`) may need updating.

4. **Paper-only deps.** Install with `pip install -e ".[paper]"` to pull in `rank-bm25`, `hdbscan`, `umap-learn`.

## Layout

```
paper/
├── __init__.py
├── README.md             # this file
├── cli.py                # baselines_main, evaluation_main, experiment_main
├── baselines.py          # B1–B6 implementations
├── evaluation.py         # coverage, groundedness, distinctiveness, balance, novelty
├── prompt_optimization.py
├── prompts/
│   ├── policy_option_generation.txt
│   └── prompt_optimization.txt
├── data/
│   └── example_reference_concourse.json
├── run_baselines.py
├── run_evaluation.py
└── run_experiment.py
```

Case data and configs (healthcare costs, food policy, health prevention, TA-Swiss) live in `projects/<case>/`; reference concourses go in `projects/<case>/reference/`.
