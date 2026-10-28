# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoConcourse generates concourses (sets of opinion statements covering the
discourse on a topic, for Q-methodology or DRI surveys) from a text corpus:
anchor-guided dense retrieval per subcategory, then LLM synthesis. It began as
a Master's thesis on Swiss health costs (DRI statements, DE/FR/IT) and is
being generalised and validated for publication.

## Environment

- Python 3.11+; install with `pip install -e .` (add `".[paper]"` for baselines/evaluation deps).
- API keys in `.env` via python-dotenv: `OPENROUTER_API_KEY` (default provider) or `OPENAI_API_KEY`; `liri_key` for Swissdox.
- Tests run fully offline (stubbed embedder and LLM): `python -m pytest -q`.

## Layout

- `autoconcourse/` — the bare method. Pipeline steps `corpus → anchors → embed → retrieve → generate`, run by `python -m autoconcourse.cli --config <config.yaml> [--steps ...] [--force]`.
  - `adapters/` convert raw input (csv, jsonl, swissdox, text_dir) into the canonical corpus shape (`id`, `text`, optional `language`, `group`, `source_id`). The rest of the pipeline never sees source-specific columns.
  - `subcategories.py`, `keywords.py` — LLM-assisted bootstrapping of a new topic (`python -m autoconcourse.subcategories|keywords --config ...`).
  - `prompts/*.txt` — prompt templates with `[SYSTEM]`/`[USER]` sections; `prompts/examples/` holds the DRI example bank.
  - `generation.process`: `q-concourse` (considerations) or `dri` (considerations + preferences).
- `projects/<case>/` — one self-contained folder per case: `config.yaml`, `subcategories.json`, `keywords.json`, `anchors.json`, `groups.json`, `reference/` (tracked); `context/`, `data/`, `outputs/` (git-ignored: licensed corpora, transcripts, confidential documents, regenerable artifacts). See `projects/README.md` for case status.
- `paper/` — parked validation code (baselines B1–B6, evaluation metrics, ablations, prompt optimisation). Must stay a plug-in on top of `autoconcourse/`; see `paper/README.md` for the schema gaps to bridge.
- `scripts/` — Swissdox fetch and local keyword filter.
- `docs/` — planning docs (`docs/planning/`), thesis PDF, posters/talks.
- `legacy/` — original thesis notebooks and thesis-era data. Read-only reference.

## Key Technical Details

- Config paths are relative to the repo root (found by walking up to `pyproject.toml`/`.git`); artifacts go to `<project>/data/` and `<project>/outputs/` next to the config.
- Multilingual retrieval: one anchor per (subcategory × language); a passage's score is the max (or mean) over its subcategory's anchors.
- Retrieval streams similarity in chunks and embedding checkpoints per batch, so multi-GB corpora work.
- `concourse.json` is the output contract that `paper/` depends on — don't change its schema casually.
- No dimensionality reduction in the bare method (the thesis used UMAP).

# Main Task

Your task will be to improve the codebase to make this a proper scientific contribution. We will generalize the method for concourse generation and build a robust pipeline for reproducibility. We validate test that pipeline across at least 3 sufficiently different cases. We will present the work at CompText conference and submit to EMNLP conference for peer reviewed publication.
