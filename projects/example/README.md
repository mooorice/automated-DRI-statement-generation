# Example topics

Two minimal, runnable example topics that document the canonical AutoConcourse data shape.

| Dir | Languages | Use it as a template for |
|---|---|---|
| `monolingual/` | English | Single-language corpora |
| `multilingual/` | DE / FR / IT | Multilingual corpora (one anchor per language per subcategory) |

Each is fully self-contained — config, corpus, subcategories, anchors. Run from the project root:

```bash
python -m autoconcourse.cli --config projects/example/monolingual/config.yaml
python -m autoconcourse.cli --config projects/example/multilingual/config.yaml
```

(Both call the LLM in the `generate` step. Use `--steps corpus,anchors,embed,retrieve` to dry-run everything except generation.)

## Files in each example

### `config.yaml`
Pipeline configuration. See `projects/example/schema.yaml` for the full set of options.

### `corpus.csv`
Canonical corpus shape, one passage per row. Required columns: `id`, `text`. Optional: `language`, `source`, plus any extra metadata you want preserved through the pipeline.

### `subcategories.json`
The thematic dimensions the pipeline organizes retrieval and generation along. Each entry:

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | int | yes | 0-indexed |
| `name` | str | yes | Short label |
| `description` | str | yes | What this subtopic covers |
| `topic_focus` | str | yes | Short focus phrase, used in the generation prompt |
| `anchor_prompt` | str | optional | Override prompt for LLM-based anchor generation |

### `anchors.json`
Pre-generated anchors (used when `anchors.method: from_file`). Each entry:

| Field | Type | Required | Notes |
|---|---|---|---|
| `category` | int | yes | Matches a `subcategories[].id` |
| `text` | str | yes | The anchor passage |
| `language` | str | optional | ISO code; multilingual examples use one entry per language per subcategory |
| `subcategory_name` | str | optional | For human readability |

### `keywords.json` (optional, monolingual example only)
Optional corpus filter. Either a flat list of strings or a `{lang_code: [...]}` dict. The pipeline drops passages that contain none of these keywords.

## Adding a new topic

1. `cp -r projects/example/monolingual projects/<your_topic>` (or multilingual).
2. Edit `config.yaml`: change `topic.name`, `topic.id`, point `corpus.path` and the anchor/subcategory paths at your new dir.
3. Replace `corpus.csv` with your data (canonical columns: `id`, `text`, optional `language`/`source`/etc.).
4. Edit `subcategories.json` for your topic.
5. Decide anchor method:
   - `from_file`: hand-write `anchors.json`.
   - `llm_generated`: leave the anchor file out, set `anchors.method: llm_generated` in config — the pipeline will generate them on first run.
6. Run: `python -m autoconcourse.cli --config projects/<your_topic>/config.yaml`.

For non-CSV inputs, change `corpus.adapter` to `jsonl`, `swissdox`, or `text_dir` and adjust column-mapping fields.
