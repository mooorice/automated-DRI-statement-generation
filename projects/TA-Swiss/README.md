# TA-Swiss: LLMs, skills and qualifications

Advantages and risks of LLMs for skills and qualifications in work and
education. The first case built entirely with the generalised pipeline
(LLM-assisted subcategories and keywords, Swissdox fetch, DRI process).

## Reproduce

```bash
python -m autoconcourse.subcategories --config projects/TA-Swiss/config.yaml   # from the seed clusters in config.yaml
python -m autoconcourse.keywords      --config projects/TA-Swiss/config.yaml
python scripts/fetch_swissdox.py --keywords-file projects/TA-Swiss/keywords.json   # needs liri_key in .env
python scripts/filter_swissdox_local.py ...                                        # keyword pre-filter → *.filtered.tsv
python -m autoconcourse.cli --config projects/TA-Swiss/config.yaml
```

| File | Content |
|---|---|
| `config.yaml` | DeepSeek V4 Pro via OpenRouter, `F2LLM-v2-160M` embeddings on CUDA, top-1000, DRI with 10 considerations + 6 preferences per subcategory |
| `subcategories.json` | final 8 subcategories (7 seed clusters + "AI on the job market") |
| `subcategories-by-claude.json` | earlier 7-subcategory LLM draft, kept for comparison |
| `keywords.json`, `anchors.json` | generated, EN/DE/FR/IT |
| `context/` *(local)* | project description, 3 focus-group reports, media/literature review |
| `data/` *(local)* | Swissdox download (`*.tsv.xz`), filtered TSV, paragraphs, embeddings |
| `outputs/` *(local)* | `anchor_guided/concourse.{json,csv}`; `concourse.before-jobmarket.*` is the run before subcategory 7 was added |

No reference concourse exists. Validation here is prospective: domain experts
judge the generated concourse.
