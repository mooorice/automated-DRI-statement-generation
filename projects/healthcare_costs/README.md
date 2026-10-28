# Healthcare costs in Switzerland

The original Master's thesis case: what drives rising health costs and what
to do about them, from Swiss news in DE/FR/IT, stratified by the political
leaning of outlets.

| File | Origin |
|---|---|
| `config.yaml` | ported from the thesis-era `paper/configs/healthcare_ch.yaml` to the current schema |
| `subcategories.json` | the 8 thesis subcategories (general costs, reforms, utilisation, hospital planning, insurance design, prevention incentives, coordination, household burden) |
| `keywords.json` | thesis keyword filter (DE/FR/IT) |
| `groups.json` | outlet `medium_code` → 5 political-leaning groups |
| `context/` *(local)* | `B_Politikfeldanalyse_Gesundheitskosten_2024_08_02_INTERFACE.pdf` — policy analysis used to ground anchors |

## Missing before this case can run

- **Corpus.** The thesis Swissdox download is no longer on disk. Re-fetch it
  with `scripts/fetch_swissdox.py` (thesis window: 2018-01-01 – 2024-08-29) to
  `data/swissdox_download.tsv.xz`.
- **Reference concourse.** Put the human DRI survey used for this topic in
  `reference/reference_concourse.json` (same shape as
  `projects/food_policy/reference/`).

Thesis-era intermediate outputs (top-500 passages per group, example
statements) are kept in `legacy/data/05_LLM_data/`.
