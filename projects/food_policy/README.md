# Future of Switzerland's food system

The thesis validation case: the method was run on a topic for which a
traditionally constructed DRI survey already existed, and the outputs were
compared by hand.

| File | Origin |
|---|---|
| `config.yaml` | ported from the thesis-era `paper/configs/food_policy_ch.yaml` to the current schema |
| `subcategories.json`, `keywords.json` | thesis validation run |
| `groups.json` | outlet `medium_code` → 5 political-leaning groups (same mapping as healthcare_costs) |
| `reference/reference_concourse.json` | **Mathys (2024) SCAFP DRI survey**: 41 considerations + 8 policy preferences, extracted from the `validation_statement` column of `legacy/validation/data/06_comparison/*_summary.xlsx` |
| `context/` *(local)* | `Merged_Full_Validation_Anchor_Background_Information.pdf` (also tracked under `legacy/validation/data/99_anchor_data/`) |

The thesis's manual comparison (top-5 nearest generated statements per
reference item, with ratings) is in `legacy/validation/data/06_comparison/`.

## Missing before this case can run

- **Corpus.** Re-fetch the Swissdox download with `scripts/fetch_swissdox.py`
  to `data/swissdox_download.tsv.xz`.

## Leakage note

The packaged DRI example bank (`autoconcourse/prompts/examples/`) contains
7 of the 41 reference considerations and 3 of the 8 preferences, so
`config.yaml` disables in-prompt examples for this case.
