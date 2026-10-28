# Health prevention (Bevölkerungsrat)

A different corpus type for the same pipeline: transcripts of a deliberative
mini-public (Bevölkerungsrat on health prevention; source folder was dated 7.25) instead of news.

| Path | Content |
|---|---|
| `data/transcripts/` *(local)* | 114 transcript files, sessions S1–S4, DE/FR/IT regional groups, breakout and plenum. `*_aggr.txt` files (27) aggregate the per-facilitator files of a session. |
| `data/merged_transcripts.txt` *(local)* | all transcripts concatenated |
| `context/` *(local)* | `DE_Abschlussbereicht_Bev-Rat_E11_digital.pdf` — the citizens' assembly's final report |
| `prior_runs/` | two 25-statement consideration sets from earlier runs: from transcripts only, and from transcripts + final report |

Participants appear as numeric IDs, but file names carry facilitators' first
names. Keep the transcripts out of git and check consent terms before any
release. (They were committed in 5ff6f6e, so they remain in git history and on
origin/main.)

## Missing before this case can run

- `subcategories.json` (and anchors) for the topic.
- `config.yaml` — use the `text_dir` adapter on `data/transcripts/` with
  `options: {glob: "*_aggr.txt"}`, or another de-duplicated file set, so
  per-facilitator and aggregated files are not counted twice.
- **Reference concourse.** The DRI survey used in the assembly, as
  `reference/reference_concourse.json`.
