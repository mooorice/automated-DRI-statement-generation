# projects/

One folder per case. Every case uses the same layout:

```
projects/<case>/
  README.md             status, sources, what is missing
  config.yaml           pipeline config (paths relative to repo root)
  subcategories.json    taxonomy driving anchors and retrieval
  keywords.json         corpus pre-filter / Swissdox query terms
  anchors.json          generated anchors (one per subcategory × language)
  groups.json           optional stratification, e.g. outlet → political leaning
  reference/            human reference concourse(s) for validation   [tracked]
  context/              grounding documents for anchor generation       [local only]
  data/                 raw corpus + pipeline artifacts                 [local only]
  outputs/              generated concourses                             [local only]
```

`context/`, `data/` and `outputs/` are git-ignored: corpora are Swissdox-licensed
full text or deliberation transcripts, and some context documents are
confidential. Each case README says where they come from.

## Cases

| Case | Languages | Corpus | Process | Reference concourse | Status |
|---|---|---|---|---|---|
| [`healthcare_costs/`](healthcare_costs/) | DE/FR/IT | Swissdox news | DRI | missing | config ported, corpus must be re-fetched |
| [`food_policy/`](food_policy/) | DE/FR/IT | Swissdox news | DRI | Mathys (2024) SCAFP survey, 41 + 8 | config ported, corpus must be re-fetched |
| [`health_prevention/`](health_prevention/) | DE/FR/IT | Bevölkerungsrat transcripts | DRI | to extract from final report | data only, needs subcategories + config |
| [`TA-Swiss/`](TA-Swiss/) | EN/DE/FR/IT | Swissdox news | DRI | none (prospective, expert-judged) | runs end to end |
| [`example/`](example/) | EN; DE/FR/IT | toy CSV | q-concourse | — | runnable demos used by tests |

`example/schema.yaml` documents every config field.
