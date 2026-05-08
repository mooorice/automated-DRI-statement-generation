# From Corpora to Concourse
## Automated Extraction of Representative Public Opinion Statements from News Text

### EMNLP Submission — Complete Preparation Plan

---

## The Paper in One Paragraph

Public opinion research, survey design, and deliberative democracy assessment all require **concourses**: structured sets of statements that represent the landscape of perspectives on a topic. Constructing concourses is expensive, subjective, and slow — typically requiring expert interviews, media analysis, and manual coding. We formalize **Representative Discourse Statement Generation (RDSG)** as an NLP task: given a topic and a large news corpus, produce a set of statements that are *representative*, *grounded*, *distinct*, *balanced*, and *clear*. We propose an anchor-guided retrieval pipeline that uses dense embeddings and subcategory-specific semantic anchors to select the most relevant passages, then prompts an LLM to synthesize opinion statements. Experiments across three topics, two languages, five baselines, and multiple LLMs show that anchor-guided retrieval significantly outperforms both zero-context generation and unsupervised alternatives. Human evaluation confirms that generated concourses match 50–65% of expert-crafted reference statements while surfacing perspectives experts missed.

---

## 0. Framing Decisions

### What this paper IS about
- A new NLP task (RDSG) with formal definition and evaluation framework
- A retrieval-augmented method for solving it
- Empirical evidence across topics, languages, and models
- Practical utility for social science (Q-methodology, survey design, deliberation research)

### What this paper is NOT about
- Deliberative democracy theory (mention once in intro, brief paragraph in related work)
- The DRI specifically (one application among several; mentioned in related work / experiments)
- The Bevölkerungsrat 2025 (brief mention as a downstream deployment, not the focus)
- Swiss healthcare policy (one of three experimental topics, not the topic)

### Reviewer mindset to write for
An EMNLP reviewer will ask:
1. Is the task well-defined and interesting? → Phase 2
2. Is the method novel enough? → anchor-guided retrieval is the key contribution
3. Are the baselines fair and comprehensive? → Phase 3
4. Is the evaluation rigorous? → Phase 5
5. Does it generalize? → multiple topics, languages, models → Phase 4
6. Can I reproduce it? → Phase 8

---

## Phase 1: Repository & Pipeline (Week 1)

### 1.1 Audit the existing codebase

```
Clone https://github.com/mooorice/automated-DRI-statement-generation
```

- [ ] Map every script into a pipeline DAG:
  ```
  keywords → Swissdox query → raw articles
       ↓
  paragraph chunking → keyword filtering → clean paragraphs
       ↓                                        ↓
  anchor generation (LLM)              embedding (SBERT)
       ↓                                        ↓
  anchor embedding                     UMAP reduction
       ↓                                        ↓
       └──────── cosine similarity ─────────────┘
                        ↓
                top-k selection per category
                        ↓
                LLM statement generation
                        ↓
                generated concourse
  ```
- [ ] Identify all hardcoded values (topic names, paths, API keys, model names, hyperparameters)
- [ ] Note all external dependencies and versions

### 1.2 Refactor into a modular, config-driven pipeline

The goal: anyone can run the full pipeline on a new topic by providing a config file.

- [ ] Create `config/` directory with YAML configs per topic:
  ```yaml
  # config/healthcare.yaml
  topic: "Rising healthcare costs in Switzerland"
  languages: ["de", "fr", "it"]
  corpus:
    source: "swissdox"
    date_range: ["2018-01-01", "2024-08-29"]
    keywords_file: "data/healthcare/keywords.json"
  anchors:
    method: "llm_generated"  # or "manual"
    subcategories_file: "data/healthcare/subcategories.json"
    context_document: "data/healthcare/policy_analysis.pdf"  # optional
  embedding:
    model: "paraphrase-multilingual-MiniLM-L12-v2"
  reduction:
    method: "umap"
    n_components: 50
    n_neighbors: 30
    min_dist: 0.0
    metric: "cosine"
  retrieval:
    method: "anchor_cosine"  # or "bm25", "random", "bertopic"
    top_k: 500
  generation:
    model: "gpt-4o"
    statements_per_category: 5
    political_leanings: ["left", "left-liberal", "centrist", "right-liberal", "right"]
  ```

- [ ] Refactor scripts into modules:
  ```
  src/
    corpus.py          # data loading, chunking, filtering
    embedding.py       # embed paragraphs and anchors
    reduction.py       # UMAP or alternatives
    retrieval.py       # anchor-guided, BM25, random, BERTopic
    generation.py      # LLM prompting for statement generation
    evaluation.py      # all automatic metrics
    baselines.py       # zero-context, extractive, etc.
  run_pipeline.py      # orchestrates everything from config
  run_evaluation.py    # computes all metrics given system outputs + references
  ```

- [ ] Create `requirements.txt` with pinned versions
- [ ] Write a minimal `README.md` with setup and usage

### 1.3 Data release plan

- [ ] Determine Swissdox redistribution terms → likely can't share raw articles
- [ ] Plan to release:
  - All config files
  - All keywords and anchor definitions
  - All prompts (system + generation)
  - All generated outputs (the concourses themselves)
  - All evaluation results and scores
  - Pre-computed embeddings (if permitted)
  - Reference concourses (human-generated)
  - Evaluation annotation data
- [ ] For the English topic (Phase 4): use an open corpus so this experiment is fully reproducible

### 1.4 Directory structure

```
project/
  configs/
    healthcare.yaml
    food_policy.yaml
    english_topic.yaml
  data/
    healthcare/
      keywords.json
      subcategories.json
      anchors.json
      political_leanings.json
    food_policy/
      ...
    english_topic/
      ...
  src/
    ...
  outputs/
    healthcare/
      main_system/
        statements.json
      baseline_zero_context/
        statements.json
      baseline_random/
        statements.json
      ...
    food_policy/
      ...
  evaluation/
    metrics/
      results_all.json
    human/
      annotation_guidelines.md
      annotations.csv
  paper/
    main.tex
    figures/
    tables/
```

---

## Phase 2: Task Formalization (Week 1–2)

This is a genuine contribution. No one has formally defined this task.

### 2.1 Task definition

Draft the formal definition:

> **Representative Discourse Statement Generation (RDSG)**
>
> **Input:**
> - A topic description *T* (natural language string)
> - A text corpus *C* = {d₁, ..., dₙ} of documents relevant to *T*
> - (Optional) A set of *k* subcategory descriptions *S* = {s₁, ..., sₖ} partitioning the discourse space
>
> **Output:**
> - A concourse *Q* = {q₁, ..., qₘ} of opinion statements, where each qᵢ is a natural language sentence expressing a position, value, consideration, or policy preference related to *T*
>
> **Desiderata** (ranked by importance for evaluation):
>
> | Property | Definition | Why it matters |
> |---|---|---|
> | **Coverage** | Q spans the major perspectives in C | Missing a perspective invalidates downstream use |
> | **Groundedness** | Each qᵢ is supported by content in C | Prevents LLM hallucination/injection of parametric bias |
> | **Distinctiveness** | Statements are semantically diverse | Redundant statements waste survey slots |
> | **Balance** | Perspectives are proportionally represented | Over-representing one view skews results |
> | **Clarity** | Statements are well-formed and interpretable | Participants must understand what they're rating |
> | **Neutrality** | Statements are framed so agreement/disagreement are both reasonable | Leading questions bias responses |

- [ ] Write this up as Section 3 of the paper (~0.5 pages)
- [ ] Discuss trade-offs: coverage vs. distinctiveness (more statements = more coverage but risk redundancy); groundedness vs. balance (corpus may over-represent dominant views)

### 2.2 Relationship to existing tasks

Position RDSG relative to known NLP tasks. This goes in Related Work but the distinctions should be clear in the task definition too.

| Existing task | How RDSG differs |
|---|---|
| Opinion summarization | RDSG produces *individual statements*, not a summary paragraph. Statements must be ratable (agree/disagree). |
| Aspect-based sentiment | RDSG extracts the *landscape* of positions, not sentiment toward known aspects |
| Argument mining | RDSG identifies *what people think*, not argument structures (premise→conclusion) |
| Abstractive summarization | RDSG produces multiple independent outputs, not one coherent text |
| Survey question generation | RDSG generates opinion statements, not interrogative questions; must cover a discourse space, not just a knowledge domain |
| Keyphrase extraction | RDSG outputs full sentences representing positions, not keywords |

---

## Phase 3: Baselines & Systems (Weeks 2–4)

### System inventory

Every system must produce the same output format: a JSON file with a list of statements, each tagged with its subcategory.

```json
{
  "system": "anchor_guided_gpt4o",
  "topic": "healthcare",
  "statements": [
    {
      "id": "hc_001",
      "text": "The mandatory health insurance...",
      "subcategory": "insurance_organization",
      "source_paragraphs": ["p_12345", "p_67890"]  // for groundedness eval
    }
  ]
}
```

### 3.1 Zero-context generation (Baseline B1)

**What:** Give the LLM only the topic name + subcategory names. No corpus access.

**Why it matters:** If this works just as well, the entire retrieval pipeline is unnecessary. This is the baseline reviewers will check first.

- [ ] Prompt template:
  ```
  Generate {m} representative public opinion statements about "{topic}".
  The statements should cover these subtopics: {subcategory_list}.
  Each statement should be a single declarative sentence that a person could
  agree or disagree with. Statements should represent the diversity of
  real public opinion, including positions from different political perspectives.
  ```
- [ ] Run with: GPT-4o, GPT-4o-mini, Llama 3.1 70B, Llama 3.1 8B
- [ ] Use the same number of output statements as the main system (fair comparison)

### 3.2 Random retrieval + generation (Baseline B2)

**What:** Retrieve random paragraphs from the corpus (same budget: 500 per subcategory), feed to LLM.

**Why it matters:** Tests whether *any* corpus context helps, or whether *targeted* retrieval is needed.

- [ ] Sample 500 random paragraphs from the keyword-filtered corpus per subcategory
- [ ] Feed to LLM with same generation prompt as main system
- [ ] Run 3 times with different random seeds, report mean ± std

### 3.3 BM25 retrieval + generation (Baseline B3)

**What:** Use sparse keyword retrieval instead of dense embeddings.

**Why it matters:** Tests whether dense retrieval outperforms traditional IR for this task.

- [ ] Index all paragraphs with BM25 (`rank_bm25` or Pyserini)
- [ ] Use subcategory anchor text as query
- [ ] Retrieve top 500 per subcategory
- [ ] Feed to same LLM with same prompt

### 3.4 BERTopic clustering + generation (Baseline B4)

**What:** Unsupervised clustering to organize the corpus, then select representative paragraphs from each cluster.

**Why it matters:** Tests guided (anchor) vs. unguided (clustering) corpus organization. You already showed in the thesis that this doesn't work well — now formalize it as an empirical comparison.

- [ ] Run BERTopic (HDBSCAN + c-TF-IDF) on embedded paragraphs
- [ ] Map clusters to subcategories by maximum centroid similarity
- [ ] Select top 500 paragraphs per mapped subcategory
- [ ] Feed to LLM

### 3.5 Extractive selection (Baseline B5)

**What:** No LLM generation. Select the most representative existing sentences from the corpus.

**Why it matters:** Tests whether LLM *generation* adds value over pure *extraction*.

- [ ] From top-500 paragraphs per category, split into sentences
- [ ] Rank by centrality: average cosine similarity to all other sentences in the category
- [ ] Select top-m, skipping any sentence with >0.85 similarity to an already-selected one
- [ ] Light cleanup only (no rewriting)

### 3.6 Dense retrieval without anchors (Baseline B6)

**What:** Use a single general topic embedding instead of per-subcategory anchors. Retrieve top-k globally.

**Why it matters:** Isolates the contribution of the anchor-guided subcategory structure.

- [ ] Embed only the general topic description (not subcategory anchors)
- [ ] Compute cosine similarity to all paragraphs
- [ ] Select top (500 × k) paragraphs globally
- [ ] Feed to LLM with subcategory structure in the prompt only (not in the retrieval)

### 3.7 Main system: Anchor-guided retrieval + generation

- [ ] Run full pipeline as described in thesis
- [ ] Parameterize everything through config

### 3.8 Ablations of main system

Each ablation changes one component, holding everything else constant:

| Ablation | Change |
|---|---|
| **A1: No UMAP** | Cosine similarity on raw 384-dim embeddings |
| **A2: No political leaning split** | Pool all outlets, don't stratify by political orientation |
| **A3: Top-k variation** | k ∈ {50, 100, 500, 1000, 5000} |
| **A4: Embedding model** | Replace paraphrase-multilingual-MiniLM with: multilingual-e5-large, BGE-M3 |
| **A5: Generator model** | Replace GPT-4o with: GPT-4o-mini, Claude Sonnet, Llama 3.1 70B |
| **A6: UMAP dimensions** | n_components ∈ {10, 25, 50, 100, 200} |
| **A7: Number of anchors per subcategory** | 1 anchor (current) vs. 3 anchors vs. 5 anchors per subcategory |

---

## Phase 4: Topics & Corpora (Weeks 2–4, parallel with Phase 3)

### 4.1 Topic 1: Swiss healthcare costs (German/French/Italian)

- Source: Swissdox
- Status: Complete from thesis
- Reference concourse: The DRI survey used in the Bevölkerungsrat 2025 (28 statements + 8 policy options)
- [ ] Verify all outputs are saved in standard format
- [ ] Document corpus statistics (articles, paragraphs after filtering, language distribution)

### 4.2 Topic 2: Swiss food policy (German/French/Italian)

- Source: Swissdox
- Status: Complete from thesis (validation experiment)
- Reference concourse: Mathys (2024) SCAFP DRI survey (41 statements + 8 policy options)
- [ ] Verify all outputs are saved in standard format
- [ ] Document corpus statistics

### 4.3 Topic 3: English-language topic (NEW — critical for EMNLP)

Must be in English. Must have (or you must create) a reference concourse.

**Recommended: Climate change policy**

Reasons:
- Abundant open English news data available
- Multiple existing Q-studies with published concourses (e.g., Danielson et al. 2013; Nies & Hoppe 2024)
- Topic complexity comparable to healthcare and food policy
- Broad public interest, recognizable to reviewers

Alternative options: immigration policy, AI regulation, housing affordability.

Steps:
- [ ] **Corpus assembly:**
  - Option A: Use an existing dataset (e.g., climate news from HuggingFace, GDELT filtered, or All The News dataset)
  - Option B: Scrape open-access news sources (The Guardian, BBC, AP, Reuters all have permissive terms for research)
  - Option C: Use the MediaCloud API or similar open research tool
  - Target: 50k–500k paragraphs minimum
- [ ] **Subcategory definition:**
  - Draw from existing climate discourse frameworks (e.g., IPCC categories, or discourse analysis literature)
  - Example subcategories: mitigation vs. adaptation, economic impacts, energy transition, climate justice, individual vs. systemic responsibility, international cooperation, technological solutions
- [ ] **Anchor generation:**
  - Same process as Swiss topics: prompt LLM with subcategory name + context document
  - Use IPCC summary for policymakers or similar as context document
- [ ] **Reference concourse:**
  - Best case: find a published Q-study on climate policy and use their concourse
  - Good case: recruit 2–3 researchers to independently write reference statements
  - Acceptable: create one yourself + have 1 colleague validate
- [ ] **Run full pipeline + all baselines**

### 4.4 Corpus statistics table (for the paper)

Prepare a table like:

| Topic | Languages | Articles | Paragraphs | Subcategories | Reference statements | Reference source |
|---|---|---|---|---|---|---|
| Healthcare (CH) | de, fr, it | 444,869 | 738,598 | 7 + general | 28 + 8 policy | Bevölkerungsrat 2025 |
| Food policy (CH) | de, fr, it | ~X | 393,258 | 7 + general | 41 + 8 policy | Mathys (2024) |
| Climate (EN) | en | ~Y | ~Z | 7 (tbd) | N (tbd) | Published Q-study / expert panel |

---

## Phase 5: Evaluation (Weeks 4–6)

### 5.1 Automatic metrics

Implement all metrics in a single evaluation script that takes:
- `system_output.json` (generated concourse)
- `reference.json` (human reference concourse)
- `corpus_embeddings.npy` (for groundedness)

And produces a results JSON with all scores.

#### Metric 1: Coverage

How much of the reference concourse does the system capture?

```python
def coverage(generated, reference, threshold=0.8):
    """For each reference statement, find best match in generated set."""
    ref_embeds = embed(reference)
    gen_embeds = embed(generated)
    similarities = cosine_similarity(ref_embeds, gen_embeds)
    best_match = similarities.max(axis=1)  # best match per reference item
    coverage_at_threshold = (best_match >= threshold).mean()
    mean_coverage = best_match.mean()
    return {
        f"coverage@{threshold}": coverage_at_threshold,
        "mean_coverage": mean_coverage,
        "per_item": best_match.tolist()
    }
```

- [ ] Report Coverage@0.7, Coverage@0.8, Coverage@0.9
- [ ] Report mean best-match similarity
- [ ] Identify uncovered reference items (for error analysis)

#### Metric 2: Groundedness

Are statements supported by the source corpus?

```python
def groundedness(generated, retrieved_paragraphs):
    """For each statement, how well is it supported by retrieved text?"""
    gen_embeds = embed(generated)
    para_embeds = embed(retrieved_paragraphs)
    similarities = cosine_similarity(gen_embeds, para_embeds)
    best_support = similarities.max(axis=1)
    return {
        "mean_grounding": best_support.mean(),
        "min_grounding": best_support.min(),
        "pct_grounded@0.7": (best_support >= 0.7).mean(),
        "per_item": best_support.tolist()
    }
```

- [ ] Also compute groundedness against the *full* corpus (not just retrieved paragraphs) — this catches cases where the statement is in the corpus but wasn't retrieved

#### Metric 3: Distinctiveness

Are statements diverse (not redundant)?

```python
def distinctiveness(generated):
    """Measure semantic diversity of the generated set."""
    embeds = embed(generated)
    pairwise = cosine_similarity(embeds, embeds)
    # Zero out diagonal
    np.fill_diagonal(pairwise, 0)
    mean_pairwise = pairwise.sum() / (len(generated) * (len(generated) - 1))
    max_pairwise = pairwise.max()
    n_near_duplicates = ((pairwise > 0.85).sum() - len(generated)) // 2
    return {
        "distinctiveness": 1 - mean_pairwise,
        "max_pairwise_sim": max_pairwise,
        "near_duplicates": n_near_duplicates
    }
```

#### Metric 4: Balance

Does the distribution across subcategories match the corpus?

```python
def balance(generated, subcategory_anchors, corpus_paragraphs):
    """Compare subcategory distribution of generated vs. corpus."""
    # Assign each generated statement to nearest subcategory
    gen_dist = assign_to_subcategories(generated, subcategory_anchors)
    # Compute corpus distribution
    corpus_dist = assign_to_subcategories(corpus_paragraphs, subcategory_anchors)
    # Jensen-Shannon divergence
    jsd = jensenshannon(gen_dist, corpus_dist)
    return {
        "balance_jsd": jsd,
        "generated_distribution": gen_dist.tolist(),
        "corpus_distribution": corpus_dist.tolist()
    }
```

#### Metric 5: Novelty (bonus)

Does the system find perspectives that the reference concourse missed?

```python
def novelty(generated, reference, threshold=0.6):
    """Statements in generated set that have NO close match in reference."""
    gen_embeds = embed(generated)
    ref_embeds = embed(reference)
    similarities = cosine_similarity(gen_embeds, ref_embeds)
    best_match_to_ref = similarities.max(axis=1)
    novel = (best_match_to_ref < threshold)
    return {
        "n_novel": novel.sum(),
        "pct_novel": novel.mean(),
        "novel_statements": [g for g, is_novel in zip(generated, novel) if is_novel]
    }
```

This is interesting because it captures cases where the AI finds valid perspectives that human experts overlooked.

### 5.2 Main results table structure

```
Table 1: Main results across three topics. Best per column in bold.

System                  | Coverage@0.8 | Grounding | Distinct. | Balance | 
                        | HC  FP  CL   | HC FP CL  | HC FP CL  | HC FP CL|
------------------------|--------------|-----------|-----------|---------|
B1: Zero-context        |              |           |           |         |
B2: Random retrieval    |              |           |           |         |
B3: BM25 retrieval      |              |           |           |         |
B4: BERTopic clustering |              |           |           |         |
B5: Extractive          |              |           |           |         |
B6: Dense (no anchors)  |              |           |           |         |
Ours: Anchor-guided     |              |           |           |         |

HC = Healthcare, FP = Food Policy, CL = Climate
```

### 5.3 Human evaluation

#### Design

- [ ] **Evaluators:** 3–5 people (mix of domain experts and NLP-aware researchers)
- [ ] **What they evaluate:** Statement sets from 3–4 systems (ours, zero-context, best baseline, human reference)
- [ ] **Blind:** Systems are anonymized (System A, B, C, D) and order randomized per annotator

#### Per-statement rating (Likert 1–5)

| Dimension | Question to annotator |
|---|---|
| Clarity | "Is this statement clear and unambiguous?" |
| Relevance | "Is this statement relevant to the topic of [topic]?" |
| Neutrality | "Is this statement framed so that a reasonable person could either agree or disagree?" |
| Specificity | "Does this statement make a specific enough claim to be meaningful?" |

#### Per-set comparative evaluation

For each subcategory, show annotators two anonymized statement sets side by side:

> "Which set better captures the range of public perspectives on [subcategory]?"
> Options: Set A / Set B / Tie

- [ ] Run all pairwise comparisons: ours vs. each baseline, ours vs. human reference
- [ ] Compute win rates

#### Inter-annotator agreement

- [ ] Compute Krippendorff's α for per-statement ratings
- [ ] Compute pairwise Cohen's κ for comparative judgments
- [ ] Report in paper; α > 0.4 is acceptable, > 0.6 is good

#### Logistics

- [ ] Write annotation guidelines document (1–2 pages, with examples of good/bad statements)
- [ ] Build annotation interface:
  - Simple option: Google Sheets with randomized rows per annotator
  - Better option: Label Studio or Potato (annotation tool for NLP)
- [ ] Pilot with 1 annotator on ~20 statements, refine guidelines
- [ ] Full annotation: target ~200 statements total (50 per system × 4 systems) rated by 3 annotators
- [ ] Budget: if paying annotators, ~4–6 hours per annotator at this scale

### 5.4 Human ceiling measurement

How much do human experts agree with each other?

- [ ] Have 2 experts independently create reference concourses for at least one topic
- [ ] Compute Coverage between their sets (same metric used for systems)
- [ ] This number becomes the ceiling: if experts achieve Coverage@0.8 = 0.65 against each other, a system achieving 0.60 is remarkably good

---

## Phase 6: Error Analysis & Taxonomy (Week 6–7)

### 6.1 Classify all misses

For every reference statement not covered by the main system (Coverage score < 0.7):

- [ ] Manually categorize the failure mode:

| Error type | Description | Example from thesis |
|---|---|---|
| **Corpus gap** | Perspective absent from news corpus entirely | Fringe view: "Climate warming benefits food production" |
| **Retrieval miss** | Perspective exists in corpus but not in top-k retrieved paragraphs | Relevant paragraphs ranked below cutoff |
| **Generation miss** | Relevant paragraphs retrieved but LLM didn't produce a matching statement | LLM defaulted to neutral/positive framing |
| **Framing mismatch** | Generated statement covers same topic but with wrong framing (too general, too specific, wrong valence) | General statement when specific was needed |
| **Perspective mismatch** | Statement addresses the right subtopic but from a different angle | Economic framing vs. rights-based framing |

### 6.2 Characterize LLM generation biases

From your thesis, you already identified:
- Positivity/neutrality bias (missing pessimistic framings)
- Missing first-person perspective statements
- Missing highly general statements
- Missing fringe/radical positions

- [ ] Quantify each bias across all three topics
- [ ] Test whether different LLMs show different bias profiles
- [ ] Report as a table: error type × system × topic

### 6.3 Qualitative examples for the paper

Select examples that tell a story:

- [ ] 2 examples where anchor-guided retrieval captures something baselines miss (demonstrates the method's value)
- [ ] 2 examples where the system matches a reference statement closely (demonstrates feasibility)
- [ ] 2 examples of interesting failures (demonstrates honesty and understanding)
- [ ] 1 example of a novel valid statement the system found that humans missed (demonstrates added value)

Format these as a figure or table in the paper with the generated statement, matched reference, and retrieved supporting paragraph.

---

## Phase 7: Paper Writing (Weeks 7–10)

### Target: 8 pages + unlimited references and appendix (EMNLP format)

### Section 1: Introduction (~1 page)

**Paragraph 1 — The problem:**
Understanding public discourse on a topic requires structured representations: not just "what are people talking about?" but "what are the distinct positions, values, and considerations in play?" Such representations — often called *concourses* in Q-methodology — are essential for survey design, deliberative process assessment, opinion mapping, and media analysis. Currently, building them requires labor-intensive expert analysis.

**Paragraph 2 — Why it's hard / why now:**
The task sits at an interesting intersection: it requires *coverage* of a discourse space (information retrieval), *synthesis* into well-formed statements (generation), and *balance* across perspectives (fairness). Recent advances in dense retrieval and LLM generation make automation feasible, but naive application risks hallucination, redundancy, and perspective bias.

**Paragraph 3 — What we do:**
We formalize Representative Discourse Statement Generation (RDSG) as an NLP task. We propose an anchor-guided retrieval pipeline: subcategory-specific semantic anchors guide dense retrieval from a large news corpus, and retrieved passages are fed to an LLM for statement synthesis. We evaluate across three topics (healthcare, food policy, climate), two language settings (multilingual, English), five baselines, and multiple LLMs.

**Paragraph 4 — Contributions:**
1. Formal task definition (RDSG) with evaluation framework
2. Anchor-guided retrieval method that outperforms baselines on coverage while maintaining groundedness
3. Comprehensive experiments showing generalization across topics, languages, and models
4. Error taxonomy revealing systematic biases in LLM-generated opinion statements
5. Open evaluation framework and generated datasets

**Do NOT include:**
- Extended motivation about deliberative democracy
- The word "DRI" (save for related work)
- Details about the Bevölkerungsrat

### Section 2: Related Work (~1.5 pages)

Organize by NLP task, not by political science subfield.

**2.1 Opinion mining and summarization** (~1 paragraph)
- Aspect-based sentiment analysis, opinion summarization
- Key distinction: RDSG produces *individual ratable statements*, not summaries

**2.2 Argument mining** (~1 paragraph)
- Argument extraction, stance detection
- Key distinction: RDSG extracts the *landscape of positions*, not argument structures

**2.3 Retrieval-augmented generation** (~1 paragraph)
- RAG fundamentals, dense retrieval
- Position anchor-guided retrieval as a structured variant of RAG

**2.4 LLMs for social science** (~1 paragraph)
- Silicon sampling (Argyle et al. 2023), LLM-as-judge, survey generation
- Tessler et al. 2024 (Habermas Machine) — finding common ground via LLMs
- Key distinction: RDSG generates the *input instrument*, not the deliberation itself

**2.5 Q-methodology and concourse construction** (~1 paragraph)
- Brief explanation of Q-sorts and why concourse quality matters
- DRI as one downstream application (Niemeyer & Veri 2022)
- Current manual process and its limitations

### Section 3: Task Definition (~0.5 pages)

- Formal definition from Phase 2
- Table of desiderata with definitions
- Brief discussion of trade-offs

### Section 4: Method (~1.5 pages)

**4.1 Overview** — Pipeline diagram (1 clean figure)

**4.2 Corpus preparation** — Keyword-based collection, paragraph chunking, keyword filtering. Mention multilingual aspect.

**4.3 Anchor generation** — LLM-prompted subcategory descriptions as semantic reference points. Explain why guided > unguided (foreshadow results).

**4.4 Embedding and dimensionality reduction** — SBERT model, UMAP. Brief justification of choices.

**4.5 Anchor-guided retrieval** — Cosine similarity to subcategory anchors, top-k selection, overlap analysis. This is the core contribution — give it space.

**4.6 Statement generation** — LLM prompting with retrieved context. Perspective-stratified generation. Describe the prompt structure (full prompts in appendix).

**4.7 Human review** — Brief mention that final concourses undergo expert review. Frame as a feature (human-in-the-loop), not a limitation.

### Section 5: Experimental Setup (~1 page)

**5.1 Topics and corpora** — Table with statistics (Phase 4.4)

**5.2 Reference concourses** — Where they come from, how many statements, who created them

**5.3 Baselines** — Table listing all systems (B1–B6 + ablations)

**5.4 Evaluation metrics** — Brief definitions (details in appendix); mention human evaluation

**5.5 Implementation details** — Models, hyperparameters, compute cost, number of API calls

### Section 6: Results (~1.5 pages)

**6.1 Main results** — Table 1 (all systems × all metrics × all topics). Key findings in prose.

**6.2 Human evaluation** — Table 2 (per-statement ratings + comparative wins). Inter-annotator agreement.

**6.3 Ablation study** — Table 3 (ablation variants). What matters most? Expected findings:
- Anchor guidance matters a lot (B6 vs. ours)
- Retrieval matters (B1 vs. ours)
- UMAP effect is modest
- Generator model matters less than retrieval quality
- Political leaning split helps balance but not coverage

**6.4 Cross-topic consistency** — Does the method's advantage hold across all three topics?

### Section 7: Analysis (~0.75 pages)

**7.1 Error analysis** — Taxonomy table, quantified across topics

**7.2 What anchors buy you** — Qualitative comparison of anchor-guided vs. unguided retrieval. Show examples where anchors surface subcategory-specific content that global retrieval misses.

**7.3 Novel perspectives** — Statements the system generated that weren't in the reference but are valid (novelty metric). This is a selling point: the system can surface perspectives human experts missed.

**7.4 Limitations** — Be direct:
- Corpus-dependent (garbage in, garbage out)
- News bias (mainstream perspectives overrepresented)
- LLM framing biases (positivity, avoiding extremes)
- Closed-model reproducibility concerns (mitigated by open-model experiments)
- Subcategory structure requires domain knowledge to define

### Section 8: Conclusion (~0.25 pages)

- RDSG is a well-defined, practically useful NLP task
- Anchor-guided retrieval is simple but effective
- The method generalizes across topics and languages
- Future: social media corpora, real-time discourse tracking, integration with deliberative platforms

### Required sections (EMNLP policy)

- [ ] **Ethics Statement** — Potential misuse (generating biased surveys), news corpus biases, human evaluation ethics, no PII in generated statements
- [ ] **Limitations** — Can be integrated into Section 7 or standalone
- [ ] **Acknowledgments** — Funding, compute, annotators (after acceptance; anonymous at submission)

---

## Phase 8: Appendix, Polish & Submit (Weeks 10–12)

### 8.1 Appendix contents (unlimited pages)

- [ ] A: Full prompt templates (anchor generation, statement generation, all baselines)
- [ ] B: Complete anchor definitions for all topics (all languages)
- [ ] C: Subcategory definitions and keywords for all topics
- [ ] D: Full per-subcategory results breakdown
- [ ] E: Complete ablation results
- [ ] F: Annotation guidelines for human evaluation
- [ ] G: Inter-annotator agreement details
- [ ] H: Example outputs from all systems (side-by-side comparison table)
- [ ] I: Corpus statistics and filtering details
- [ ] J: Datasheet (Gebru et al. 2021) for each dataset

### 8.2 Figures

- [ ] Figure 1: Pipeline diagram (clean, publication-quality)
- [ ] Figure 2: Coverage@threshold curves for all systems (one subplot per topic)
- [ ] Figure 3: Ablation results (bar chart or heatmap)
- [ ] Figure 4: Error type distribution across systems (stacked bar chart)
- [ ] Figure 5 (optional): t-SNE/UMAP visualization of generated statements colored by system, with reference statements overlaid

### 8.3 Tables

- [ ] Table 1: Corpus statistics
- [ ] Table 2: Main results (all systems × metrics × topics)
- [ ] Table 3: Human evaluation results
- [ ] Table 4: Ablation results
- [ ] Table 5: Error taxonomy with counts

### 8.4 Pre-submission checklist

- [ ] Paper compiles in EMNLP 2026 LaTeX template
- [ ] Within 8-page limit (excluding references and appendix)
- [ ] Fully anonymous: no author names, no identifiable GitHub links, no "our university" references
- [ ] All figures legible in grayscale and at reduced size
- [ ] All tables have captions above, all figures have captions below
- [ ] Ethics statement present
- [ ] Limitations section present
- [ ] Supplementary material properly organized
- [ ] Anonymous code repository prepared (e.g., anonymous GitHub)
- [ ] Responsible NLP checklist filled out (if EMNLP requires one)

### 8.5 Feedback loop

- [ ] Share draft with 2 NLP colleagues by week 10
- [ ] Incorporate feedback by week 11
- [ ] Final proofread by week 12
- [ ] Submit

---

## Timeline Summary

| Week | Phase | Key deliverables |
|---|---|---|
| 1 | Repo cleanup + task definition | Clean pipeline, config-driven, formal task definition drafted |
| 2 | Baselines + English topic setup | B1–B3 implemented, English corpus assembled |
| 3 | Remaining baselines + experiments | B4–B6 implemented, all systems running on topic 1 |
| 4 | Run everything | All systems × all topics × all ablations complete |
| 5 | Automatic evaluation + human eval prep | All metrics computed, annotation interface ready, pilot done |
| 6 | Human evaluation + analysis | Annotations collected, error analysis complete |
| 7 | Writing: sections 1–4 | Introduction, related work, task definition, method |
| 8 | Writing: sections 5–8 | Experiments, results, analysis, conclusion |
| 9 | Full draft complete | All figures, tables, appendix |
| 10 | External feedback | Draft to 2 colleagues |
| 11 | Revision | Incorporate feedback |
| 12 | Polish and submit | Final check, anonymous repo, submit |

---

## Claude Code Task List

### Priority 1: Infrastructure (Phase 1)

```
[ ] Refactor pipeline into config-driven modular scripts
[ ] Create standard output format (JSON schema for concourses)
[ ] Write run_pipeline.py that takes a config and produces outputs
[ ] Write run_evaluation.py that takes outputs + reference and produces all metrics
```

### Priority 2: Baselines (Phase 3)

```
[ ] Implement zero-context generation (B1) — just LLM prompting, no retrieval
[ ] Implement random retrieval (B2) — random paragraph sampling + LLM
[ ] Implement BM25 retrieval (B3) — sparse retrieval + LLM
[ ] Implement BERTopic baseline (B4) — clustering + selection + LLM
[ ] Implement extractive baseline (B5) — centrality-based sentence selection
[ ] Implement no-anchor dense retrieval (B6) — single topic query + LLM
```

### Priority 3: Evaluation (Phase 5)

```
[ ] Implement coverage metric (with threshold parameter)
[ ] Implement groundedness metric
[ ] Implement distinctiveness metric
[ ] Implement balance metric (JSD)
[ ] Implement novelty metric
[ ] Create results aggregation script → LaTeX tables
[ ] Create visualization scripts → figures
```

### Priority 4: Paper production (Phase 7–8)

```
[ ] Set up EMNLP LaTeX template
[ ] Generate all LaTeX tables from results JSON
[ ] Generate all figures (matplotlib/seaborn)
[ ] Create pipeline diagram (tikz or programmatic)
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Zero-context baseline matches our system | Fatal — invalidates retrieval pipeline | Unlikely for coverage+groundedness jointly, but if so, pivot: the finding itself is publishable ("LLMs already know public discourse") |
| No good English reference concourse exists | Weakens third experiment | Create your own with 2 experts; report human ceiling |
| Human evaluation too slow | Delays submission | Start with 2 annotators, 1 topic; scale later if accepted |
| Swissdox data can't be released | Reproducibility concern | English experiment uses open data; release embeddings + outputs for Swiss data |
| Open-weight LLMs perform much worse | Limits reproducibility story | Report honestly — this is an interesting finding about model capability |
| Paper rejected from main conference | Delays publication | Findings track (automatic consideration); NLP+CSS workshop; ICWSM as backup venue |
| Reviewer says "just engineering, no novelty" | Rejection risk | Emphasize: (1) task formalization, (2) anchor-guided retrieval as a contribution, (3) evaluation framework, (4) error taxonomy — four contributions, not just one pipeline |