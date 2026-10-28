"""Configuration loading, validation, and typed access."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TopicConfig:
    name: str
    id: str
    languages: list[str]
    description: str | None = None    # Rich, human-readable framing of the topic
                                      # for LLM prompts. Both `keywords` and
                                      # `subcategories` use this as their topic
                                      # fallback before `name`.


@dataclass
class FilteringConfig:
    keywords_file: str | None = None
    require_keyword: bool = False
    min_length: int = 20
    languages: list[str] = field(default_factory=list)


@dataclass
class CorpusConfig:
    """Corpus loading config.

    `adapter` selects the input-format adapter (csv, jsonl, swissdox, text_dir).
    The column-mapping fields below are passed to adapters that read tabular
    formats; `options` is a free-form dict for adapter-specific extras.
    """

    adapter: str = "csv"
    path: str = ""
    text_column: str = "text"
    id_column: str | None = None
    language_column: str | None = None
    group_column: str | None = None
    source_id_column: str | None = None
    options: dict = field(default_factory=dict)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)


@dataclass
class LLMConfig:
    """LLM provider configuration.

    `provider` selects how api_key, base_url, and default headers are resolved:
      - "openrouter" (default): base_url = https://openrouter.ai/api/v1,
        api_key from OPENROUTER_API_KEY, sends optional OpenRouter ranking
        headers (HTTP-Referer, X-Title) if site_url/site_name are set.
      - "openai":              base_url = SDK default (api.openai.com),
        api_key from OPENAI_API_KEY.
      - "custom":              base_url and api_key_env must be set explicitly.

    base_url, api_key_env, and extra_headers override the provider defaults
    when set (useful e.g. for Azure OpenAI, local vLLM, or proxying).
    """

    provider: str = "openrouter"
    base_url: str | None = None
    api_key_env: str | None = None
    default_model: str = "openai/gpt-4o"
    extra_headers: dict[str, str] = field(default_factory=dict)
    # OpenRouter ranking headers — surfaced only when provider == "openrouter".
    site_url: str | None = None     # → HTTP-Referer
    site_name: str | None = None    # → X-Title


@dataclass
class AnchorConfig:
    """LLM-generated anchor texts (one per subcategory × language).

    Strict, self-contained context loading — mirrors `keywords` and
    `subcategories`. Both `context_folder` (auto-loads every supported
    file inside) and `context_files` (explicit list) are supported and
    stack. Missing files raise a hard error.
    """

    enabled: bool = True
    method: str = "llm_generated"
    anchors_file: str | None = None
    subcategories_file: str = ""
    # Folder; ALL supported files inside (.txt/.md/.pdf, recursive) are
    # loaded as grounding context for every anchor generation call.
    context_folder: str | None = None
    # Additional individual files, stacked on top of context_folder.
    context_files: list[str] = field(default_factory=list)
    # Languages to generate anchors in. Default is monolingual English; for a
    # multilingual corpus, set to e.g. ["de", "fr", "it"] so each subcategory
    # has one anchor per language.
    languages: list[str] = field(default_factory=lambda: ["en"])
    model: str = "openai/gpt-4o"
    anchors_per_subcategory: int = 1


@dataclass
class KeywordsConfig:
    """LLM-generated keyword list for `corpus.filtering.keywords_file`.

    Run via `python -m autoconcourse.keywords --config <config.yaml>`.

    Strict, self-contained section: keyword generation does NOT pull defaults
    from anchors.* or other unrelated sections. Each field's resolution order
    (highest priority first):

      topic              → CLI --topic > keywords.topic > topic.description > topic.name
      languages          → CLI --language > keywords.languages > topic.languages
      per_language       → CLI --per-language > keywords.per_language > 50
      context paths      → CLI --context-file (stacked) + keywords.context_folder
                            (all files in dir) + keywords.context_files. Never
                            inherits from anchors.context_document.
      subcategories_file → CLI --subcategories > keywords.subcategories_file > none
      model              → CLI --model > keywords.model > llm.default_model
      output_file        → CLI --output > keywords.output_file
                            > corpus.filtering.keywords_file
    """

    topic: str | None = None
    per_language: int = 50
    languages: list[str] = field(default_factory=list)
    context_folder: str | None = None
    context_files: list[str] = field(default_factory=list)
    subcategories_file: str | None = None
    model: str | None = None
    output_file: str | None = None


@dataclass
class SubcategoriesConfig:
    """LLM-generated subcategories taxonomy.

    Run via `python -m autoconcourse.subcategories --config <config.yaml>`.
    Output drops into anchors.subcategories_file (or `output_file` if set).

    Strict, self-contained section — no silent fallbacks to anchors.* for
    context. Two modes:

      - **Seeded**: user provides rough cluster definitions in plain text
        (`seeds` inline or `seeds_file`). The LLM produces one structured
        subcategory per seed, preserving order, names, and intent.
      - **Unseeded**: LLM generates `count` subcategories from topic + context.

    Resolution order per field:
      topic         → CLI --topic > subcategories.topic > topic.description > topic.name
      language      → CLI --language > subcategories.language > topic.languages[0] > "en"
      count         → CLI --count > len(seeds) > subcategories.count
      context paths → CLI flags + subcategories.context_folder
                      + subcategories.context_files (no anchors fallback)
      seeds         → CLI --seeds-file > subcategories.seeds_file
                      > subcategories.seeds (inline)
      model         → CLI --model > subcategories.model > llm.default_model
      output_file   → CLI --output > subcategories.output_file
                      > anchors.subcategories_file
    """

    topic: str | None = None
    count: int = 8
    language: str | None = None
    context_folder: str | None = None
    context_files: list[str] = field(default_factory=list)
    seeds: str | None = None             # Plain-text cluster definitions, inline
    seeds_file: str | None = None        # OR path to a .txt/.md file with the same
    model: str | None = None
    output_file: str | None = None


@dataclass
class EmbeddingConfig:
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class RetrievalConfig:
    method: str = "anchor_cosine"
    top_k: int = 500
    # How to combine per-anchor similarities within a subcategory:
    #   "max"  — best-matching anchor (multilingual-friendly)
    #   "mean" — average across anchors
    aggregation: str = "max"


@dataclass
class GroupsConfig:
    """Optional stratification by a corpus column (e.g. political leaning, source type).

    When enabled, retrieval runs once per group, producing a (group, subcategory)
    keyed result. The actual group → values mapping is read from `groups_file`.
    """

    enabled: bool = False
    groups_file: str | None = None
    column: str = "group"


@dataclass
class GenerationConfig:
    model: str = "openai/gpt-4o-mini"
    # Controls which generation workflow to run:
    #   - "q-concourse": generate one list of consideration statements
    #   - "dri": generate both consideration and preference statements
    process: str = "q-concourse"
    statements_per_group_per_subcategory: int = 5
    # DRI-specific overrides. When set, these take precedence over
    # `statements_per_group_per_subcategory` for that statement type.
    dri_considerations_per_group_per_subcategory: int | None = None
    dri_preferences_per_group_per_subcategory: int | None = None
    # Reference examples for DRI prompts. null → packaged defaults under
    # `autoconcourse/prompts/examples/`. Set to a path (absolute or relative
    # to the project root) to override. Set to "" to disable in-prompt
    # examples for that statement type.
    dri_consideration_examples_file: str | None = None
    dri_preference_examples_file: str | None = None
    # Cap the number of example lines injected into the prompt. null = all.
    dri_max_examples: int | None = 100
    output_language: str = "en"
    prompt_template: str = "default"
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 5.0
    # Optional: a string with a {group} placeholder, rendered into the
    # generation prompt when retrieval is stratified by groups. If null,
    # group labels are not surfaced to the LLM even when retrieval groups them.
    group_instruction_template: str | None = None


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    topic: TopicConfig
    llm: LLMConfig
    corpus: CorpusConfig
    anchors: AnchorConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    groups: GroupsConfig
    generation: GenerationConfig
    keywords: KeywordsConfig = field(default_factory=KeywordsConfig)
    subcategories: SubcategoriesConfig = field(default_factory=SubcategoriesConfig)

    # Set at load time
    _base_dir: Path = field(default_factory=lambda: Path("."))
    _config_path: Path | None = field(default=None, repr=False)
    _raw: dict = field(default_factory=dict, repr=False)

    def resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to the config base directory."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self._base_dir / p

    @property
    def config_hash(self) -> str:
        """Short hash of the raw config for output tracking."""
        raw_json = json.dumps(self._raw, sort_keys=True, default=str)
        return hashlib.sha256(raw_json.encode()).hexdigest()[:12]

    @property
    def output_dir(self) -> Path:
        """Project-local outputs directory.

        Sits next to the project's `data/` folder so caches, intermediate
        artifacts, and final deliverables all live inside the project.
        """
        return self.project_dir / "outputs"

    @property
    def cache_dir(self) -> Path:
        """Legacy alias for project-local artifacts root."""
        return self.project_data_dir

    @property
    def project_dir(self) -> Path:
        if self._config_path is not None:
            return self._config_path.parent
        return self._base_dir

    @property
    def project_data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def paragraphs_cache_path(self) -> Path:
        return self.project_data_dir / "paragraphs" / "paragraphs.csv"

    @property
    def embeddings_dir(self) -> Path:
        return self.project_data_dir / "embeddings"

    @property
    def paragraph_embeddings_path(self) -> Path:
        return self.embeddings_dir / "paragraph_embeddings.npy"

    @property
    def anchor_embeddings_path(self) -> Path:
        return self.embeddings_dir / "anchor_embeddings.npy"

    @property
    def anchors_cache_path(self) -> Path:
        return self.project_data_dir / "anchors" / "anchors.csv"


def _build_dataclass(cls, data: dict | None):
    """Recursively build a dataclass from a dict, handling nested dataclasses."""
    if data is None:
        return cls()

    # Get field types from the dataclass
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key.startswith("_"):
            continue
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Check if the field is itself a dataclass
        if isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__"):
            kwargs[key] = _build_dataclass(field_type, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def _find_project_root(start: Path) -> Path:
    """Walk upward from `start` until a pyproject.toml or .git is found.

    Falls back to the current working directory if no marker is found. This
    lets configs live anywhere in the tree (projects/<topic>/config.yaml)
    while paths in the config still resolve relative to the project root.
    """
    p = start.resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").is_dir():
            return candidate
    return Path.cwd()


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load and validate a YAML config file, returning a typed PipelineConfig."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    base_dir = _find_project_root(config_path)

    topic = _build_dataclass(TopicConfig, raw.get("topic"))
    llm = _build_dataclass(LLMConfig, raw.get("llm"))
    corpus = _build_dataclass(CorpusConfig, raw.get("corpus"))
    anchors = _build_dataclass(AnchorConfig, raw.get("anchors"))
    embedding = _build_dataclass(EmbeddingConfig, raw.get("embedding"))
    retrieval = _build_dataclass(RetrievalConfig, raw.get("retrieval"))
    groups = _build_dataclass(GroupsConfig, raw.get("groups"))
    generation = _build_dataclass(GenerationConfig, raw.get("generation"))
    keywords = _build_dataclass(KeywordsConfig, raw.get("keywords"))
    subcategories = _build_dataclass(SubcategoriesConfig, raw.get("subcategories"))

    config = PipelineConfig(
        topic=topic,
        llm=llm,
        corpus=corpus,
        anchors=anchors,
        embedding=embedding,
        retrieval=retrieval,
        groups=groups,
        generation=generation,
        keywords=keywords,
        subcategories=subcategories,
        _base_dir=base_dir,
        _config_path=config_path.resolve(),
        _raw=raw,
    )

    _validate_config(config)
    return config


def _validate_config(config: PipelineConfig) -> None:
    """Validate required fields and path existence."""
    errors = []

    if not config.topic.name:
        errors.append("topic.name is required")
    if not config.topic.id:
        errors.append("topic.id is required")
    if not config.topic.languages:
        errors.append("topic.languages must have at least one language")

    if config.corpus.path and not config.resolve_path(config.corpus.path).exists():
        # Only warn — corpus might not exist yet if pipeline hasn't run
        pass

    if config.anchors.enabled:
        if config.anchors.subcategories_file:
            p = config.resolve_path(config.anchors.subcategories_file)
            if not p.exists():
                errors.append(f"anchors.subcategories_file not found: {p}")
        if config.anchors.method == "from_file" and config.anchors.anchors_file:
            p = config.resolve_path(config.anchors.anchors_file)
            if not p.exists():
                errors.append(f"anchors.anchors_file not found: {p}")

    if config.groups.enabled and config.groups.groups_file:
        p = config.resolve_path(config.groups.groups_file)
        if not p.exists():
            errors.append(f"groups.groups_file not found: {p}")

    if config.retrieval.aggregation not in ("max", "mean"):
        errors.append(f"retrieval.aggregation must be 'max' or 'mean', got '{config.retrieval.aggregation}'")

    if config.generation.process not in ("q-concourse", "dri"):
        errors.append(
            "generation.process must be 'q-concourse' or 'dri', "
            f"got '{config.generation.process}'"
        )

    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))
