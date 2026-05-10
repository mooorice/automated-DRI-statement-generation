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
    base_url: str | None = None
    api_key_env: str | None = None
    default_model: str = "openai/gpt-4o"
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class AnchorConfig:
    enabled: bool = True
    method: str = "llm_generated"
    anchors_file: str | None = None
    subcategories_file: str = ""
    context_document: str | None = None
    # Languages to generate anchors in. Default is monolingual English; for a
    # multilingual corpus, set to e.g. ["de", "fr", "it"] so each subcategory
    # has one anchor per language.
    languages: list[str] = field(default_factory=lambda: ["en"])
    model: str = "openai/gpt-4o"
    anchors_per_subcategory: int = 1


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
    model: str = "openai/gpt-4o"
    statements_per_group_per_subcategory: int = 5
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

    # Set at load time
    _base_dir: Path = field(default_factory=lambda: Path("."))
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
        return self._base_dir / "outputs" / self.topic.id

    @property
    def cache_dir(self) -> Path:
        return self._base_dir / "cache" / self.topic.id


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
    lets configs live anywhere in the tree (configs/, data/example/<topic>/)
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

    config = PipelineConfig(
        topic=topic,
        llm=llm,
        corpus=corpus,
        anchors=anchors,
        embedding=embedding,
        retrieval=retrieval,
        groups=groups,
        generation=generation,
        _base_dir=base_dir,
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

    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))
