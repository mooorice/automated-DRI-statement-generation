"""Configuration loading, validation, and typed access."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TopicConfig:
    name: str
    id: str
    languages: list[str]


@dataclass
class ChunkingConfig:
    method: str = "paragraph"
    min_length: int = 20


@dataclass
class FilteringConfig:
    keywords_file: str | None = None
    require_keyword: bool = True


@dataclass
class CorpusConfig:
    source: str = "csv"
    path: str = ""
    text_column: str = "text"
    metadata_columns: list[str] = field(default_factory=lambda: ["medium_code", "language", "id"])
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
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
    languages: list[str] = field(default_factory=lambda: ["de", "fr", "it"])
    model: str = "openai/gpt-4o"
    anchors_per_subcategory: int = 3


@dataclass
class EmbeddingConfig:
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class UMAPParams:
    n_neighbors: int = 30
    min_dist: float = 0.0
    metric: str = "cosine"


@dataclass
class ReductionConfig:
    method: str = "umap"
    n_components: int = 50
    umap_params: UMAPParams = field(default_factory=UMAPParams)


@dataclass
class RetrievalConfig:
    method: str = "anchor_cosine"
    top_k: int = 500


@dataclass
class PerspectiveGroupsConfig:
    enabled: bool = False
    groups_file: str | None = None
    metadata_column: str = "medium_code"


@dataclass
class PromptOptimizationConfig:
    enabled: bool = False
    iterations: int = 3
    optimization_model: str | None = None


@dataclass
class GenerationConfig:
    model: str = "openai/gpt-4o"
    statements_per_group_per_subcategory: int = 5
    output_language: str = "en"
    prompt_template: str = "default"
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 5.0
    prompt_optimization: PromptOptimizationConfig = field(default_factory=PromptOptimizationConfig)


@dataclass
class EvaluationConfig:
    reference_file: str | None = None
    coverage_thresholds: list[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    topic: TopicConfig
    llm: LLMConfig
    corpus: CorpusConfig
    anchors: AnchorConfig
    embedding: EmbeddingConfig
    reduction: ReductionConfig
    retrieval: RetrievalConfig
    perspective_groups: PerspectiveGroupsConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

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


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load and validate a YAML config file, returning a typed PipelineConfig."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    base_dir = config_path.parent.parent  # configs/ -> project root

    # Build each section
    topic = _build_dataclass(TopicConfig, raw.get("topic"))
    llm = _build_dataclass(LLMConfig, raw.get("llm"))
    corpus = _build_dataclass(CorpusConfig, raw.get("corpus"))
    anchors = _build_dataclass(AnchorConfig, raw.get("anchors"))
    embedding = _build_dataclass(EmbeddingConfig, raw.get("embedding"))
    reduction = _build_dataclass(ReductionConfig, raw.get("reduction"))
    retrieval = _build_dataclass(RetrievalConfig, raw.get("retrieval"))
    perspective_groups = _build_dataclass(
        PerspectiveGroupsConfig, raw.get("perspective_groups")
    )
    generation = _build_dataclass(GenerationConfig, raw.get("generation"))
    evaluation = _build_dataclass(EvaluationConfig, raw.get("evaluation"))

    config = PipelineConfig(
        topic=topic,
        llm=llm,
        corpus=corpus,
        anchors=anchors,
        embedding=embedding,
        reduction=reduction,
        retrieval=retrieval,
        perspective_groups=perspective_groups,
        generation=generation,
        evaluation=evaluation,
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

    if config.perspective_groups.enabled and config.perspective_groups.groups_file:
        p = config.resolve_path(config.perspective_groups.groups_file)
        if not p.exists():
            errors.append(f"perspective_groups.groups_file not found: {p}")

    if config.reduction.method not in ("umap", "pca", "none"):
        errors.append(f"reduction.method must be 'umap', 'pca', or 'none', got '{config.reduction.method}'")

    valid_retrieval = ("anchor_cosine", "bm25", "random", "dense_no_anchor")
    if config.retrieval.method not in valid_retrieval:
        errors.append(f"retrieval.method '{config.retrieval.method}' not in {valid_retrieval}")

    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))
