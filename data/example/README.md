# AutoConcourse Data Format Reference

This directory contains minimal example files documenting the expected data formats for the AutoConcourse pipeline. Use these as templates when setting up a new topic.

## Files

### `example_config.yaml`
Main pipeline configuration. Copy to `configs/your_topic.yaml` and adjust paths and parameters.

### `example_keywords.json`
Keywords for corpus filtering, organized by language. The pipeline checks whether any keyword appears in a paragraph (case-insensitive) and discards paragraphs that match none.

**Structure:** `{ "language_code": ["keyword1", "keyword2", ...] }`

### `example_subcategories.json`
User-defined subtopic definitions. These are the thematic dimensions along which the pipeline organizes both retrieval and generation. Each subcategory needs:
- `id` (int): Unique identifier (0-indexed)
- `name` (str): Short English name
- `description` (str): What this subtopic covers
- `topic_focus` (str): Concise focus phrase used in the generation prompt
- `anchor_prompt` (str): Prompt used for LLM-based anchor generation (can be in any language)

### `example_anchors.json`
Pre-generated anchor texts (used when `anchors.method: from_file`). Each entry ties a text passage to a subcategory and language. These are embedded alongside corpus paragraphs to guide retrieval.

**Structure:** Array of `{ "category": int, "text": str, "language": str, "subcategory_name": str }`

### `example_perspective_groups.json`
Optional grouping of media outlets (or other sources) by perspective. When enabled, the pipeline retrieves and generates statements separately for each group, producing a more balanced concourse.

**Structure:** `{ "groups": { "group_id": { "label": str, "medium_codes": [str] } }, "metadata_column": str }`

### `example_reference_concourse.json`
A human-authored reference concourse for evaluation. The evaluation module compares generated statements against this reference to compute coverage, novelty, and other metrics.

**Structure:**
```json
{
  "topic": "topic_id",
  "source": "Description of who created the reference (e.g. expert panel, published Q-study)",
  "statements": [
    {
      "id": "ref_001",
      "text": "A declarative opinion statement...",
      "subcategory": "subcategory_name",
      "type": "consideration"
    }
  ]
}
```
The `type` field is either `"consideration"` (opinion/value statement) or `"policy_option"` (concrete policy proposal).

### `example_corpus.csv`
Minimal corpus file. Must contain at least a text column and an id column. Additional metadata columns (e.g., `medium_code`, `language`) are preserved through the pipeline.
