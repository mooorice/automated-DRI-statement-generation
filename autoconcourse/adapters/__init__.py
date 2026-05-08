"""Adapters: convert raw input formats into the canonical corpus shape.

Canonical corpus shape (pandas.DataFrame):

| Column      | Type   | Required | Notes                                  |
|-------------|--------|----------|----------------------------------------|
| id          | str/int| yes      | unique passage id                      |
| text        | str    | yes      | the passage                            |
| language    | str    | no       | ISO code (de, fr, it, en, ...)         |
| group       | str    | no       | stratification column (any user label) |
| source_id   | str    | no       | original document id, for tracing      |
| *           | any    | no       | extra metadata kept as-is              |

Adapters live as siblings in this package. Each exposes a `load(path, **opts)
-> pd.DataFrame` function that returns canonical shape.

To add a new format: write a new adapter module with a `load` function and
register it in ADAPTERS below.
"""

from pathlib import Path

import pandas as pd

from . import csv as csv_adapter
from . import jsonl as jsonl_adapter
from . import swissdox as swissdox_adapter
from . import text_dir as text_dir_adapter

ADAPTERS = {
    "csv": csv_adapter.load,
    "jsonl": jsonl_adapter.load,
    "swissdox": swissdox_adapter.load,
    "text_dir": text_dir_adapter.load,
}


def load_with_adapter(adapter_name: str, path: str | Path, **opts) -> pd.DataFrame:
    """Load a corpus using the named adapter, returning canonical-shape DataFrame."""
    if adapter_name not in ADAPTERS:
        raise ValueError(
            f"Unknown corpus adapter '{adapter_name}'. "
            f"Available: {sorted(ADAPTERS.keys())}"
        )
    return ADAPTERS[adapter_name](Path(path), **opts)
