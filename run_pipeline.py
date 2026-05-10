#!/usr/bin/env python3
"""Run the AutoConcourse pipeline.

Usage:
    python run_pipeline.py --config data/example/monolingual/config.yaml
    python run_pipeline.py --config data/example/multilingual/config.yaml --steps corpus,anchors,embed,retrieve
    python run_pipeline.py --config <your_topic>/config.yaml --force

Equivalent invocations:
    python -m autoconcourse.cli --config <...>
    autoconcourse --config <...>          # if installed via `pip install -e .`
"""

from autoconcourse.cli import pipeline_main

if __name__ == "__main__":
    pipeline_main()
