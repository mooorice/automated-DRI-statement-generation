#!/usr/bin/env python3
"""Run the AutoConcourse pipeline.

Usage:
    python run_pipeline.py --config configs/healthcare_ch.yaml
    python run_pipeline.py --config configs/healthcare_ch.yaml --steps embed,reduce,retrieve,generate
    python run_pipeline.py --config configs/healthcare_ch.yaml --force
"""

from autoconcourse.cli import pipeline_main

if __name__ == "__main__":
    pipeline_main()
