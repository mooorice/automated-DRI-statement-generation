#!/usr/bin/env python3
"""Run baseline systems for comparison.

Usage:
    python run_baselines.py --config configs/healthcare_ch.yaml
    python run_baselines.py --config configs/healthcare_ch.yaml --baselines B1_zero_context,B3_bm25_retrieval
"""

from paper.cli import baselines_main

if __name__ == "__main__":
    baselines_main()
