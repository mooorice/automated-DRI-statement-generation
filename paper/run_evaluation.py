#!/usr/bin/env python3
"""Evaluate system outputs against reference concourse.

Usage:
    python run_evaluation.py --config projects/healthcare_costs/config.yaml
    python run_evaluation.py --config projects/healthcare_costs/config.yaml --systems anchor_guided,B1_zero_context
"""

from paper.cli import evaluation_main

if __name__ == "__main__":
    evaluation_main()
