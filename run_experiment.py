#!/usr/bin/env python3
"""Run ablation experiments: systematically vary pipeline parameters.

Usage:
    python run_experiment.py --config configs/healthcare_ch.yaml --ablations all
    python run_experiment.py --config configs/healthcare_ch.yaml --ablations A1,A3,A8
    python run_experiment.py --config configs/healthcare_ch.yaml --ablations A5 --dry-run

Ablation IDs:
    A1: No dimensionality reduction (reduction.method: none)
    A2: PCA reduction (reduction.method: pca)
    A3: UMAP reduction (reduction.method: umap)
    A4: No perspective groups (perspective_groups.enabled: false)
    A5: Top-k variation (retrieval.top_k: 50, 100, 500, 1000)
    A6: Embedding model variation
    A7: Generator model variation
    A8: No anchors (anchors.enabled: false)
"""

from autoconcourse.cli import experiment_main

if __name__ == "__main__":
    experiment_main()
