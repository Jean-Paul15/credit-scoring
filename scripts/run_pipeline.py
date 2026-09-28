"""Génération des données synthétiques (étape DVC `generate`) et construction du notebook IFRS 9.

Les étapes de modélisation s'exécutent via `dvc repro` (voir dvc.yaml).
"""
from __future__ import annotations

import argparse

from credit_risk_graph.config import GenerationConfig
from credit_risk_graph.data.pipeline import build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Génère les tables brutes, nettoyées, enrichies et les découpages temporels.")
    parser.add_argument("--notebook", action="store_true", help="Construit et exécute le notebook IFRS 9.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate:
        build_datasets(GenerationConfig())
    if args.notebook:
        from build_ifrs9_notebook import main as build_notebook

        build_notebook()


if __name__ == "__main__":
    main()
