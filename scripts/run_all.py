"""One-command pipeline runner: synthetic/GEO data -> 5 feature groups -> 5-fold CV & training."""

from __future__ import annotations

from scripts.build_features import main as build_features
from scripts.make_synthetic_data import main as make_synthetic_data
from scripts.train_all import main as train_all


def main() -> None:
    make_synthetic_data()
    build_features()
    train_all()


if __name__ == "__main__":
    main()
