#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Create non-scientific synthetic BOLD data for a smoke test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects-per-class", type=int, default=6)
    parser.add_argument("--time-points", type=int, default=60)
    parser.add_argument("--rois", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    for group in ("AD", "CN"):
        folder = args.output_dir / group
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(args.subjects_per_class):
            latent = rng.normal(size=(args.time_points, 3))
            weights = rng.normal(size=(3, args.rois))
            signal = latent @ weights + rng.normal(scale=0.7 if group == "AD" else 0.5,
                                                    size=(args.time_points, args.rois))
            np.savetxt(folder / f"{group.lower()}_{idx:03d}.txt", signal)
    print(f"Synthetic smoke-test data written to {args.output_dir}")


if __name__ == "__main__":
    main()

