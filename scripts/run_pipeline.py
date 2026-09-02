#!/usr/bin/env python
import argparse
from pathlib import Path

from topofmri.evaluation import evaluate
from topofmri.pipeline import extract_dataset
from topofmri.topology import TopologyConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and classify multiscale fMRI topological features")
    parser.add_argument("--ad-dir", required=True, type=Path)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-kind", choices=["timeseries", "correlation"], default="timeseries")
    parser.add_argument("--roi-axis", choices=[0, 1], type=int, default=1,
                        help="Axis containing ROIs for time-series input (default: columns)")
    parser.add_argument("--homology-dimensions", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--heat-n-bins", type=int, default=10)
    parser.add_argument("--heat-sigmas", nargs="+", type=float, default=[1.2, 1.4])
    parser.add_argument("--infinity-values", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--evaluation", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--svm-kernel", choices=["linear", "rbf"], default="linear")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale",
                        help="SVM gamma: 'scale', 'auto', or a positive float")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-diagrams", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TopologyConfig(homology_dimensions=tuple(args.homology_dimensions), n_bins=args.n_bins,
                            heat_n_bins=args.heat_n_bins, heat_sigmas=tuple(args.heat_sigmas),
                            infinity_values=args.infinity_values, n_jobs=args.n_jobs)
    X, y, manifest = extract_dataset(args.ad_dir, args.cn_dir, args.output_dir,
                                     input_kind=args.input_kind, roi_axis=args.roi_axis,
                                     topology_config=config, save_diagrams=args.save_diagrams)
    gamma = args.svm_gamma if args.svm_gamma in {"scale", "auto"} else float(args.svm_gamma)
    _, summary = evaluate(X, y, args.output_dir, evaluation=args.evaluation,
                          test_size=args.test_size, folds=args.folds, seed=args.seed,
                          kernel=args.svm_kernel, c=args.svm_c, gamma=gamma)
    print(f"Extracted {manifest['total_features']} features for {manifest['n_subjects']} subjects")
    print(summary.to_string())


if __name__ == "__main__":
    main()
