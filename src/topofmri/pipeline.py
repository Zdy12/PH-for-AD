import json
from pathlib import Path

import numpy as np
import pandas as pd

from .connectivity import correlation_from_timeseries, correlation_to_distance, sanitize_correlation, upper_triangle
from .io import discover_subjects, load_array
from .topology import TopologyConfig, compute_diagrams, quantify_diagrams


def extract_dataset(ad_dir, cn_dir, output_dir, input_kind="timeseries", roi_axis=1,
                    topology_config=None, save_diagrams=False):
    config = topology_config or TopologyConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_subjects(Path(ad_dir), Path(cn_dir))
    correlations, distances, edge_features = [], [], []
    n_rois = None
    for record in records:
        array = load_array(record["path"])
        correlation = (correlation_from_timeseries(array, roi_axis=roi_axis)
                       if input_kind == "timeseries" else sanitize_correlation(array))
        if n_rois is None:
            n_rois = correlation.shape[0]
        if correlation.shape != (n_rois, n_rois):
            raise ValueError(f"{record['path']}: inconsistent ROI count")
        correlations.append(correlation)
        distances.append(correlation_to_distance(correlation))
        edge_features.append(upper_triangle(correlation))
    distances = np.stack(distances)
    diagrams = compute_diagrams(distances, config)
    blocks = {"connectivity_edges": np.stack(edge_features)}
    blocks.update(quantify_diagrams(diagrams, config))
    feature_names = list(blocks)
    features = np.concatenate([blocks[name] for name in feature_names], axis=1)
    labels = np.asarray([r["label"] for r in records], dtype=int)
    subjects = pd.DataFrame([{k: str(v) if k == "path" else v for k, v in r.items()} for r in records])
    subjects.to_csv(output_dir / "subjects.csv", index=False)
    np.savez_compressed(output_dir / "features.npz", X=features, y=labels,
                        subject_id=subjects.subject_id.to_numpy())
    if save_diagrams:
        np.save(output_dir / "persistence_diagrams.npy", diagrams)
    manifest = {
        "n_subjects": len(records), "n_rois": n_rois, "total_features": features.shape[1],
        "feature_blocks": {name: int(blocks[name].shape[1]) for name in feature_names},
        "input_kind": input_kind,
        "topology": {
            "homology_dimensions": list(config.homology_dimensions), "n_bins": config.n_bins,
            "landscape_layers": list(config.landscape_layers), "heat_n_bins": config.heat_n_bins,
            "heat_sigmas": list(config.heat_sigmas), "infinity_values": config.infinity_values,
        },
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return features, labels, manifest

