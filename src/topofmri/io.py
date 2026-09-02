from pathlib import Path

import numpy as np


SUPPORTED_SUFFIXES = {".txt", ".csv", ".tsv", ".npy"}


def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        try:
            array = np.loadtxt(path)
        except ValueError:
            array = np.loadtxt(path, delimiter=",")
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2D array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: input contains NaN or infinity")
    return array


def discover_subjects(ad_dir: Path, cn_dir: Path):
    records = []
    for label, group, folder in [(1, "AD", ad_dir), (0, "CN", cn_dir)]:
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class directory: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if not files:
            raise ValueError(f"No supported files found in {folder}")
        records.extend({"path": p, "subject_id": p.stem, "group": group, "label": label} for p in files)
    ids = [r["subject_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Subject filenames must be unique across AD and CN folders")
    return records

