from dataclasses import dataclass

import numpy as np
from gtda.diagrams import BettiCurve, HeatKernel, PersistenceEntropy, PersistenceLandscape
from gtda.homology import VietorisRipsPersistence


@dataclass(frozen=True)
class TopologyConfig:
    homology_dimensions: tuple = (0, 1, 2)
    n_bins: int = 100
    landscape_layers: tuple = (1, 2)
    heat_n_bins: int = 10
    heat_sigmas: tuple = (1.2, 1.4)
    infinity_values: float = 2.0
    n_jobs: int = 1


def compute_diagrams(distance_matrices: np.ndarray, config: TopologyConfig) -> np.ndarray:
    transformer = VietorisRipsPersistence(
        metric="precomputed",
        homology_dimensions=list(config.homology_dimensions),
        infinity_values=config.infinity_values,
        n_jobs=config.n_jobs,
    )
    return transformer.fit_transform(distance_matrices)


def _flatten(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values.reshape(values.shape[0], -1)


def quantify_diagrams(diagrams: np.ndarray, config: TopologyConfig):
    blocks = {}
    for layers in config.landscape_layers:
        key = f"landscape_{layers}layer"
        blocks[key] = _flatten(PersistenceLandscape(
            n_bins=config.n_bins, n_layers=layers, n_jobs=config.n_jobs
        ).fit_transform(diagrams))
    blocks["betti_curve"] = _flatten(BettiCurve(
        n_bins=config.n_bins, n_jobs=config.n_jobs
    ).fit_transform(diagrams))
    for sigma in config.heat_sigmas:
        key = f"heat_sigma_{sigma:g}"
        blocks[key] = _flatten(HeatKernel(
            n_bins=config.heat_n_bins, sigma=sigma, n_jobs=config.n_jobs
        ).fit_transform(diagrams))
    blocks["persistence_entropy"] = _flatten(PersistenceEntropy(
        nan_fill_value=0.0, n_jobs=config.n_jobs
    ).fit_transform(diagrams))
    return blocks

