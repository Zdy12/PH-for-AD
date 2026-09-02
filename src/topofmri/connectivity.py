import numpy as np


def correlation_from_timeseries(array: np.ndarray, roi_axis: int = 1) -> np.ndarray:
    if roi_axis not in (0, 1):
        raise ValueError("roi_axis must be 0 or 1")
    timeseries = array if roi_axis == 1 else array.T
    if timeseries.shape[0] < 3 or timeseries.shape[1] < 2:
        raise ValueError(f"Invalid time-series shape: {timeseries.shape}")
    correlation = np.corrcoef(timeseries, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=-1.0)
    return sanitize_correlation(correlation)


def sanitize_correlation(correlation: np.ndarray) -> np.ndarray:
    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError(f"Correlation matrix must be square, got {correlation.shape}")
    correlation = np.clip(np.asarray(correlation, dtype=float), -1.0, 1.0)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    return correlation


def correlation_to_distance(correlation: np.ndarray) -> np.ndarray:
    distance = 1.0 - sanitize_correlation(correlation)
    distance = np.maximum((distance + distance.T) / 2.0, 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def upper_triangle(correlation: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(correlation.shape[0], k=1)
    return correlation[indices]

