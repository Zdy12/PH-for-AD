import numpy as np

from topofmri.connectivity import correlation_from_timeseries, correlation_to_distance, upper_triangle


def test_connectivity_shapes_and_invariants():
    rng = np.random.default_rng(0)
    correlation = correlation_from_timeseries(rng.normal(size=(30, 5)))
    distance = correlation_to_distance(correlation)
    assert correlation.shape == (5, 5)
    assert upper_triangle(correlation).shape == (10,)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.min(distance) >= 0.0 and np.max(distance) <= 2.0

