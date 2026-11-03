import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import make_blobs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "K-Means Clustering.py")  # the filename contains spaces

spec = importlib.util.spec_from_file_location("kmeans_script", SCRIPT)
kmeans_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kmeans_script)
KMeans = kmeans_script.KMeans

DATA, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)


def fitted(k, data=DATA, **kwargs):
    km = KMeans(k=k, **kwargs)
    km.fit(data)
    return km


def direct_inertia(data, centroids):
    """Sum of squared distances of every point to its nearest final centroid."""
    d = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d.min(axis=1).sum()


def test_import_does_not_run_the_exercise():
    assert plt.get_fignums() == []


@pytest.mark.parametrize("cap", [1, 2, 3, 100])
def test_state_matches_final_centroids_even_at_iteration_cap(cap):
    km = fitted(4, max_iterations=cap)
    labels = np.array(km.predict(DATA))
    assert [len(c) for c in km.get_clusters()] == [int((labels == i).sum()) for i in range(4)]
    assert km.get_inertia() == pytest.approx(direct_inertia(DATA, km.get_centroids()))


def test_iteration_cap_is_really_reached_in_the_cap_case():
    # with a single iteration the centroids are still moving, so this case exercises the cap
    km1 = fitted(4, max_iterations=1)
    km100 = fitted(4, max_iterations=100)
    assert not np.array_equal(km1.get_centroids(), km100.get_centroids())


def test_converged_result_is_unchanged():
    # value printed by the original script for k=4 on the committed synthetic data
    assert fitted(4).get_inertia() == pytest.approx(2058.54, abs=0.01)


def test_same_seed_gives_same_result():
    a, b = fitted(4), fitted(4)
    assert np.array_equal(a.get_centroids(), b.get_centroids())
    assert a.get_inertia() == b.get_inertia()


def test_empty_cluster_keeps_a_finite_centroid():
    data = np.array([[1.0, 1.0]] * 6 + [[9.0, 9.0]] * 2)
    km = fitted(3, data)
    assert np.isfinite(km.get_centroids()).all()
    assert len(km.get_clusters()) == 3
    assert sum(len(c) for c in km.get_clusters()) == len(data)
    assert any(len(c) == 0 for c in km.get_clusters())


@pytest.mark.parametrize("k", [0, -1, 2.5, "3", None, True])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        KMeans(k=k)


@pytest.mark.parametrize("cap", [0, -5, 1.5])
def test_invalid_max_iterations_is_rejected(cap):
    with pytest.raises(ValueError, match="max_iterations"):
        KMeans(k=2, max_iterations=cap)


def test_k_larger_than_samples_is_rejected():
    with pytest.raises(ValueError, match="larger than the number of samples"):
        KMeans(k=5).fit(DATA[:3])


@pytest.mark.parametrize("bad", [
    np.empty((0, 2)),
    np.arange(10.0),
    np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 5.0], [6.0, 6.0]]),
    np.array([[0.0, 0.0], [1.0, np.inf], [2.0, 5.0], [6.0, 6.0]]),
])
def test_invalid_training_data_is_rejected(bad):
    with pytest.raises(ValueError):
        KMeans(k=2).fit(bad)


def test_plain_lists_are_accepted():
    km = fitted(2, [[0, 0], [1, 1], [5, 5], [6, 6]])
    assert sorted(km.predict([[0, 0], [6, 6]])) == [0, 1]


def test_use_before_fit_raises_a_clear_error():
    km = KMeans(k=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        km.predict(DATA)
    with pytest.raises(RuntimeError, match="not fitted"):
        km.get_inertia()


@pytest.mark.parametrize("shape", [(3, 1), (3, 3)])
def test_predict_rejects_wrong_feature_count(shape):
    km = fitted(4)
    with pytest.raises(ValueError, match="features"):
        km.predict(np.zeros(shape))


def test_predict_rejects_a_single_1d_point():
    with pytest.raises(ValueError):
        fitted(4).predict(DATA[0])


def test_plot_progress_runs_with_current_matplotlib():
    # plt.cm.get_cmap was removed in matplotlib 3.9, which made plot_progress=True crash
    data = DATA[:60]
    km = KMeans(k=3)
    km.fit(data, plot_progress=True)
    kmeans_script.plot_final_clusters(data, km.get_centroids(), km.get_clusters(), 3)
    plt.close("all")
