# K-Means Clustering from Scratch

A custom implementation of the K-Means clustering algorithm built from scratch in Python, without using scikit-learn's built-in KMeans. Developed as part of the **COME 401/1 Data Mining** course.

## Overview

This project implements the K-Means algorithm itself with NumPy only (scikit-learn is used just to generate the sample data and to compute silhouette scores, and Matplotlib draws the plots), focusing on understanding the inner workings of unsupervised learning. It includes:

- **From-scratch K-Means** with Euclidean distance calculation
- **Elbow Method** to determine the optimal number of clusters
- **Silhouette Score** evaluation for clustering quality
- **Step-by-step visualization** of the clustering process at each iteration
- **Convergence detection** — stops automatically when centroids stabilize, or after `max_iterations` (default 100)

## How It Works

1. Randomly initialize `k` centroids from the dataset
2. Assign each data point to the nearest centroid using Euclidean distance
3. Recalculate centroids as the mean of assigned points
4. Repeat until centroids no longer change (convergence) or `max_iterations` is reached. A cluster that receives no points keeps its previous centroid. If the iteration limit is reached first, the clusters and inertia are computed for the final centroids, so they agree with `predict()`.

## Features

- **Elbow Method**: Tests k values from 2 to 8 and plots inertia (sum of squared distances) to help identify the optimal number of clusters
- **Silhouette Score**: Provides a quantitative measure of clustering quality for each k value
- **Dynamic Visualization**: Uses colormaps to support any number of clusters, not limited to a fixed set
- **Reproducibility**: Fixed random seed (42) ensures consistent results across runs

## Dataset

Uses synthetic data generated with `sklearn.datasets.make_blobs` — 300 points distributed across 4 cluster centers (`random_state=42`).

## Results

Output of the current script (Python 3.11.9, numpy 2.4.6, scikit-learn 1.9.1, matplotlib 3.11.2):

| k | Inertia | Silhouette |
|---|---------|------------|
| 2 | 9416.21 | 0.5902 |
| 3 | 2110.41 | 0.7569 |
| 4 | 2058.54 | 0.6143 |
| 5 | 2007.60 | 0.4798 |
| 6 | 461.71 | 0.5321 |
| 7 | 434.73 | 0.5366 |
| 8 | 434.73 | 0.5366 |

The silhouette score picks **k=3**, not the 4 centers used to generate the data. The reason is the initialization: each run uses a single random initialization (no k-means++ and no restarts), and with seed 42 the k=4 run stops in a poor local minimum (inertia 2058.54; scikit-learn's `KMeans(n_clusters=4, n_init=10)` reaches about 564.9 on the same data, used here only as a reference). The k=8 run ends with one empty cluster, so its numbers equal those of k=7. These are limitations of the exercise, not changes made to the algorithm.

## How to Run

```
pip install numpy matplotlib scikit-learn
python "K-Means Clustering.py"
```

The quotes are needed because the filename contains spaces. The script opens plot windows, including one per iteration of the final fit. On a machine without a display, set `MPLBACKEND=Agg` first. The run is repeatable because the seed is fixed. Importing the file does not run the exercise; it only runs when the file is executed.

Only matplotlib 3.11.2 was tested. An earlier version of the script called `plt.cm.get_cmap`, which matplotlib removed in 3.9 and which made the step-by-step plots crash; the script now uses `plt.get_cmap`.

To run the checks (no downloads):
```
pip install pytest
python -m pytest tests -q
```

## Limitations

- `KMeans(k, max_iterations=100)` needs a positive integer `k` that is at most the number of samples, finite numeric 2-D data, and `predict()` data with the same number of features as the training data. Other input raises a `ValueError`; using `predict()` or `get_inertia()` before `fit()` raises a `RuntimeError`.
- The silhouette score from scikit-learn needs between 2 and `n_samples - 1` distinct cluster labels. The fixed data and the k range 2 to 8 in the script satisfy this; other inputs may not.
- The plots show only the first two features.
- `fit()` seeds NumPy's global random generator.

## Project Structure
```
kmeans-clustering-python/
    K-Means Clustering.py   # main script
    tests/
        test_kmeans.py      # correctness checks (no downloads)
    LICENSE
    README.md
```

## Technologies
- Python 3
- NumPy (numerical computation)
- Matplotlib (visualization)
- scikit-learn (dataset generation, silhouette score)

## Key Takeaways
- Random centroid initialization can significantly affect results — fixed seeds ensure reproducibility
- The Elbow Method provides visual guidance but isn't always clear-cut; Silhouette Score offers a numerical alternative
- K-Means performs well on spherical, similarly-sized clusters but struggles with irregular shapes

## License
[MIT](LICENSE)
