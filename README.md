# K-Means Clustering from Scratch

A custom implementation of the K-Means clustering algorithm built from scratch in Python, without using scikit-learn's built-in KMeans. Developed as part of the **COME 401/1 Data Mining** course.

## Overview

This project implements the K-Means algorithm using only NumPy and Matplotlib, focusing on understanding the inner workings of unsupervised learning. It includes:

- **From-scratch K-Means** with Euclidean distance calculation
- **Elbow Method** to determine the optimal number of clusters
- **Silhouette Score** evaluation for clustering quality
- **Step-by-step visualization** of the clustering process at each iteration
- **Convergence detection** — stops automatically when centroids stabilize

## How It Works

1. Randomly initialize `k` centroids from the dataset
2. Assign each data point to the nearest centroid using Euclidean distance
3. Recalculate centroids as the mean of assigned points
4. Repeat until centroids no longer change (convergence)

## Features

- **Elbow Method**: Tests k values from 2 to 8 and plots inertia (sum of squared distances) to help identify the optimal number of clusters
- **Silhouette Score**: Provides a quantitative measure of clustering quality for each k value
- **Dynamic Visualization**: Uses colormaps to support any number of clusters, not limited to a fixed set
- **Reproducibility**: Fixed random seed (42) ensures consistent results across runs

## Dataset

Uses synthetic data generated with `sklearn.datasets.make_blobs` — 300 points distributed across 4 cluster centers. This provides a controlled environment to validate the algorithm's correctness.

## How to Run

```
pip install numpy matplotlib scikit-learn
python "K-Means Clustering.py"
```

## Project Structure
```
kmeans-clustering-python/
    K-Means Clustering.py   # main script
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
