import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import random
random_seed=42


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.clusters = None
        self.random_seed = random_seed

    def fit(self, data, plot_progress=False):
        n_samples, n_features = data.shape
        if self.random_seed:
           np.random.seed(self.random_seed)
        self.centroids = data[np.random.choice(n_samples, self.k, replace=False)]

        for iteration in range(self.max_iterations):

            self.clusters = [[] for _ in range(self.k)]
            for point in data:
                distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
                closest_centroid = np.argmin(distances)
                self.clusters[closest_centroid].append(point)

            self.clusters = [np.array(cluster) for cluster in self.clusters]


            print(f"Iteration {iteration} distances:")
            for i, cluster in enumerate(self.clusters):
                if len(cluster) > 0:
                    distances = [euclidean_distance(point, self.centroids[i]) for point in cluster]
                    print(f"  Cluster {i + 1}: Average Distance = {np.mean(distances):.2f}, Max Distance = {np.max(distances):.2f}, Min Distance = {np.min(distances):.2f}")
                else:
                    print(f"  Cluster {i + 1}: No points assigned")


            if plot_progress:
                self.plot_iteration(data, iteration)


            new_centroids = np.array([
                cluster.mean(axis=0) if len(cluster) > 0 else self.centroids[i]
                for i, cluster in enumerate(self.clusters)
            ])




            if np.all(self.centroids == new_centroids):
                print(f"Convergence reached at iteration {iteration}.")
                self.centroids = new_centroids
                if plot_progress:
                    self.plot_iteration(data, iteration, final=True)
                break

            self.centroids = new_centroids

    def plot_iteration(self, data, iteration, final=False):
        # use a colormap so it works for any number of clusters
        cmap = plt.cm.get_cmap('tab10', self.k)
        plt.figure(figsize=(8, 6))

        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                plt.scatter(cluster[:, 0], cluster[:, 1], color=cmap(i), label=f"Cluster {i + 1}")

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color="black", marker="x", s=200, label="Centroids")

        plt.title(f"K-Means Iteration {iteration}{' (Final)' if final else ''}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    def predict(self, data):
        predicted_clusters = []
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid = np.argmin(distances)
            predicted_clusters.append(closest_centroid)
        return predicted_clusters

    def get_centroids(self):
        return self.centroids

    def get_clusters(self):
        return self.clusters

    def get_inertia(self):
        # inertia = sum of squared distances of each point to its centroid
        total = 0
        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                for point in cluster:
                    total += euclidean_distance(point, self.centroids[i]) ** 2
        return total


# generate synthetic dataset
data, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)


# plot the initial dataset
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], color="gray", label="Data Points", alpha=0.5)
plt.title("Initial Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# ============================================
# Elbow Method - find the best k
# ============================================
print("=" * 40)
print("Elbow Method - Testing k from 2 to 8")
print("=" * 40)

inertia_values = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(k=k)
    km.fit(data, plot_progress=False)
    inertia = km.get_inertia()
    inertia_values.append(inertia)
    print(f"k={k}, Inertia={inertia:.2f}")

# plot the elbow
plt.figure(figsize=(8, 6))
plt.plot(list(k_range), inertia_values, 'bo-')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.xticks(list(k_range))
plt.show()
print()


# ============================================
# Silhouette Score - another way to evaluate
# ============================================
print("=" * 40)
print("Silhouette Scores for different k values")
print("=" * 40)

silhouette_values = []

for k in k_range:
    km = KMeans(k=k)
    km.fit(data, plot_progress=False)
    labels = km.predict(data)
    score = silhouette_score(data, labels)
    silhouette_values.append(score)
    print(f"k={k}, Silhouette Score={score:.4f}")

# plot silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(list(k_range), silhouette_values, 'ro-')
plt.title("Silhouette Score for Different k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_range))
plt.show()

best_k = list(k_range)[np.argmax(silhouette_values)]
print(f"\nBest k based on silhouette score: {best_k}")
print()


# ============================================
# Run K-Means with the best k (k=4 in this case)
# ============================================
print("=" * 40)
print(f"Running K-Means with k={best_k}")
print("=" * 40)

kmeans = KMeans(k=best_k)
kmeans.fit(data, plot_progress=True)

centroids = kmeans.get_centroids()
clusters = kmeans.get_clusters()


# plot the final result
def plot_final_clusters(data, centroids, clusters, k):
    cmap = plt.cm.get_cmap('tab10', k)
    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], color=cmap(i), label=f"Cluster {i + 1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=200, label="Centroids")
    plt.title("Final Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

plot_final_clusters(data, centroids, clusters, best_k)

# print final evaluation
final_labels = kmeans.predict(data)
final_silhouette = silhouette_score(data, final_labels)
final_inertia = kmeans.get_inertia()
print(f"\nFinal Results:")
print(f"  Inertia: {final_inertia:.2f}")
print(f"  Silhouette Score: {final_silhouette:.4f}")
print(f"  Number of iterations until convergence: check output above")
