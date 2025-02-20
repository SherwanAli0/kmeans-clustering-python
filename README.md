# K-Means Clustering in Python

This project implements the K-Means clustering algorithm from scratch in Python. It provides visualization for each step of the clustering process and allows users to inspect specific iterations. A fixed random seed is used to ensure consistent results, making this implementation ideal for learning, experimentation, and demonstration purposes.

## Table of Contents
- Features
- Dataset
- Challenges and Solutions
- Example Usage
- Installation and Usage
- Motivation
- Acknowledgments
- License

## Features
### Custom Implementation
- K-Means clustering is implemented without relying on high-level libraries like scikit-learn.

### Step-by-Step Visualization
- Displays the clustering process at each iteration, showing the movement of centroids and data point assignments.

### Specific Iteration Inspection
- Enables users to view the clustering state at any given iteration for detailed analysis.

### Reproducibility
- A fixed random seed (`random_seed=42`) ensures consistency across multiple runs.

### Convergence Detection
- Automatically identifies when centroids stabilize, marking the end of the clustering process.

## Dataset
The dataset is generated using the `make_blobs` function from scikit-learn, which creates synthetic data with predefined clusters. This setup ensures a clear and interpretable demonstration of the K-Means algorithm.

## Challenges and Solutions
### 1. Handling Inconsistent Outputs
- **Issue:** The algorithm initially produced different results each time due to random centroid initialization.
- **Solution:** Implemented a fixed random seed to guarantee consistent outputs.

### 2. Specific Iteration Visualization
- **Issue:** There was no way to view clustering progress at a chosen iteration.
- **Solution:** Introduced a `plot_specific_iteration` function to display the clustering state at any specified step.

## Example Usage
### Initial Dataset
The following plot shows the initial dataset with randomly distributed data points across multiple clusters:

![image](https://github.com/user-attachments/assets/1bb8b7ef-5620-430b-9a37-7e6245b9e823)

### Iteration Progress
The algorithm visualizes clustering evolution at each iteration, showing centroid adjustments and cluster assignments:

![image](https://github.com/user-attachments/assets/3a2220a8-8c65-4249-a1f6-0434066c0253)


**Note:** Iterations start from index `0`, where:
- **Iteration 0** represents the initial centroid assignment.
- **Subsequent iterations** depict centroid movements and cluster reassignments.

### Final Clustering Output
After convergence, the final clustering state is displayed:

![image](https://github.com/user-attachments/assets/8f20f091-f6c4-416e-87d4-c27c76426e71)


### Specific Iteration Display
Users can view clustering at a specific iteration by adjusting the `specific_iteration` variable in the script. For example, setting `specific_iteration = 3` will display the clustering at iteration 3:

![image](https://github.com/user-attachments/assets/ede2f600-3fc0-4e42-98a8-02390e4d288a)


## Installation and Usage
### Clone the Repository
```sh
git clone https://github.com/SherwanAli0/kmeans-clustering-python.git
cd kmeans-clustering-python
```

### Install Dependencies
Ensure Python 3.x is installed along with the required libraries:
```sh
pip install numpy matplotlib scikit-learn
```

### Run the Script
Execute the following command to run the K-Means clustering implementation:
```sh
python kmeans_clustering.py
```

## Motivation
K-Means is widely used in data science and machine learning for clustering analysis. This project was developed to:
- Provide a deeper understanding of how K-Means operates through a custom implementation.
- Offer a visual representation of the clustering process step-by-step.
- Help learners and practitioners analyze clustering principles interactively.

## Acknowledgments
This project was inspired by the article ["Create Your Own K-Means Clustering Algorithm in Python"](https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670) from Towards Data Science. Enhancements were made to include features like reproducibility, handling of empty clusters, and iteration-based visualization.

## License
This project is licensed under the MIT License.

