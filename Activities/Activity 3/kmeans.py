import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic dataset using make_blobs
data, true_labels = make_blobs(
    n_samples=300,        # Number of points
    centers=8,            # Number of clusters
    cluster_std=1.0,      # Standard deviation of the clusters
    random_state=42       # For reproducibility
)

# Function to visualize K-Means step-by-step
def slow_kmeans(data, k=5, max_iterations=10, sleep_time=1):
    # Step 1: Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    plt.figure(figsize=(8, 8))
    for iteration in range(max_iterations):
        # Step 2: Assign data points to nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)
        
        # Plot current clustering
        plt.clf()
        for i in range(k):
            plt.scatter(data[clusters == i, 0], data[clusters == i, 1], label=f"Cluster {i+1}")
            plt.scatter(centroids[i, 0], centroids[i, 1], s=200, c='black', marker='x')  # Centroids
            
        plt.title(f"K-Means Iteration {iteration+1}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.pause(sleep_time)  # Pause for slow-motion effect
        
        # Step 3: Update centroids
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence (if centroids don't change)
        if np.allclose(centroids, new_centroids, rtol=1e-6):
            print(f"Converged at iteration {iteration+1}")
            break
        
        centroids = new_centroids
    
    plt.show()

# Run the slow-motion K-Means demo
slow_kmeans(data, k=5, max_iterations=10, sleep_time=1)