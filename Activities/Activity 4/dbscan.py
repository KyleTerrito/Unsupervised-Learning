import numpy as np
import matplotlib.pyplot as plt
import pacmap
import imageio
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from matplotlib.patches import Circle

# Generate synthetic dataset using make_blobs
def load_dataset(n_samples=1000, n_features=10, centers=5, random_state=42):
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    return data

# Apply PaCMAP for dimensionality reduction
def apply_pacmap(data):
    reducer = pacmap.PaCMAP(n_components=2)
    return reducer.fit_transform(data)

'''you can try running this one but it may be slow'''
# Generate animation with neighborhood radius visualization
# def generate_animation(embedding, min_iter, max_iter, step):
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     def update(eps):
#         ax.clear()
#         dbscan = DBSCAN(eps=eps, min_samples=5)
#         labels = dbscan.fit_predict(embedding)
#         scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
#         # Add neighborhood radius circles for core points
#         for i, label in enumerate(labels):
#             if label != -1:  # Ignore noise points
#                 circle = Circle((embedding[i, 0], embedding[i, 1]), eps, color='red', fill=False, alpha=0.3)
#                 ax.add_patch(circle)
        
#         ax.set_title(f'DBSCAN Clustering (eps={eps:.2f})')
#         ax.set_xlabel('PaCMAP Dimension 1')
#         ax.set_ylabel('PaCMAP Dimension 2')
#         return scatter,
    
#     ani = animation.FuncAnimation(fig, update, frames=np.arange(min_iter, max_iter + step, step), interval=500, repeat=True)
#     plt.show()

'''this version only plots a single fixed neighborhood radius'''
# Generate animation with a single fixed neighborhood radius visualization
def generate_animation(embedding, min_iter, max_iter, step):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Run DBSCAN once to pick a fixed core point (avoid noise)
    dbscan_init = DBSCAN(eps=min_iter, min_samples=5)
    labels_init = dbscan_init.fit_predict(embedding)
    
    core_indices = np.where(labels_init != -1)[0]
    if len(core_indices) > 0:
        fixed_index = core_indices[0]  # Choose the first core point
    else:
        fixed_index = np.random.randint(0, len(embedding))  # Fallback: pick any point if no core found
    
    fixed_point = embedding[fixed_index]

    def update(eps):
        ax.clear()
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(embedding)
        
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        # Draw a single circle around the fixed core point
        circle = Circle((fixed_point[0], fixed_point[1]), eps, color='red', fill=False, alpha=0.7)
        ax.add_patch(circle)

        ax.set_title(f'DBSCAN Clustering (eps={eps:.2f})')
        ax.set_xlabel('PaCMAP Dimension 1')
        ax.set_ylabel('PaCMAP Dimension 2')
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(min_iter, max_iter + step, step), interval=500, repeat=True)
    plt.show()

#############################################################################################
# Main execution
data = load_dataset()
data = StandardScaler().fit_transform(data)  # Normalize data
embedding = apply_pacmap(data)

# Define DBSCAN iteration range and interval
min_iter = 0.1  # Starting eps
max_iter = 1.0  # Ending eps
step = 0.1  # Step size

generate_animation(embedding, min_iter, max_iter, step)
#############################################################################################