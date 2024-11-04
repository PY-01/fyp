import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def perform_kmeans(x, y, cx, cy, k, max_iterations=10):
    # Limit k to between 2 and 6
    if not (2 <= k <= 6):
        raise ValueError("The number of clusters 'k' must be between 2 and 6.")

    # Ensure cx and cy are NumPy arrays for element-wise operations
    cx = np.array(cx)
    cy = np.array(cy)
    
    fig, ax = plt.subplots()
    data = np.column_stack((x, y))
    
    # Define colors and markers for up to 6 clusters
    colors = ['blue', 'green', 'orange', 'purple', 'pink', 'cyan']
    markers = ['o', 's', '^', 'D', 'P', 'X']  # Circle, Square, Triangle, Diamond, Plus, and X
    
    # Calculate limits based on data points and centroids
    min_x = min(np.min(x), np.min(cx)) - 1
    max_x = max(np.max(x), np.max(cx)) + 1
    min_y = min(np.min(y), np.min(cy)) - 1
    max_y = max(np.max(y), np.max(cy)) + 1
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title('K-means Clustering Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Function to assign clusters
    def assign_clusters(x, y, cx, cy, k):
        clusters = [[] for _ in range(k)]
        for xi, yi in zip(x, y):
            distances = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append((xi, yi))
        return clusters

    # Function to update centroids
    def update_centroids(clusters):
        new_cx, new_cy = [], []
        for cluster in clusters:
            if cluster:
                new_cx.append(np.mean([point[0] for point in cluster]))
                new_cy.append(np.mean([point[1] for point in cluster]))
            else:
                new_cx.append(np.nan)  # Retain previous centroid if cluster is empty
                new_cy.append(np.nan)
        return np.array(new_cx), np.array(new_cy)

    # Initial assignment
    clusters = assign_clusters(x, y, cx, cy, k)

    def update(frame):
        nonlocal cx, cy, clusters
    
        ax.clear()
        ax.grid(True)  # Enable the grid for each frame
        ax.set_xlim(min_x, max_x)  # Updated limits
        ax.set_ylim(min_y, max_y)  # Updated limits
    
        # Title and labels
        ax.set_title(f'Iteration {frame+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
        # For Iteration 1, only plot the initial centroids and data points
        if frame == 0:
            # Plot data points without assigning them to clusters
            ax.scatter(x, y, c='gray', s=30, alpha=0.5, label='Data Points')
            ax.scatter(cx, cy, marker="*", s=100, c='red', label='Initial Centroids')
        else:
            # Assign clusters and plot each cluster with a different marker and color
            clusters = assign_clusters(x, y, cx, cy, k)
            for i, cluster in enumerate(clusters):
                cluster_x = [point[0] for point in cluster]
                cluster_y = [point[1] for point in cluster]
                ax.scatter(cluster_x, cluster_y, c=colors[i % len(colors)], marker=markers[i % len(markers)], s=30, alpha=0.5, label=f'Cluster {i+1}')
    
            # Update centroids
            new_cx, new_cy = update_centroids(clusters)
            
            # Plot updated centroids
            ax.scatter(new_cx, new_cy, marker="*", s=100, c='red', label='Centroids')
            
            # Stop if centroids have not changed significantly
            if np.allclose(cx, new_cx, atol=1e-4) and np.allclose(cy, new_cy, atol=1e-4):
                ani.event_source.stop()
            else:
                # Only update centroids if they have changed
                cx, cy = new_cx, new_cy
    
        # Add legend to distinguish clusters and centroids
        ax.legend(loc='best')

    # Initialize animation
    ani = FuncAnimation(fig, update, frames=max_iterations, interval=1000, repeat=False)
    html_str = ani.to_jshtml()
    plt.close(fig)
    return html_str
