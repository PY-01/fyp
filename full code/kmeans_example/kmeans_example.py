import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Function to assign clusters based on current centroids
def assign_clusters(x, y, cx, cy, k):
    clusters = [[] for _ in range(k)]
    for xi, yi in zip(x, y):
        distances = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append((xi, yi))
    return clusters

# Function to update centroids based on assigned clusters
def update_centroids(clusters):
    new_cx, new_cy = [], []
    for cluster in clusters:
        if cluster:
            new_cx.append(np.mean([point[0] for point in cluster]))
            new_cy.append(np.mean([point[1] for point in cluster]))
        else:
            new_cx.append(np.nan)
            new_cy.append(np.nan)
    return np.array(new_cx), np.array(new_cy)

# Main function to perform K-means clustering and generate animation
def perform_kmeans(x, y, cx, cy, k, max_iterations=10):
    # Limit k to between 2 and 6
    if not (2 <= k <= 6):
        raise ValueError("The number of clusters 'k' must be between 2 and 6.")

    cx = np.array(cx)
    cy = np.array(cy)
    
    fig, ax = plt.subplots()
    data = np.column_stack((x, y))
    
    colors = ['blue', 'green', 'orange', 'purple', 'pink', 'cyan']
    markers = ['o', 's', '^', 'D', 'P', 'X']
    
    min_x = min(np.min(x), np.min(cx)) - 1
    max_x = max(np.max(x), np.max(cx)) + 1
    min_y = min(np.min(y), np.min(cy)) - 1
    max_y = max(np.max(y), np.max(cy)) + 1
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title('K-means Clustering Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Determine the actual convergence iteration
    converged = False
    iteration_count = 0
    initial_cx, initial_cy = cx.copy(), cy.copy()  # Save initial centroids
    while not converged and iteration_count < max_iterations:
        clusters = assign_clusters(x, y, cx, cy, k)
        new_cx, new_cy = update_centroids(clusters)
        
        # Check for convergence
        if np.allclose(cx, new_cx, atol=1e-4) and np.allclose(cy, new_cy, atol=1e-4):
            converged = True
        else:
            cx, cy = new_cx, new_cy
            iteration_count += 1

    # Set max_iterations to the convergence iteration
    max_iterations = iteration_count + 1  # Include the convergence iteration

    # Reset centroids to initial values for the animation
    cx, cy = initial_cx, initial_cy
    clusters = assign_clusters(x, y, cx, cy, k)

    # Update function for animation
    def update(frame):
        nonlocal cx, cy, clusters  # Use nonlocal instead of global
        
        ax.clear()
        ax.grid(True)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        if frame == 0:
            # Display initial centroids and data points only
            ax.set_title('Iteration 0 (Initial Centroids)')
            ax.scatter(x, y, c='gray', s=30, alpha=0.5, label='Data Points')
            ax.scatter(cx, cy, marker="*", s=100, c='red', label='Initial Centroids')
        else:
            ax.set_title(f'Iteration {frame}')
            clusters = assign_clusters(x, y, cx, cy, k)
            for i, cluster in enumerate(clusters):
                cluster_x = [point[0] for point in cluster]
                cluster_y = [point[1] for point in cluster]
                ax.scatter(cluster_x, cluster_y, c=colors[i % len(colors)], marker=markers[i % len(markers)], s=30, alpha=0.5, label=f'Cluster {i+1}')

            new_cx, new_cy = update_centroids(clusters)
            ax.scatter(new_cx, new_cy, marker="*", s=100, c='red', label='Centroids')
            cx, cy = new_cx, new_cy

        ax.legend(loc='best')

    ani = FuncAnimation(fig, update, frames=max_iterations + 1, interval=1000, repeat=False)  # Start from frame 0
    html_str = ani.to_jshtml()
    plt.close(fig)
    return html_str
