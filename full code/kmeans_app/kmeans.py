from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Function to assign clusters based on current centroids
def assign_clusters(x, y, cx, cy, k):
    clusters = [[] for _ in range(k)]
    cluster_indices = []
    for xi, yi in zip(x, y):
        distances = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append((xi, yi))
        cluster_indices.append(cluster_index)
    return clusters, cluster_indices

# Function to update centroids based on assigned clusters
def update_centroids(clusters):
    new_cx, new_cy = [], []
    for cluster in clusters:
        if cluster:
            new_cx.append(np.mean([point[0] for point in cluster]))
            new_cy.append(np.mean([point[1] for point in cluster]))
        else:
            new_cx.append(np.nan)  # Placeholder for empty clusters
            new_cy.append(np.nan)
    return np.array(new_cx), np.array(new_cy)

# Main function to perform K-means clustering and generate animation
def perform_kmeans(x, y, cx, cy, k, frames_per_move=5, max_iterations=100, tolerance=1e-4):
    if not (2 <= k <= 6):
        raise ValueError("The number of clusters 'k' must be between 2 and 6.")

    cx = np.array(cx)
    cy = np.array(cy)

    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'orange', 'purple', 'pink', 'cyan']
    markers = ['o', 's', '^', 'D', 'P', 'X']

    min_x = min(np.min(x), np.min(cx)) - 1
    max_x = max(np.max(x), np.max(cx)) + 1
    min_y = min(np.min(y), np.min(cy)) - 1
    max_y = max(np.max(y), np.max(cy)) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    frames = []

    # Step 1: Initially plot all points in gray
    cluster_indices = [-1] * len(x)  # Initialize as unassigned
    frames.append((cx.copy(), cy.copy(), cluster_indices, -1, [-1] * k))

    # Step 2: Assign all points to their nearest cluster based on the initial centroids
    clusters, cluster_indices = assign_clusters(x, y, cx, cy, k)

    # Add frames for visualizing each cluster one by one
    highlight_mask = [-1] * k  # Initially, no cluster is highlighted
    for cluster_index in range(k):
        highlight_mask[cluster_index] = cluster_index  # Highlight the current cluster
        temp_indices = cluster_indices.copy()
        frames.append((cx.copy(), cy.copy(), temp_indices, 0, highlight_mask.copy()))

    # Step 3: Proceed with the usual K-means iterations
    iteration = 0
    while iteration < max_iterations:
        new_cx, new_cy = update_centroids(clusters)

        # Handle empty clusters by re-initializing with random points
        for i in range(k):
            if np.isnan(new_cx[i]) or np.isnan(new_cy[i]):
                new_cx[i] = np.random.uniform(min_x, max_x)
                new_cy[i] = np.random.uniform(min_y, max_y)

        centroid_shift = np.sqrt((new_cx - cx) ** 2 + (new_cy - cy) ** 2)
        if np.all(centroid_shift < tolerance):
            break

        # Add interpolated frames to show centroid movement
        for t in np.linspace(0, 1, frames_per_move):
            interpolated_cx = cx + t * (new_cx - cx)
            interpolated_cy = cy + t * (new_cy - cy)
            frames.append((interpolated_cx, interpolated_cy, cluster_indices.copy(), iteration, highlight_mask.copy()))

        # Update centroids and reassign clusters in one step
        cx, cy = new_cx, new_cy
        clusters, cluster_indices = assign_clusters(x, y, cx, cy, k)
        frames.append((cx.copy(), cy.copy(), cluster_indices.copy(), iteration + 1, highlight_mask.copy()))
        iteration += 1

    # Update function for animation
    def update(frame_index):
        current_cx, current_cy, current_indices, current_iteration, current_highlight = frames[frame_index]

        # Clear previous plot and setup
        ax.clear()

        # Set limits and labels
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        title = f'K-means Clustering'
        if current_iteration >= 0:
            title += f' - Iteration {current_iteration}'
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Plot centroids first so they appear in front of the grid
        ax.scatter(current_cx, current_cy, marker="*", s=100, c='red', label='Centroids', zorder=5)

        # Plot data points with their respective cluster colors or gray
        for i, (xi, yi) in enumerate(zip(x, y)):
            cluster_index = current_indices[i]
            if cluster_index == -1:  # Unassigned points
                ax.scatter(xi, yi, c='gray', marker='o', s=30, alpha=0.5, zorder=4)
            else:
                color = colors[cluster_index % len(colors)]
                if current_highlight[cluster_index] == -1:
                    color = 'gray'  # De-emphasize other clusters
                ax.scatter(xi, yi, c=color, marker=markers[cluster_index % len(markers)], s=30, alpha=0.5, zorder=4)

        # Set the grid style and make the grid lines dashed (set lower zorder to place behind the points)
        ax.grid(True, linestyle='--', zorder=1)  # Set grid's zorder to be the lowest to place it behind everything else

        # Add legend after the points to ensure it's placed correctly
        ax.legend(loc='best')

    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, repeat=False)
    html_str = ani.to_jshtml()
    plt.close(fig)
    return html_str
