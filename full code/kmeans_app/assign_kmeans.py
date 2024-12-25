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

def animate_assign_clusters(x, y, cx, cy, k):
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
    frames.append((cx.copy(), cy.copy(), cluster_indices, -1, [-1] * k, None))

    # Step 2: Assign all points to their nearest cluster based on the initial centroids
    clusters, cluster_indices = assign_clusters(x, y, cx, cy, k)

    # Add frames for visualizing each cluster one by one
    highlight_mask = [-1] * k  # Initially, no cluster is highlighted
    for cluster_index in range(k):
        highlight_mask[cluster_index] = cluster_index  # Highlight the current cluster
        temp_indices = cluster_indices.copy()
        frames.append((cx.copy(), cy.copy(), temp_indices, 0, highlight_mask.copy(), None))

    # Update function for animation
    def update(frame_index):
        current_cx, current_cy, current_indices, current_iteration, current_highlight, _ = frames[frame_index]

        ax.clear()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_title('Assign Clusters')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        for i, (xi, yi) in enumerate(zip(x, y)):
            cluster_index = current_indices[i]
            if cluster_index == -1:  # Unassigned points
                ax.scatter(xi, yi, c='gray', marker='o', s=30, alpha=0.5, zorder=4)
            else:
                color = colors[cluster_index % len(colors)]
                if current_highlight[cluster_index] == -1:
                    color = 'gray'  # De-emphasize other clusters
                ax.scatter(xi, yi, c=color, marker=markers[cluster_index % len(markers)], s=30, alpha=0.5, zorder=4)

        for centroid_index, (cxi, cyi) in enumerate(zip(current_cx, current_cy)):
            if np.isnan(cxi) or np.isnan(cyi):
                continue
            centroid_color = colors[centroid_index % len(colors)]
            ax.scatter(cxi, cyi, marker="*", s=100, c=centroid_color, zorder=5)

        ax.grid(True, linestyle='--', zorder=1)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, repeat=False)
    html_str = ani.to_jshtml()
    plt.close(fig)
    return html_str
