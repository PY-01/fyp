from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

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

# Function to assign clusters
def assign_clusters(x, y, cx, cy, k):
    clusters = [[] for _ in range(k)]
    cluster_indices = []
    for xi, yi in zip(x, y):
        distances = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append((xi, yi))
        cluster_indices.append(cluster_index)
    return clusters, cluster_indices

def animate_iterations(x, y, cx, cy, k, frames_per_move=5, max_iterations=100, tolerance=1e-4):
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

    # Add initial frame (iteration 0)
    clusters, cluster_indices = assign_clusters(x, y, cx, cy, k)
    frames.append((cx.copy(), cy.copy(), cluster_indices.copy(), 0, None, None, False))

    iteration = 0
    while iteration < max_iterations:
        clusters, cluster_indices = assign_clusters(x, y, cx, cy, k)
        new_cx, new_cy = update_centroids(clusters)

        # Handle empty clusters by re-initializing
        for i in range(k):
            if np.isnan(new_cx[i]) or np.isnan(new_cy[i]):
                new_cx[i] = np.random.uniform(min_x, max_x)
                new_cy[i] = np.random.uniform(min_y, max_y)

        centroid_shift = np.sqrt((new_cx - cx) ** 2 + (new_cy - cy) ** 2)
        if np.all(centroid_shift < tolerance):
            break

        # Add a frame for showing new red centroids
        frames.append((cx.copy(), cy.copy(), cluster_indices.copy(), iteration + 1, new_cx.copy(), new_cy.copy(), True))

        # Add interpolated frames for centroid movement
        for t in np.linspace(0, 1, frames_per_move):
            interpolated_cx = cx + t * (new_cx - cx)
            interpolated_cy = cy + t * (new_cy - cy)
            frames.append((interpolated_cx, interpolated_cy, cluster_indices.copy(), iteration + 1, None, None, False))

        cx, cy = new_cx, new_cy
        iteration += 1

    def update(frame_index):
        current_cx, current_cy, current_indices, current_iteration, target_cx, target_cy, show_new_centroids = frames[frame_index]

        ax.clear()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_title(f'K-means Iteration - Iteration {current_iteration}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Draw data points
        for i, (xi, yi) in enumerate(zip(x, y)):
            cluster_index = current_indices[i]
            color = colors[cluster_index % len(colors)]
            ax.scatter(xi, yi, c=color, marker=markers[cluster_index % len(markers)], s=30, alpha=0.5, zorder=4)

        # Draw centroids at current positions with cluster colors
        for i, (cxi, cyi) in enumerate(zip(current_cx, current_cy)):
            color = colors[i % len(colors)]
            ax.scatter(cxi, cyi, marker="*", s=100, c=color, zorder=5)

        # If new positions are available, draw them in red and connect with lines
        if show_new_centroids and target_cx is not None and target_cy is not None:
            for old_cx, old_cy, new_cx, new_cy in zip(current_cx, current_cy, target_cx, target_cy):
                ax.scatter(new_cx, new_cy, marker="*", s=100, c='red', zorder=6)  # Draw new centroid in red
                ax.plot([old_cx, new_cx], [old_cy, new_cy], c='red', linestyle='--', zorder=3)  # Connect with a line

        ax.grid(True, linestyle='--', zorder=1)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, repeat=False)
    html_str = ani.to_jshtml()
    plt.close(fig)
    return html_str
