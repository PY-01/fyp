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

# Function for K-means++ initialization
def kmeans_plus_plus_init_with_frames(x, y, k):
    frames = []
    centroids = [np.random.choice(range(len(x)))]  # Random index from the dataset
    cx, cy = [x[centroids[0]]], [y[centroids[0]]]
    
    # Step 1: Show the first randomly selected centroid
    cluster_indices = [-1] * len(x)
    frames.append((cx.copy(), cy.copy(), cluster_indices, 'Init'))
    
    for step in range(1, k):
        # Step 2: Compute the squared distances from the nearest centroid
        distances = np.array([
            min([np.sqrt((xi - cx_i)**2 + (yi - cy_i)**2) for cx_i, cy_i in zip(cx, cy)])**2
            for xi, yi in zip(x, y)
        ])
        
        # Step 3: Select a new centroid with probability proportional to distance squared
        probs = distances / np.sum(distances)
        new_centroid = np.random.choice(range(len(x)), p=probs)
        
        # Step 4: Add the new centroid
        centroids.append(new_centroid)
        cx.append(x[new_centroid])
        cy.append(y[new_centroid])
        
        # Add frame showing the new centroid added
        frames.append((cx.copy(), cy.copy(), cluster_indices, f'Init Step {step}'))
    
    return np.array(cx), np.array(cy), frames

# Main function to perform K-means clustering and generate animation
def perform_kmeans_plus_plus(x, y, k, frames_per_move=5, max_iterations=100, tolerance=1e-4):
    if not (2 <= k <= 6):
        raise ValueError("The number of clusters 'k' must be between 2 and 6.")

    # K-means++ initialization
    cx, cy, init_frames = kmeans_plus_plus_init_with_frames(x, y, k)

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

    frames = init_frames  # Start with initialization frames

    # Step 1: Proceed with the usual K-means iterations
    clusters, cluster_indices = assign_clusters(x, y, cx, cy, k)
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
            frames.append((interpolated_cx, interpolated_cy, cluster_indices.copy(), iteration))

        # Update centroids and reassign clusters
        cx, cy = new_cx, new_cy
        clusters, cluster_indices = assign_clusters(x, y, cx, cy, k)
        frames.append((cx.copy(), cy.copy(), cluster_indices.copy(), iteration + 1))
        iteration += 1

    # Update function for animation
    def update(frame_index):
        current_cx, current_cy, current_indices, step_description = frames[frame_index]
        
        # Clear previous plot and setup
        ax.clear()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        title = f'K-means Clustering'
        if isinstance(step_description, int):
            title += f' - Iteration {step_description}'
        else:
            title += f' - {step_description}'
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Plot centroids
        ax.scatter(current_cx, current_cy, marker="*", s=100, c='red', label='Centroids', zorder=5)
        
        # Plot data points
        for i, (xi, yi) in enumerate(zip(x, y)):
            if current_indices[i] == -1:
                ax.scatter(xi, yi, c='gray', marker='o', s=30, alpha=0.5, zorder=4)
            else:
                cluster_index = current_indices[i]
                ax.scatter(xi, yi, c=colors[cluster_index % len(colors)], marker=markers[cluster_index % len(markers)], s=30, alpha=0.5, zorder=4)
        
        ax.grid(True, linestyle='--', zorder=1)
        ax.legend(loc='best')

    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, repeat=False)
    html_str = ani.to_jshtml()
    plt.close(fig)
    return html_str
