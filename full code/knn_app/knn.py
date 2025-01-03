import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Initialize animation
def init(dotted_lines, k_nearest_lines, distance_texts):
    for ax_dotted_lines, ax_k_nearest_lines, distance_text in zip(dotted_lines, k_nearest_lines, distance_texts):
        for line in ax_dotted_lines + ax_k_nearest_lines:
            line.set_data([], [])
        for text in distance_text:
            text.set_text('')
    return [line for sublist in dotted_lines + k_nearest_lines for line in sublist] + [text for sublist in distance_texts for text in sublist]

# Animation function
def animate(frame, axs, dataSet, k_values, new_x, new_y, dotted_lines, k_nearest_lines, distance_texts, new_point_scatter_list, classes):
    new_point = np.array([new_x, new_y])
    for subplot_index, ax in enumerate(axs):
        k = k_values[subplot_index]
        distances = np.sqrt(np.sum((dataSet - new_point) ** 2, axis=1))
        sorted_indices = np.argsort(distances)
        sorted_x = dataSet[sorted_indices, 0]
        sorted_y = dataSet[sorted_indices, 1]
        sorted_distances = distances[sorted_indices]

        # Reset lines and labels for frames < 3
        if frame < 3:
            for line in dotted_lines[subplot_index]:
                line.set_data([], [])
            for line in k_nearest_lines[subplot_index]:
                line.set_data([], [])
            for text in distance_texts[subplot_index]:
                text.set_text('')

        if frame == 0:
            continue
        elif frame == 1:
            for i, line in enumerate(dotted_lines[subplot_index]):
                line.set_data([sorted_x[i], new_x], [sorted_y[i], new_y])
                line.set_linestyle(':')
        elif frame == 2:
            for i, line in enumerate(k_nearest_lines[subplot_index]):
                if i < k:
                    line.set_data([sorted_x[i], new_x], [sorted_y[i], new_y])
                    line.set_linewidth(2)
                    line.set_color('grey')
            # Add coordinates with an offset
            for i in range(k):
                text_obj = ax.text(sorted_x[i] + 0.1, sorted_y[i] + 0.1, f"({sorted_x[i]:.2f}, {sorted_y[i]:.2f})", color='black', fontsize=10)
                distance_texts[subplot_index].append(text_obj)
        elif frame == 3:
            for text in distance_texts[subplot_index]:
                text.set_text('')  # Clear coordinate texts
            for i in range(k):
                line = k_nearest_lines[subplot_index][i]
                # Keep the lines active by not resetting their data
                mid_x = (sorted_x[i] + new_x) / 2
                mid_y = (sorted_y[i] + new_y) / 2
                distance_texts[subplot_index][i].set_text(f"{sorted_distances[i]:.2f}")
                distance_texts[subplot_index][i].set_position((mid_x, mid_y))
        elif frame == 4:
            nearest_labels = [classes[sorted_indices[j]] for j in range(k)]
            
            # Count occurrences of each class
            label_counts = {label: nearest_labels.count(label) for label in set(nearest_labels)}
            
            # Find the classes with the maximum count (possible ties)
            max_count = max(label_counts.values())
            tied_classes = [label for label, count in label_counts.items() if count == max_count]
            
            if len(tied_classes) > 1:
                # Resolve tie by choosing the class with the smallest total distance
                total_distances = {label: sum(sorted_distances[i] for i in range(k) if classes[sorted_indices[i]] == label) 
                                   for label in tied_classes}
                majority_label = min(total_distances, key=total_distances.get)
            else:
                # No tie, just select the class with the highest count
                majority_label = tied_classes[0]

            # Highlight the k-nearest points and their distances
            for i in range(k):
                line = k_nearest_lines[subplot_index][i]
                mid_x = (sorted_x[i] + new_x) / 2
                mid_y = (sorted_y[i] + new_y) / 2
                if classes[sorted_indices[i]] == majority_label:
                    distance_texts[subplot_index][i].set_color('red')
                else:
                    distance_texts[subplot_index][i].set_color('black')
                distance_texts[subplot_index][i].set_text(f"{sorted_distances[i]:.2f}")
                distance_texts[subplot_index][i].set_position((mid_x, mid_y))
        elif frame == 5:
            # Clear the distance text but keep the k-nearest lines
            for text in distance_texts[subplot_index]:
                text.set_text('')  # Clear distance text

            # Determine the majority label using the same logic as frame 4
            nearest_labels = [classes[sorted_indices[j]] for j in range(k)]

            # Count occurrences of each class
            label_counts = {label: nearest_labels.count(label) for label in set(nearest_labels)}

            # Find the classes with the maximum count (possible ties)
            max_count = max(label_counts.values())
            tied_classes = [label for label, count in label_counts.items() if count == max_count]

            if len(tied_classes) > 1:
                # Resolve tie by choosing the class with the smallest total distance
                total_distances = {label: sum(sorted_distances[i] for i in range(k) if classes[sorted_indices[i]] == label)
                                   for label in tied_classes}
                majority_label = min(total_distances, key=total_distances.get)
            else:
                # No tie, just select the class with the highest count
                majority_label = tied_classes[0]

            # Highlight points belonging to the majority label
            majority_points = dataSet[sorted_indices[:k]][np.array(nearest_labels) == majority_label]
            ax.scatter(majority_points[:, 0], majority_points[:, 1], edgecolors='black', facecolors='none', s=400, label='Majority')
        elif frame == 6:
            # Reset the k-nearest lines
            for line in k_nearest_lines[subplot_index]:
                line.set_data([], [])
            for text in distance_texts[subplot_index]:
                text.set_text('')  # Clear any distance text

            # Determine the majority label using the same logic as frame 4
            nearest_labels = [classes[sorted_indices[j]] for j in range(k)]

            # Count occurrences of each class
            label_counts = {label: nearest_labels.count(label) for label in set(nearest_labels)}

            # Find the classes with the maximum count (possible ties)
            max_count = max(label_counts.values())
            tied_classes = [label for label, count in label_counts.items() if count == max_count]

            if len(tied_classes) > 1:
                # Resolve tie by choosing the class with the smallest total distance
                total_distances = {label: sum(sorted_distances[i] for i in range(k) if classes[sorted_indices[i]] == label)
                                   for label in tied_classes}
                majority_label = min(total_distances, key=total_distances.get)
            else:
                # No tie, just select the class with the highest count
                majority_label = tied_classes[0]

            # Update new point color based on majority label
            new_point_color = 'purple' if majority_label == 0 else 'orange'
            new_point_scatter_list[subplot_index].set_color(new_point_color)

            # Add custom legend next to the test point
            ax.text(new_x + 0.5, new_y, f'Test Data belongs to: Class {majority_label}', color='black', fontsize=12, verticalalignment='center')

    return [line for sublist in dotted_lines + k_nearest_lines for line in sublist] + [text for sublist in distance_texts for text in sublist]

# Validate k values
def validate_k_values(k_values, n_points):
    max_k = int(np.sqrt(n_points))  # Maximum k is the square root of the number of points
    if any(k > max_k for k in k_values):
        raise ValueError(f"Invalid k value! Maximum allowed k for the current dataset is {max_k}.")
    if not all(k > 0 for k in k_values):
        raise ValueError("k values must be greater than 0.")
    return k_values

def perform_knn_animation_html(x, y, classes, new_x, new_y, k_values, show_grid=True):
    if len(k_values) > 4:
        raise ValueError("You can input up to 4 k values only.")

    x, y, classes = map(np.array, (x, y, classes))
    new_point = np.array([new_x, new_y])
    dataSet = np.column_stack((x, y))

    # Validate k values based on dataset size
    k_values = validate_k_values(k_values, len(dataSet))

    # Single row layout for up to 4 k-values
    fig, axs = plt.subplots(1, len(k_values), figsize=(8 * len(k_values), 8))
    axs = [axs] if len(k_values) == 1 else axs

    dotted_lines, k_nearest_lines, distance_texts, new_point_scatter_list = [], [], [], []

    for i, ax in enumerate(axs):
        ax.scatter(x[classes == 0], y[classes == 0], c='purple', label='Class 0', marker='D', s=100, zorder=3)
        ax.scatter(x[classes == 1], y[classes == 1], c='orange', label='Class 1', marker='^', s=100, zorder=3)
        new_point_scatter = ax.scatter(new_x, new_y, c='red', marker='*', s=150, label='New Point', zorder=3)
        new_point_scatter_list.append(new_point_scatter)
        
        # Ensure grid lines are behind other elements
        ax.set_axisbelow(True)
        ax.grid(show_grid, linestyle='--')
        
        dotted_lines.append([ax.plot([], [], linestyle=':', color='gray', zorder=2)[0] for _ in range(len(dataSet))])
        k_nearest_lines.append([ax.plot([], [], color='blue', zorder=2)[0] for _ in range(k_values[i])])
        distance_texts.append([ax.text(0, 0, '', color='black', fontsize=10) for _ in range(k_values[i])])
        ax.legend(loc='upper left', fontsize=10)
        ax.set_title(f'k = {k_values[i]}', fontsize=15)

    fig.tight_layout(pad=3.0)

    ani = FuncAnimation(fig, animate, fargs=(axs, dataSet, k_values, new_x, new_y, dotted_lines, k_nearest_lines, distance_texts, new_point_scatter_list, classes),
                        frames=7, init_func=lambda: init(dotted_lines, k_nearest_lines, distance_texts), blit=False, interval=1000)

    return ani.to_jshtml()

