# knn.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def classify0(newPoint, dataSet, labels, k):
    newPoint = np.array(newPoint)
    testDataDiff = dataSet - newPoint
    testDataDist = np.sqrt((testDataDiff ** 2).sum(axis=1))
    sortedIndex = testDataDist.argsort()
    sortedLabels = [labels[i] for i in sortedIndex]
    K_nearest = sortedLabels[:k]
    return K_nearest, testDataDist[sortedIndex]

def init(dotted_lines, k_nearest_lines, distance_texts):
    for ax_dotted_lines, ax_k_nearest_lines, distance_text in zip(dotted_lines, k_nearest_lines, distance_texts):
        for line in ax_dotted_lines + ax_k_nearest_lines:
            line.set_data([], [])
        distance_text.set_text('')
    return [line for sublist in dotted_lines + k_nearest_lines for line in sublist] + distance_texts

def animate(frame, axs, dataSet, k_values, new_x, new_y, dotted_lines, k_nearest_lines, distance_texts, new_point_scatter_list, classes):
    total_frames = len(dataSet) + max(k_values) + 1
    new_point = np.array([new_x, new_y])

    for subplot_index, ax in enumerate(axs):
        k = k_values[subplot_index]
        distances = np.sqrt(np.sum((dataSet - new_point) ** 2, axis=1))
        sorted_indices = np.argsort(distances).astype(int)
        sorted_x = dataSet[sorted_indices, 0]
        sorted_y = dataSet[sorted_indices, 1]
        sorted_distances = distances[sorted_indices]

        for line in dotted_lines[subplot_index]:
            line.set_data([], [])
        for line in k_nearest_lines[subplot_index]:
            line.set_data([], [])

        frame_in_subplot = int(frame % total_frames)

        if frame_in_subplot < len(dataSet):
            for i, line in enumerate(dotted_lines[subplot_index]):
                if frame_in_subplot == i:
                    line.set_data([sorted_x[i], new_x], [sorted_y[i], new_y])
                    dist = sorted_distances[i]
                    mid_x = (sorted_x[i] + new_x) / 2
                    mid_y = (sorted_y[i] + new_y) / 2
                    distance_texts[subplot_index].set_text(f"{dist:.2f}")
                    distance_texts[subplot_index].set_position((mid_x, mid_y))
        else:
            for i, line in enumerate(k_nearest_lines[subplot_index]):
                if frame_in_subplot - len(dataSet) > i:
                    line.set_data([new_x, sorted_x[i]], [new_y, sorted_y[i]])

            if frame_in_subplot == len(dataSet) + k:
                nearest_labels = [classes[sorted_indices[j]] for j in range(k)]
                nearest_points = dataSet[sorted_indices[:k]]
                update_new_point(new_point_scatter_list[subplot_index], nearest_labels, nearest_points, ax)

    return [line for sublist in dotted_lines + k_nearest_lines for line in sublist] + distance_texts

def update_new_point(new_point_scatter, nearest_labels, nearest_points, ax):
    unique_labels = list(set(nearest_labels))
    counts = [nearest_labels.count(i) for i in unique_labels]
    majority_label = unique_labels[counts.index(max(counts))]
    majority_color = 'purple' if majority_label == 0 else 'orange'
    new_point_scatter.set_color(majority_color)

    majority_points = [(x, y) for (x, y), label in zip(nearest_points, nearest_labels) if label == majority_label]
    if majority_points:
        majority_x, majority_y = zip(*majority_points)
        ax.scatter(majority_x, majority_y, facecolors='none', edgecolors='black', s=300, label='Majority Points')
        ax.legend()

def perform_knn_animation_html(x, y, classes, new_x, new_y, k_values, show_grid=True):
    x = np.array(x)
    y = np.array(y)
    classes = np.array(classes)
    new_point = np.array([new_x, new_y])
    dataSet = np.column_stack((x, y))

    num_k = len(k_values)
    if num_k == 1:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        axs = [axs]
    elif num_k == 3:
        # Set up a 2x2 grid but only use the first 3 subplots for k values
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()
    elif num_k == 4:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, num_k, figsize=(10 * num_k, 7))

    dotted_lines = []
    k_nearest_lines = []
    distance_texts = []
    new_point_scatter_list = []

    # Define markers for each class
    class_markers = {0: 'D', 1: '^'}  # Class 0 = circle, Class 1 = square
    new_point_marker = '*'  # Star for the new point

    for i, ax in enumerate(axs):
        if i < num_k:
            # Only populate the first `num_k` subplots
            # Plot each class with its unique marker
            for class_label, marker in class_markers.items():
                ax.scatter(x[classes == class_label], y[classes == class_label], c='purple' if class_label == 0 else 'orange', 
                           label=f'Class {class_label}', s=50, marker=marker)

            # Plot the new point with a star marker
            new_point_scatter = ax.scatter(new_x, new_y, c="red", marker=new_point_marker, label='New Point', s=100)
            new_point_scatter_list.append(new_point_scatter)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(min(x) - 1, max(x) + 1)
            ax.set_ylim(min(y) - 1, max(y) + 1)
            ax.grid(linestyle='--', color='grey', alpha=0.7)
            ax.legend()

            dotted_lines.append([ax.plot([], [], linestyle=':', color='grey')[0] for _ in range(len(dataSet))])
            k_nearest_lines.append([ax.plot([], [], linestyle='-', linewidth=1.5, color='grey')[0] for _ in range(k_values[i])])
            distance_texts.append(ax.text(0, 0, '', color='black', fontsize=10))
        else:
            # Leave the last subplot blank
            ax.axis('off')

    total_frames = len(dataSet) + max(k_values) + 1
    ani = FuncAnimation(
        fig, animate, fargs=(axs[:num_k], dataSet, k_values, new_x, new_y, dotted_lines, k_nearest_lines, distance_texts, new_point_scatter_list, classes),
        frames=total_frames, init_func=lambda: init(dotted_lines, k_nearest_lines, distance_texts), blit=False, interval=1000
    )

    html_str = ani.to_jshtml()
    plt.close(fig)
    return html_str
