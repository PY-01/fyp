import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

def classify0(newPoint, dataSet, labels, k):
    newPoint = np.array(newPoint)
    testDataDiff = dataSet - newPoint
    testDataDist = np.sqrt((testDataDiff ** 2).sum(axis=1))
    sortedIndex = testDataDist.argsort()
    sortedLabels = [labels[i] for i in sortedIndex]
    K_nearest = sortedLabels[:k]
    return K_nearest, testDataDist[sortedIndex]

def perform_knn_animation(x, y, classes, new_x, new_y, k_values, show_grid=True):
    # Ensure new_x and new_y are numpy arrays
    new_point = np.array([new_x, new_y])
    dataSet = np.column_stack((x, y))  # Create a 2D array for dataset

    num_k = len(k_values)
    fig, axs = plt.subplots(1, num_k, figsize=(10 * num_k, 7))  # Create subplots for each k

    if num_k == 1:
        axs = [axs]  # Ensure axs is iterable

    dotted_lines = []
    k_nearest_lines = []
    distance_texts = []
    
    # Scatter the data points and hold references
    scatter0_list = []
    scatter1_list = []
    new_point_scatter_list = []

    for i, ax in enumerate(axs):
        scatter0 = ax.scatter(x[classes == 0], y[classes == 0], c='blue', label='Class 0', s=50)
        scatter1 = ax.scatter(x[classes == 1], y[classes == 1], c='green', label='Class 1', s=50)
        new_point_scatter = ax.scatter(new_x, new_y, c="red", marker="o", label='New Point', s=100)

        scatter0_list.append(scatter0)
        scatter1_list.append(scatter1)
        new_point_scatter_list.append(new_point_scatter)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(min(x) - 1, max(x) + 1)
        ax.set_ylim(min(y) - 1, max(y) + 1)
        
        ax.grid(show_grid, linestyle='--', color='grey', alpha=0.7)
        ax.legend()

        # Prepare placeholders for animation
        dotted_lines.append([ax.plot([], [], linestyle=':', color='grey')[0] for _ in range(len(dataSet))])
        k_nearest_lines.append([ax.plot([], [], linestyle='-', linewidth=1.5, color='grey')[0] for _ in range(k_values[i])])
        distance_texts.append(ax.text(0, 0, '', color='black', fontsize=10))

    def update_new_point(new_point_scatter, nearest_labels, nearest_points, ax):
        # Determine the majority label among the nearest labels
        unique_labels = list(set(nearest_labels))
        counts = [nearest_labels.count(i) for i in unique_labels]
        majority_label = unique_labels[counts.index(max(counts))]
        majority_color = 'blue' if majority_label == 0 else 'green'
        new_point_scatter.set_color(majority_color)

        # Highlight the nearest points that belong to the majority class
        majority_points = [(x, y) for (x, y), label in zip(nearest_points, nearest_labels) if label == majority_label]
        if majority_points:
            majority_x, majority_y = zip(*majority_points)
            ax.scatter(majority_x, majority_y, facecolors='none', edgecolors='black', s=300, label='Majority Points')
            ax.legend()

    def init():
        # The data points will already be drawn, so only initialize the lines
        for ax_dotted_lines, ax_k_nearest_lines, distance_text in zip(dotted_lines, k_nearest_lines, distance_texts):
            for line in ax_dotted_lines + ax_k_nearest_lines:
                line.set_data([], [])
            distance_text.set_text('')

        return [line for sublist in dotted_lines + k_nearest_lines for line in sublist] + distance_texts

    def animate(frame):
        total_frames = len(dataSet) + max(k_values) + 1

        # Iterate over all subplots to animate them simultaneously
        for subplot_index, ax in enumerate(axs):
            k = k_values[subplot_index]

            distances = [np.sqrt(np.sum((dataSet[i] - new_point) ** 2)) for i in range(len(dataSet))]
            sorted_indices = np.argsort(distances)
            sorted_x = dataSet[sorted_indices][:, 0]
            sorted_y = dataSet[sorted_indices][:, 1]
            sorted_distances = np.array(distances)[sorted_indices]

            # Reset dotted lines and k-nearest lines
            for line in dotted_lines[subplot_index]:
                line.set_data([], [])
            for line in k_nearest_lines[subplot_index]:
                line.set_data([], [])

            frame_in_subplot = frame % total_frames

            if frame_in_subplot < len(dataSet):
                # Draw dotted lines
                for i, line in enumerate(dotted_lines[subplot_index]):
                    if frame_in_subplot == i:
                        line.set_data([sorted_x[i], new_x], [sorted_y[i], new_y])
                        dist = sorted_distances[i]
                        mid_x = (sorted_x[i] + new_x) / 2
                        mid_y = (sorted_y[i] + new_y) / 2
                        distance_texts[subplot_index].set_text(f"{dist:.2f}")
                        distance_texts[subplot_index].set_position((mid_x, mid_y))
            else:
                # Draw k-nearest neighbor lines
                for i, line in enumerate(k_nearest_lines[subplot_index]):
                    if frame_in_subplot - len(dataSet) > i:
                        line.set_data([new_x, sorted_x[i]], [new_y, sorted_y[i]])

                if frame_in_subplot == len(dataSet) + k:
                    nearest_labels = [classes[i] for i in sorted_indices[:k]]
                    nearest_points = dataSet[sorted_indices[:k]]
                    update_new_point(new_point_scatter_list[subplot_index], nearest_labels, nearest_points, ax)

        return [line for sublist in dotted_lines + k_nearest_lines for line in sublist] + distance_texts

    total_frames = len(dataSet) + max(k_values) + 1
    ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init, blit=False, interval=300)

    return fig, ani

def save_animation_as_mp4(anim, output_file="knn_animation.mp4"):
    anim.save(output_file, writer='ffmpeg', fps=1)
    print(f"Animation saved as '{output_file}'")
