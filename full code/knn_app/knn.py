import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def classify0(newPoint, dataSet, labels, k):
    newPoint = np.array(newPoint)
    testDataDiff = dataSet - newPoint
    testDataDist = np.sqrt((testDataDiff ** 2).sum(axis=1))
    sortedIndex = testDataDist.argsort()
    sortedLabels = [labels[i] for i in sortedIndex]
    K_nearest = sortedLabels[:k]
    return K_nearest, testDataDist[sortedIndex]

def perform_knn_animation(x, y, classes, new_x, new_y, k, show_grid=True):
    fig, ax = plt.subplots()
    x = np.array(x)
    y = np.array(y)
    classes = np.array(classes)

    colors = ['blue' if cls == 0 else 'green' for cls in classes]
    ax.scatter(x[classes == 0], y[classes == 0], c='blue', label='Class 0', s=50)
    ax.scatter(x[classes == 1], y[classes == 1], c='green', label='Class 1', s=50)
    new_point_scatter = ax.scatter(new_x, new_y, c="red", marker="o", label='New Point', s=100)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('k-Nearest Neighbour Algorithm')
    ax.grid(show_grid, linestyle='--', color='grey', alpha=0.7)
    ax.legend()

    nearest_labels, sorted_distances = classify0((new_x, new_y), np.column_stack((x, y)), classes, k)

    dotted_lines = [ax.plot([], [], linestyle=':', color='grey')[0] for _ in range(len(x))]
    k_nearest_lines = [ax.plot([], [], linestyle='-', linewidth=1.5, color='grey')[0] for _ in range(k)]
    
    distance_text = ax.text(0, 0, '', color='black', fontsize=10)

    def init():
        for line in dotted_lines + k_nearest_lines:
            line.set_data([], [])
        distance_text.set_text('')
        return dotted_lines + k_nearest_lines + [distance_text]

    def animate(frame):
        distances = [(np.sqrt((new_x - xi) ** 2 + (new_y - yi) ** 2), xi, yi) for xi, yi in zip(x, y)]
        distances.sort()
        sorted_x = [dist[1] for dist in distances]
        sorted_y = [dist[2] for dist in distances]
        sorted_distances = [dist[0] for dist in distances]

        if frame <= len(x):
            for i, line in enumerate(dotted_lines):
                if frame == i + 1:
                    line.set_data([sorted_x[i], new_x], [sorted_y[i], new_y])
                    dist = sorted_distances[i]
                    mid_x = (sorted_x[i] + new_x) / 2
                    mid_y = (sorted_y[i] + new_y) / 2
                    distance_text.set_text(f"{dist:.2f}")
                    distance_text.set_position((mid_x, mid_y))
                else:
                    line.set_data([], [])
        else:
            for line in dotted_lines:
                line.set_data([], [])
            for i, line in enumerate(k_nearest_lines):
                if frame - len(x) >= i + 1:
                    dist, xi, yi = sorted_distances[i], sorted_x[i], sorted_y[i]
                    line.set_data([new_x, sorted_x[i]], [new_y, sorted_y[i]])
                    mid_x = (sorted_x[i] + new_x) / 2
                    mid_y = (sorted_y[i] + new_y) / 2
                    distance_text.set_text(f"{dist:.2f}")
                    distance_text.set_position((mid_x, mid_y))
                else:
                    line.set_data([], [])

            if frame == len(x) + k:
                update_new_point()
                majority_circle()  # Call to majority_circle() after updating the new point
                return dotted_lines + k_nearest_lines + [distance_text]

            if frame == len(x) + k + 1:
                distance_text.set_text('')
                ani.event_source.stop()
                return []

        return dotted_lines + k_nearest_lines + [distance_text]

    def update_new_point():
        unique_labels = list(set(nearest_labels))
        counts = [nearest_labels.count(i) for i in unique_labels]
        majority_label = unique_labels[counts.index(max(counts))]
        majority_color = 'blue' if majority_label == 0 else 'green'
        new_point_scatter.set_color(majority_color)  # Change the color of the new point to the majority color

    def majority_circle():
        distances = [(np.sqrt((new_x - xi) ** 2 + (new_y - yi) ** 2), xi, yi, label) 
                      for xi, yi, label in zip(x, y, classes)]
        distances.sort()

        k_nearest_points = distances[:k]
        k_nearest_labels = [label for _, _, _, label in k_nearest_points]
        label_counts = {label: k_nearest_labels.count(label) for label in set(k_nearest_labels)}

        majority_label = max(label_counts, key=label_counts.get)

        majority_points = [(xi, yi) for _, xi, yi, label in k_nearest_points if label == majority_label]

        if majority_points:
            majority_x, majority_y = zip(*majority_points)
            ax.scatter(majority_x, majority_y, facecolors='none', edgecolors='black', s=300, label='Majority Points')
            ax.legend()

    ani = FuncAnimation(fig, animate, frames=len(x) + k + 1, init_func=init, blit=True, interval=300)

    return fig, ani

def save_animation_as_mp4(anim, output_file="knn_animation.mp4"):
    anim.save(output_file, writer='ffmpeg', fps=1)
    print(f"Animation saved as '{output_file}'")
