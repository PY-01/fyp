import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import base64

def perform_kmeans(x, y, cx, cy, k):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c='b', marker='o', label='Data')
    centroid, = ax.plot(cx, cy, 'rx', markersize=10, label='Centroids')
    ax.legend()
    ax.set_title('K-means Clustering Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

    colors = plt.cm.jet(np.linspace(0, 1, k))
    
    def update(frame):
        # Compute distances and assign clusters
        distances = np.sqrt((np.array(x)[:, None] - np.array(cx))**2 + (np.array(y)[:, None] - np.array(cy))**2)
        clusters = np.argmin(distances, axis=1)

        # Clear previous lines
        for line in ax.lines:
            line.remove()

        # Update lines for each point
        for i in range(len(x)):
            ax.plot([x[i], cx[clusters[i]]], [y[i], cy[clusters[i]]], color=colors[clusters[i]], alpha=0.3)

        # Update centroids
        new_cx = []
        new_cy = []
        for i in range(k):
            cluster_points = [(x[j], y[j]) for j in range(len(x)) if clusters[j] == i]
            if cluster_points:
                new_cx.append(np.mean([p[0] for p in cluster_points]))
                new_cy.append(np.mean([p[1] for p in cluster_points]))
            else:
                new_cx.append(cx[i])
                new_cy.append(cy[i])
        
        cx[:] = new_cx
        cy[:] = new_cy

        centroid.set_data(cx, cy)
        return scatter, centroid

    ani = FuncAnimation(fig, update, frames=range(10), blit=True, repeat=False, interval=1000)
    return fig, ani

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')
