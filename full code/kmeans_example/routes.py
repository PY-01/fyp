from flask import Blueprint, request, jsonify
from .kmeans_example import perform_kmeans  # Import the function to perform k-means

bp = Blueprint('kmeans_example', __name__)  # Define blueprint for k-means example

@bp.route('/kmeans-example', methods=['POST'])
def kmeans_example():
    data = request.form

    try:
        # Parse form data
        x_data = list(map(float, data.get('x_data').split(',')))
        y_data = list(map(float, data.get('y_data').split(',')))
        k = int(data.get('k_value'))
        cx = list(map(float, data.get('cx').split(',')))
        cy = list(map(float, data.get('cy').split(',')))
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

    # Ensure the number of centroids matches k
    if len(cx) != k or len(cy) != k:
        return jsonify({'error': 'Number of centroids must match k'}), 400

    try:
        # Perform k-means clustering and get the animation HTML
        animation_html = perform_kmeans(x_data, y_data, cx, cy, k)
        return jsonify({'animation_html': animation_html})
    except Exception as e:
        return jsonify({'error': f'Error generating animation: {str(e)}'}), 500
