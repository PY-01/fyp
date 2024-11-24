from flask import Blueprint, request, jsonify
from .knn_example import perform_knn_animation_html  # Import kNN function

bp = Blueprint('knn_example', __name__)  # Define blueprint for knn_example

@bp.route('/input_new_point', methods=['POST'])
def input_new_point():
    data = request.form

    # Parse form data
    try:
        x_data = list(map(float, data.get('x_data').split(',')))
        y_data = list(map(float, data.get('y_data').split(',')))
        classes = list(map(int, data.get('classes').split(',')))
        new_x = float(data.get('new_x'))
        new_y = float(data.get('new_y'))
        k_values = [int(k) for k in data.get('k_values').split(',')]
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

    # Generate the k-NN animation HTML
    try:
        animation_html = perform_knn_animation_html(x_data, y_data, classes, new_x, new_y, k_values)
        return jsonify({'animation_html': animation_html})
    except Exception as e:
        return jsonify({'error': f'Error generating animation: {str(e)}'}), 500
