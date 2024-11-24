from flask import Blueprint, render_template, request, session, jsonify
from .kmeans import perform_kmeans

bp = Blueprint('kmeans', __name__, template_folder='../templates/kmeans')

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve input data from the form
            x = request.form.get('x_data', '').strip()
            y = request.form.get('y_data', '').strip()
            k = request.form.get('k_value', '').strip()
            cx = request.form.get('cx', '').strip()
            cy = request.form.get('cy', '').strip()

            # Validate inputs
            if not all([x, y, k, cx, cy]):
                return jsonify({'error': "All fields are required. Please provide x, y, k, cx, and cy values."})

            x = list(map(float, x.split(',')))
            y = list(map(float, y.split(',')))
            cx = list(map(float, cx.split(',')))
            cy = list(map(float, cy.split(',')))
            k = int(k)

            if len(cx) != k or len(cy) != k:
                return jsonify({'error': "The number of initial centroids (cx, cy) must match the value of k."})

            if len(x) != len(y):
                return jsonify({'error': "The number of x and y coordinates must be equal."})

            # Generate k-means animation HTML
            animation_html = perform_kmeans(x, y, cx, cy, k)
            return jsonify({'animation_html': animation_html})

        except ValueError:
            return jsonify({'error': "Invalid input. Please ensure all values are numeric and properly formatted."})
        except Exception as e:
            return jsonify({'error': f"An error occurred: {str(e)}"})

    return render_template('run_kmeans.html')
