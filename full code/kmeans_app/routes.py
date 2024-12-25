from flask import Blueprint, render_template, request, jsonify
from .assign_kmeans import animate_assign_clusters
from .iter_kmeans import animate_iterations
import numpy as np

# Define Blueprint
bp = Blueprint('kmeans', __name__, template_folder='../templates/kmeans')

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')  # Determine which button was clicked
        try:
            # Retrieve input data from the form
            x = request.form.get('x_data', '').strip()
            y = request.form.get('y_data', '').strip()
            k = request.form.get('k_value', '').strip()
            cx = request.form.get('cx', '').strip()
            cy = request.form.get('cy', '').strip()

            # Validate input fields
            if not all([x, y, k, cx, cy]):
                return jsonify({'error': "All fields are required. Please provide x, y, k, cx, and cy values."})

            # Convert inputs to appropriate formats
            x = np.array(list(map(float, x.split(','))))
            y = np.array(list(map(float, y.split(','))))
            cx = np.array(list(map(float, cx.split(','))))
            cy = np.array(list(map(float, cy.split(','))))
            k = int(k)

            # Ensure the input dimensions are correct
            if len(cx) != k or len(cy) != k:
                return jsonify({'error': "The number of initial centroids (cx, cy) must match the value of k."})
            if len(x) != len(y):
                return jsonify({'error': "The number of x and y coordinates must be equal."})

            # Handle actions based on button clicked
            if action == 'assign':
                # Perform initial cluster assignment and return animation for assignment
                assignment_html = animate_assign_clusters(x, y, cx, cy, k)
                return jsonify({'html': assignment_html})
            elif action == 'iterate':
                # Perform iterations and return animation for the iterations
                iteration_html = animate_iterations(x, y, cx, cy, k)
                return jsonify({'html': iteration_html})
            else:
                return jsonify({'error': "Invalid action. Please ensure the button is correctly configured."})

        except ValueError:
            return jsonify({'error': "Invalid input. Please ensure all values are numeric and properly formatted."})
        except Exception as e:
            return jsonify({'error': f"An error occurred: {str(e)}"})

    # Render the k-Means clustering page
    return render_template('run_kmeans.html')
