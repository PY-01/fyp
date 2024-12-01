from flask import Blueprint, render_template, request, jsonify
from .kmeansPP import perform_kmeans_plus_plus  # Assuming you implement k-means++ initialization in this function

bp = Blueprint('kmeansPP', __name__, template_folder='../templates')

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve input data from the form
            x = request.form.get('x_data', '').strip()
            y = request.form.get('y_data', '').strip()
            k = request.form.get('k_value', '').strip()

            # Validate inputs
            if not all([x, y, k]):
                return jsonify({'error': "All fields are required. Please provide x, y, and k values."})

            x = list(map(float, x.split(',')))
            y = list(map(float, y.split(',')))
            k = int(k)

            if len(x) != len(y):
                return jsonify({'error': "The number of x and y coordinates must be equal."})

            # Perform k-means++ initialization and generate animation HTML
            animation_html = perform_kmeans_plus_plus(x, y, k)  # K-means++ initialization here
            return jsonify({'animation_html': animation_html})

        except ValueError:
            return jsonify({'error': "Invalid input. Please ensure all values are numeric and properly formatted."})
        except Exception as e:
            return jsonify({'error': f"An error occurred: {str(e)}"})

    return render_template('run_kmeansPP.html')  # Render the appropriate template for k-means++
