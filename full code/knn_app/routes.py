from flask import Blueprint, render_template, request, session, jsonify
from .knn import perform_knn_animation_html

bp = Blueprint('knn', __name__, template_folder='../templates/knn')

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Parse input data
            x_data = request.form.get('x_data')
            y_data = request.form.get('y_data')
            classes = request.form.get('classes')
            new_x = float(request.form.get('new_x'))
            new_y = float(request.form.get('new_y'))
            k_values = request.form.get('k_values')

            # Convert input strings to lists
            x = list(map(float, x_data.split(',')))
            y = list(map(float, y_data.split(',')))
            class_list = list(map(int, classes.split(',')))
            k_list = [int(k.strip()) for k in k_values.split(',') if k.strip().isdigit()]

            # Perform k-NN and generate animation
            animation_html = perform_knn_animation_html(x, y, class_list, new_x, new_y, k_list)

            # Return JSON response
            return jsonify({"animation_html": animation_html})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400  # Send specific error message
        except Exception as e:
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    # Render the initial form on GET request
    return render_template('run_knn.html')

@bp.route('/knn/set_frame', methods=['POST'])
def set_frame():
    animation_id = request.form.get('animationId')
    frame = int(request.form.get('frame'))
    # Update the animation to the specified frame
    # Generate updated HTML or a new image
    return jsonify({"success": True, "animation_html": updated_html})

