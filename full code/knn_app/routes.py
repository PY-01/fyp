from flask import Blueprint, render_template, request, redirect, url_for, session
from .knn import perform_knn_animation, save_animation_as_mp4
import os

bp = Blueprint('knn', __name__, template_folder='../templates/knn')

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        x = request.form.get('x_data')
        y = request.form.get('y_data')
        classes = request.form.get('classes')
        
        if not x or not y or not classes:
            return redirect(url_for('knn.index'))

        try:
            session['x'] = list(map(float, x.split(',')))
            session['y'] = list(map(float, y.split(',')))
            session['classes'] = list(map(int, classes.split(',')))
        except ValueError:
            return redirect(url_for('knn.index'))
        
        return redirect(url_for('knn.input_new_point'))

    return render_template('knn.html')

@bp.route('/input_new_point', methods=['GET', 'POST'])
def input_new_point():
    if request.method == 'POST':
        new_x = request.form.get('new_x')
        new_y = request.form.get('new_y')
        k_value = request.form.get('k_value')
        show_grid = request.form.get('show_grid') == 'on'  # Get checkbox value
        
        if not new_x or not new_y or not k_value:
            return redirect(url_for('knn.input_new_point'))

        try:
            new_x = float(new_x)
            new_y = float(new_y)
            k_value = int(k_value)
        except ValueError:
            return redirect(url_for('knn.input_new_point'))

        session['new_x'] = new_x
        session['new_y'] = new_y
        session['k'] = k_value
        session['show_grid'] = show_grid

        x = session.get('x')
        y = session.get('y')
        classes = session.get('classes')
        
        # Check if all session data is valid
        if not x or not y or not classes:
            return redirect(url_for('knn.index'))

        fig, ani = perform_knn_animation(x, y, classes, new_x, new_y, k_value, show_grid)

        output_path = os.path.join('static', 'knn_animation.mp4')
        save_animation_as_mp4(ani, output_file=output_path)

        return render_template('run_knn.html', video_path=url_for('static', filename='knn_animation.mp4'))

    return render_template('input_new_point.html')
