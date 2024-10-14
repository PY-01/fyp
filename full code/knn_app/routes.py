from flask import Blueprint, render_template, request, redirect, url_for, session
from .knn import perform_knn_animation, save_animation_frames
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
            session['new_x'] = float(new_x)
            session['new_y'] = float(new_y)
            session['k_value'] = int(k_value)
            session['show_grid'] = show_grid
        except ValueError:
            return redirect(url_for('knn.input_new_point'))

        return redirect(url_for('knn.run_knn'))

    return render_template('input_new_point.html')

@bp.route('/run_knn')
def run_knn():
    x = session.get('x')
    y = session.get('y')
    classes = session.get('classes')
    new_x = session.get('new_x')
    new_y = session.get('new_y')
    k_value = session.get('k_value')
    show_grid = session.get('show_grid', True)

    if not x or not y or not classes or new_x is None or new_y is None or k_value is None:
        return redirect(url_for('knn.index'))

    try:
        frames_dir = os.path.join('static', 'frames')
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        fig, anim = perform_knn_animation(x, y, classes, new_x, new_y, k=k_value, show_grid=show_grid)

        total_frames = len(x) + k_value + 1
        save_animation_frames(anim, total_frames, output_dir=frames_dir)

        frame_files = [f'frames/frame_{i}.png' for i in range(total_frames)]
        
        return render_template('run_knn.html', frame_files=frame_files)
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred while generating the animation: {e}"
    
@bp.route('/select_k')
def select_k():
    return render_template('select_k.html')



