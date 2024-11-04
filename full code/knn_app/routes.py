from flask import Blueprint, render_template, request, redirect, url_for, session
from .knn import perform_knn_animation_html

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
        k_values = request.form.get('k_values')
        show_grid = request.form.get('show_grid') == 'on'

        if not new_x or not new_y or not k_values:
            return redirect(url_for('knn.input_new_point'))

        try:
            new_x = float(new_x)
            new_y = float(new_y)
            k_list = [int(k.strip()) for k in k_values.split(',') if k.strip().isdigit()]
            if len(k_list) < 1 or len(k_list) > 4:
                raise ValueError("k_values should have 1-4 values")
        except ValueError:
            return redirect(url_for('knn.input_new_point'))

        session['new_x'] = new_x
        session['new_y'] = new_y
        session['k_list'] = k_list
        session['show_grid'] = show_grid

        x = session.get('x')
        y = session.get('y')
        classes = session.get('classes')

        if not x or not y or not classes:
            return redirect(url_for('knn.index'))

        try:
            animation_html = perform_knn_animation_html(x, y, classes, new_x, new_y, k_list)
        except Exception:
            return redirect(url_for('knn.input_new_point'))

        return render_template('run_knn.html', animation_html=animation_html)

    return render_template('input_new_point.html')
