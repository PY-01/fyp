from flask import Blueprint, render_template, request, redirect, url_for, session
from .kmeans import perform_kmeans

bp = Blueprint('kmeans', __name__, template_folder='../templates/kmeans')

@bp.route('/', methods=['GET', 'POST'])
def index():
    # First page: Takes x and y data input
    if request.method == 'POST':
        x = request.form.get('x_data')
        y = request.form.get('y_data')

        if not x or not y:
            return redirect(url_for('kmeans.index'))

        try:
            session['x'] = list(map(float, x.split(',')))
            session['y'] = list(map(float, y.split(',')))
        except ValueError:
            return redirect(url_for('kmeans.index'))

        return redirect(url_for('kmeans.run_kmeans'))

    return render_template('kmeans.html')

@bp.route('/run_kmeans', methods=['GET', 'POST'])
def run_kmeans():
    # Second page: Takes k, cx, cy input and displays the result
    if request.method == 'POST':
        try:
            x = session.get('x', [])
            y = session.get('y', [])
            k = int(request.form['k_value'])
            cx = list(map(float, request.form['cx'].split(',')))
            cy = list(map(float, request.form['cy'].split(',')))

            if len(cx) != k or len(cy) != k:
                return render_template('run_kmeans.html', error="Number of centroids must match k")

            animation_html = perform_kmeans(x, y, cx, cy, k)
            return render_template('run_kmeans.html', animation_html=animation_html)

        except ValueError:
            return render_template('run_kmeans.html', error="Invalid input. Please ensure data is numeric.")
    
    return render_template('run_kmeans.html')
