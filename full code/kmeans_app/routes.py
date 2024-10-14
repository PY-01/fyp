from flask import request, render_template, redirect, url_for, session
from . import bp  # Import the blueprint from __init__.py
from .kmeans import perform_kmeans, fig_to_base64  # Import functions from kmeans.py
import numpy as np

@bp.route('/', methods=['GET', 'POST'])
def kmeans_home():
    if request.method == 'POST':
        # Save user input data to session
        x = request.form.get('x_data')
        y = request.form.get('y_data')
        
        if not x or not y:
            return redirect(url_for('kmeans.kmeans_home'))  # Redirect if data is missing

        try:
            session['x'] = list(map(float, x.split(',')))
            session['y'] = list(map(float, y.split(',')))
        except ValueError:
            return redirect(url_for('kmeans.kmeans_home'))  # Redirect if data conversion fails
        
        return redirect(url_for('kmeans.input_centroids'))

    # Render the template 'kmeans.html'
    return render_template('kmeans/kmeans.html')

@bp.route('/input_centroids', methods=['GET', 'POST'])
def input_centroids():
    if request.method == 'POST':
        # Save the number of clusters (k) to the session
        k_value = request.form.get('k_value')
        cx = request.form.get('cx')
        cy = request.form.get('cy')

        if not k_value or not cx or not cy:
            return redirect(url_for('kmeans.input_centroids'))  # Redirect if any input is missing

        try:
            k = int(k_value)
            session['k'] = k
            session['cx'] = list(map(float, cx.split(',')))
            session['cy'] = list(map(float, cy.split(',')))
        except ValueError:
            return redirect(url_for('kmeans.input_centroids'))  # Redirect if conversion fails

        if len(session['cx']) != k or len(session['cy']) != k:
            return redirect(url_for('kmeans.input_centroids'))  # Redirect if centroid counts don't match k

        return redirect(url_for('kmeans.run_kmeans'))

    return render_template('kmeans/input_centroids.html')

@bp.route('/run_kmeans')
def run_kmeans():
    x = session.get('x')
    y = session.get('y')
    cx = session.get('cx')
    cy = session.get('cy')
    k = session.get('k')

    if not x or not y or not cx or not cy or not k:
        return redirect(url_for('kmeans.kmeans_home'))  # Redirect if any required data is missing

    try:
        fig, ani = perform_kmeans(x, y, cx, cy, k)
        img = fig_to_base64(fig)
        return render_template('kmeans/run_kmeans.html', img=img)
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred while running K-means: {e}"
