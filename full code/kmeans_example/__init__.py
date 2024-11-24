from flask import Blueprint

knn_example_bp = Blueprint('kmeans_example', __name__, template_folder='templates')

from . import routes  # Import routes to register the routes with the blueprint
