from flask import Blueprint

bp = Blueprint('kmeans', __name__, template_folder='templates/kmeans')

from . import routes  # Import routes to register them with the blueprint
