from flask import Blueprint

bp = Blueprint('knn', __name__, template_folder='templates/knn')

from . import routes
