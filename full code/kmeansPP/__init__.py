from flask import Blueprint

bp = Blueprint('kmeansPP', __name__, template_folder='templates')

from . import routes