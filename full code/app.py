from flask import Flask, render_template
from knn_app.routes import bp as knn_bp
from kmeans_app.routes import bp as kmeans_bp

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session management

# Register blueprints
app.register_blueprint(knn_bp, url_prefix='/knn')
app.register_blueprint(kmeans_bp, url_prefix='/kmeans')

# Home route
@app.route('/')
def home():
    return render_template('mlmv.html')

# Get Started route (Guides page)
@app.route('/guides')
def guides():
    return render_template('guides.html')

# k-NN Guide route
@app.route('/guides/knn')
def knn_guide():
    return render_template('knn_guide.html')

# kmeans Guide route
@app.route('/guides/kmeans')
def kmeans_guide():
    return render_template('kmeans_guide.html')

# Explore route
@app.route('/explore')
def explore():
    return render_template('explore.html')

# Theory route
@app.route('/theory')
def theory():
    return render_template('theory.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Example routes for k-NN and K-means
@app.route('/knn-example')
def knn_example():
    return render_template('knn_example.html')

@app.route('/kmeans-example')
def kmeans_example():
    return render_template('kmeans_example.html')

if __name__ == '__main__':
    app.run(debug=True)
