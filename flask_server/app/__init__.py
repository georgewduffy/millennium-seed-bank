import sys
import os

from flask import Flask
from flask_cors import CORS
from .routes.inference import predict_bp

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..'))
sys.path.insert(0, src_dir)

def create_app():
    app = Flask(__name__)
    app.register_blueprint(predict_bp)
    CORS(app)
    # , resources={r"/predict": {"origins": "*", "methods": ["POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}}
    return app
