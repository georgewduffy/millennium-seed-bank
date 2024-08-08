from flask import Blueprint, jsonify, request, make_response
from werkzeug.utils import secure_filename
import traceback

from ..models.rcnn import RCNN
from ..models.yolo import YOLO

import io
import numpy as np
from PIL import Image
import base64


predict_bp = Blueprint('predict', __name__)

@predict_bp.errorhandler(Exception)
def handle_exception(e: Exception):
    return {"exception": traceback.format_exc()}, 500

@predict_bp.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response
    else:
        data = request.get_json()
        
        model_id = data.get('model_id')
        base64_image = data['images'][0]

        # Decode the base64 image
        image_data = base64.b64decode(base64_image.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = np.array(image)
        
        # Instantiate model based on model_id
        if model_id == "RCNN":
            model = RCNN()
        else:
            model = YOLO()
        
        # Generate prediction
        prediction = model.predict(image)
        prediction = prediction.to_json()
        return jsonify(prediction)


def filename(file):
    return secure_filename(file.filename)