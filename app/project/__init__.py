from flask import Flask, request, g
from pathlib import Path
import torch
import yaml
from box import ConfigBox
from utils import get_transform
from PIL import Image
import os

def load_params(params_file):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params


app = Flask(__name__)

MODEL_PICKLE_PATH = Path('./models/model_pickle.pkl').absolute()
PARAMS_FILE_PATH = Path('./params.yaml').absolute()

@app.before_request
def before_request():
    if 'model' not in g:
        g.model = torch.load(MODEL_PICKLE_PATH, map_location=torch.device('cpu'))
        g.model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg']

@app.route("/")
def home():
    print(os.getcwd())
    print(g.model)
    return os.getcwd()


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file part"}, 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return {"error": "No selected file or file type not allowed"}, 400

    if file:
        params = load_params(params_file=PARAMS_FILE_PATH)
        image_size = params.transform.patch_size
        img_bytes = file.read()
        trms = get_transform(image_size)
        img_tensor = trms(Image.open(img_bytes))
        with torch.no_grad():
            pred = g.model(img_tensor)
            pred = pred.detach().numpy()
            resp = {'pred': pred.tolist()}
            return resp

    return {"error": "An error occurred processing the file"}, 500
