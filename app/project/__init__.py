from flask import Flask, request, g
from pathlib import Path
import torch
import yaml
from box import ConfigBox
from utils import get_transform
from PIL import Image
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)


def load_params(params_file):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params


app = Flask(__name__)

MODEL_PICKLE_PATH = Path("./models/model_pickle.pkl").absolute()
PARAMS_FILE_PATH = Path("./params.yaml").absolute()


@app.before_request
def before_request():
    if "model" not in g:
        g.model = torch.load(MODEL_PICKLE_PATH, map_location=torch.device("cpu"))
        g.model.eval()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ["png", "jpg"]


@app.route("/")
def home():
    return "Healthy"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return {"error": "No file part"}, 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return {"error": "No selected file or file type not allowed"}, 400

    if file:
        try:
            app.logger.info(f"Process img {time.strftime('%Y-%m-%d-%H-%M-%S')}")
            params = load_params(params_file=PARAMS_FILE_PATH)
            image_size = params.transform.patch_size
            img = np.array(Image.open(file).convert("RGB"))
            trms = get_transform(image_size)
            img_tensor = trms(image=img).get("image")
            with torch.no_grad():
                pred = g.model(img_tensor.unsqueeze(0))
                activated_preds = torch.sigmoid(pred)
                activated_preds = (activated_preds > 0.5).float()
                pred = activated_preds.detach().numpy()[0]
                resp = {"pred": pred.tolist()}
                app.logger.info(
                    f"Finished process img {time.strftime('%Y-%m-%d-%H-%M-%S')}"
                )
                return resp
        except Exception as e:
            app.logger.error(e)
            return {"error": "An error occurred processing the file"}, 500

    return {"error": "An error occurred processing the file"}, 500
