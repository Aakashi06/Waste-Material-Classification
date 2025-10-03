from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
from PIL import Image
import warnings

# Load artifacts safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.pkl")
model = None
scaler = None
label_classes = []
image_size = (64, 64)
load_error = None
try:
	# Suppress sklearn InconsistentVersionWarning during unpickle
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message=r"^Trying to unpickle estimator .* when using version .*$",
			category=UserWarning,
		)
		with open(ARTIFACT_PATH, "rb") as f:
			artifacts = pickle.load(f)
	model = artifacts.get("model")
	scaler = artifacts.get("scaler")
	label_classes = artifacts.get("label_classes", [])
	image_size = tuple(artifacts.get("image_size", (64, 64)))
except Exception as e:
	load_error = f"Model not loaded. Make sure to run the notebook cell that saves models/waste_classifier.pkl. Details: {e}"

app = Flask(__name__)

@app.route("/", methods=["GET"])  # Home page with upload form
def index():
	return render_template("index.html", error=load_error)

@app.route("/predict", methods=["GET", "POST"])  # Handle image upload and prediction
def predict():
	if request.method == "GET":
		return redirect(url_for("index"))
	# Validate upload
	file = request.files.get("file")
	if not file or file.filename.strip() == "":
		return render_template("index.html", error="Please choose an image file to upload." )
	# Ensure image-like content
	allowed = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
	if not file.filename.lower().endswith(allowed):
		return render_template("index.html", error="Unsupported file type. Use JPG, PNG, BMP, or WEBP.")
	# Block prediction if model failed to load
	if model is None or scaler is None or not label_classes:
		return render_template("index.html", error=(load_error or "Model not loaded."))
	try:
		img = Image.open(file.stream).convert("RGB")
		img = img.resize(image_size)
		arr = np.array(img, dtype=np.uint8)
		flat = arr.reshape(1, -1).astype(np.float32)
		flat_scaled = scaler.transform(flat)
		pred_idx = model.predict(flat_scaled)[0]
		pred_label = label_classes[pred_idx]
		return render_template("index.html", prediction=pred_label)
	except Exception as e:
		return render_template("index.html", error=f"Failed to process image: {e}")

@app.route("/health", methods=["GET"])  # Simple health check
def health():
	status = {
		"model_loaded": model is not None and scaler is not None and bool(label_classes),
		"num_classes": len(label_classes),
		"image_size": image_size,
		"load_error": load_error,
	}
	return status, 200

if __name__ == "__main__":
	# For local development only (align with 5500 so you open the Flask app, not the raw file)
	app.run(host="127.0.0.1", port=5000, debug=True) 