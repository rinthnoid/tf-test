from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
import numpy as np
import os

# === Configuration ===
MODEL_PATH = 'model/best_model.h5'  # Ensure this is correct in your repo
LABELS = ['class1', 'class2', 'class3', 'class4']  # 🔁 Update with your real labels
IMG_SIZE = 512

# === Load model ===
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise

app = Flask(__name__)

def preprocess_image(image: Image.Image):
    """Resize, convert to RGB, and preprocess image for MobileNetV3."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image.convert('RGB')
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']
    try:
        image = Image.open(file.stream)
        preprocessed = preprocess_image(image)

        preds = model.predict(preprocessed)[0]
        pred_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        label = LABELS[pred_idx] if pred_idx < len(LABELS) else "Unknown"

        return jsonify({'class': label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def root():
    return '✅ TensorFlow model API is running!'

