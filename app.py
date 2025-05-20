from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'model/best_model.h5'
model = load_model(MODEL_PATH)

# Class names – same as during training
class_names = ['banana', 'coke', 'tiger', 'lays']  # update this!

IMG_SIZE = 512

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    img_array = preprocess_image(image)
    
    predictions = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({'class': pred_class, 'confidence': confidence})
