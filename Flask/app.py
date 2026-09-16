from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from PIL import Image
import numpy as np
import os


app = Flask(__name__)

# Allow React frontend to communicate with Flask
CORS(app)


# ==============================
# MODEL PATH
# ==============================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "ML",
    "drawing_cnn.keras"
)


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==============================
# CATEGORIES
# ==============================

categories = [
    "star",
    "moon",
    "apple",
    "ambulance",
    "basket",
    "banana",
    "bat",
    "bee",
    "book"
]


# ==============================
# HOME
# ==============================

@app.route("/")
def home():

    return {
        "message": "AI Guess My Drawing API is running!"
    }


# ==============================
# PREDICT
# ==============================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------
    # Check image
    # --------------------------

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    file = request.files["image"]


    # --------------------------
    # Open image
    # --------------------------

    try:

        image = Image.open(file)

    except Exception:

        return jsonify({
            "error": "Invalid image file"
        }), 400


    # --------------------------
    # Convert to grayscale
    # --------------------------

    image = image.convert("L")


    # --------------------------
    # Find drawing area
    # --------------------------

    bbox = image.getbbox()


    if bbox is None:

        return jsonify({
            "error": "No drawing found"
        }), 400


    # Crop only drawing
    image = image.crop(bbox)


    # --------------------------
    # Make image square
    # --------------------------

    width, height = image.size

    size = max(
        width,
        height
    )


    square = Image.new(
        "L",
        (size, size),
        0
    )


    x = (
        size - width
    ) // 2

    y = (
        size - height
    ) // 2


    square.paste(
        image,
        (x, y)
    )


    # --------------------------
    # Resize to 28x28
    # --------------------------

    small_image = square.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )


    # --------------------------
    # Convert to NumPy
    # --------------------------

    data = np.array(
        small_image
    )


    # --------------------------
    # Normalize pixels
    # --------------------------

    data = data.astype(
        "float32"
    ) / 255.0


    # --------------------------
    # Add dimensions
    # --------------------------

    data = data.reshape(
        1,
        28,
        28,
        1
    )


    # ==========================
    # MODEL PREDICTION
    # ==========================

    probabilities = model.predict(
        data,
        verbose=0
    )[0]


    # --------------------------
    # Main prediction
    # --------------------------

    prediction_index = np.argmax(
        probabilities
    )


    prediction = categories[
        prediction_index
    ]


    confidence = (
        probabilities[
            prediction_index
        ] * 100
    )


    # ==========================
    # TOP 3 PREDICTIONS
    # ==========================

    top_3_indices = np.argsort(
        probabilities
    )[-3:][::-1]


    top_predictions = []


    for index in top_3_indices:

        top_predictions.append({

            "label": categories[index],

            "confidence": round(
                float(
                    probabilities[index] * 100
                ),
                2
            )

        })


    # ==========================
    # RESPONSE
    # ==========================

    return jsonify({

        "prediction": prediction,

        "confidence": round(
            float(confidence),
            2
        ),

        "top_predictions": top_predictions

    })


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )