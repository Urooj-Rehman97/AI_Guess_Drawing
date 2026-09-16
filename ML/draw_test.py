
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# =====================================================
# LOAD TRAINED CNN MODEL
# =====================================================

model = load_model("drawing_cnn.keras")


# =====================================================
# CATEGORIES
# =====================================================

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


# =====================================================
# WINDOW
# =====================================================

window = tk.Tk()

window.title("AI Guess My Drawing")

window.geometry("600x750")

window.resizable(False, False)


# =====================================================
# TITLE
# =====================================================

title_label = tk.Label(
    window,
    text="🤖 AI Guess My Drawing",
    font=("Arial", 24, "bold")
)

title_label.pack(pady=15)


instruction_label = tk.Label(
    window,
    text="Draw something and let AI guess it!",
    font=("Arial", 12)
)

instruction_label.pack()


# =====================================================
# CANVAS
# =====================================================

CANVAS_SIZE = 300

canvas = tk.Canvas(
    window,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross"
)

canvas.pack(pady=15)


# =====================================================
# PIL IMAGE
# =====================================================

image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)

draw = ImageDraw.Draw(image)


# =====================================================
# DRAWING SETTINGS
# =====================================================

BRUSH_SIZE = 12


# =====================================================
# PAINT FUNCTION
# =====================================================

def paint(event):

    x = event.x
    y = event.y

    # Prevent drawing outside canvas
    if x < 0 or x > CANVAS_SIZE:
        return

    if y < 0 or y > CANVAS_SIZE:
        return

    # Draw on Tkinter Canvas
    canvas.create_oval(
        x - BRUSH_SIZE,
        y - BRUSH_SIZE,
        x + BRUSH_SIZE,
        y + BRUSH_SIZE,
        fill="white",
        outline="white"
    )

    # Draw on PIL image
    draw.ellipse(
        [
            x - BRUSH_SIZE,
            y - BRUSH_SIZE,
            x + BRUSH_SIZE,
            y + BRUSH_SIZE
        ],
        fill=255
    )


# Mouse drawing
canvas.bind(
    "<B1-Motion>",
    paint
)


# =====================================================
# PREPROCESS IMAGE
# =====================================================

def preprocess_image():

    # Find actual drawing
    bbox = image.getbbox()

    if bbox is None:
        return None

    # -----------------------------------------------
    # Crop drawing
    # -----------------------------------------------

    cropped = image.crop(bbox)

    width, height = cropped.size

    # -----------------------------------------------
    # Make square
    # -----------------------------------------------

    size = max(
        width,
        height
    )

    square = Image.new(
        "L",
        (size, size),
        0
    )

    x = (size - width) // 2
    y = (size - height) // 2

    square.paste(
        cropped,
        (x, y)
    )

    # -----------------------------------------------
    # Add padding
    # -----------------------------------------------

    padding = int(size * 0.15)

    padded_size = size + (
        padding * 2
    )

    padded = Image.new(
        "L",
        (padded_size, padded_size),
        0
    )

    padded.paste(
        square,
        (padding, padding)
    )

    # -----------------------------------------------
    # Resize to 28x28
    # -----------------------------------------------

    small_image = padded.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    return small_image


# =====================================================
# PREDICT
# =====================================================

def predict():

    small_image = preprocess_image()

    # -----------------------------------------------
    # No drawing
    # -----------------------------------------------

    if small_image is None:

        result_label.config(
            text="Please draw something first!"
        )

        confidence_label.config(
            text=""
        )

        return

    # -----------------------------------------------
    # Save processed image
    # -----------------------------------------------

    small_image.save(
        "processed_drawing.png"
    )

    # -----------------------------------------------
    # Convert to NumPy
    # -----------------------------------------------

    data = np.array(
        small_image
    )

    # -----------------------------------------------
    # Normalize
    # -----------------------------------------------

    data = data.astype(
        "float32"
    ) / 255.0

    # -----------------------------------------------
    # CNN input shape
    #
    # (28, 28)
    #
    # becomes
    #
    # (1, 28, 28, 1)
    # -----------------------------------------------

    data = data.reshape(
        1,
        28,
        28,
        1
    )

    # -----------------------------------------------
    # Prediction
    # -----------------------------------------------

    probabilities = model.predict(
        data,
        verbose=0
    )[0]

    # -----------------------------------------------
    # Get highest probability
    # -----------------------------------------------

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

    # =================================================
    # TOP 3 PREDICTIONS
    # =================================================

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        category = categories[index]

        probability = (
            probabilities[index] * 100
        )

        top_predictions.append(
            f"{category.upper()} - {probability:.2f}%"
        )

    # -----------------------------------------------
    # Display result
    # -----------------------------------------------

    result_label.config(
        text=(
            "🤖 AI THINKS...\n\n"
            f"{prediction.upper()}"
        )
    )

    confidence_label.config(
        text=(
            f"Confidence: {confidence:.2f}%\n\n"
            "Top 3 Predictions:\n"
            f"1. {top_predictions[0]}\n"
            f"2. {top_predictions[1]}\n"
            f"3. {top_predictions[2]}"
        )
    )


# =====================================================
# CLEAR CANVAS
# =====================================================

def clear():

    # Clear Tkinter canvas
    canvas.delete(
        "all"
    )

    # Clear PIL image
    draw.rectangle(
        [
            0,
            0,
            CANVAS_SIZE,
            CANVAS_SIZE
        ],
        fill=0
    )

    # Reset labels
    result_label.config(
        text="Draw something and click GUESS!"
    )

    confidence_label.config(
        text=""
    )


# =====================================================
# BUTTON FRAME
# =====================================================

button_frame = tk.Frame(
    window
)

button_frame.pack(
    pady=5
)


# =====================================================
# CLEAR BUTTON
# =====================================================

clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    command=clear,
    width=15,
    height=2,
    font=("Arial", 11, "bold")
)

clear_button.grid(
    row=0,
    column=0,
    padx=10
)


# =====================================================
# GUESS BUTTON
# =====================================================

guess_button = tk.Button(
    button_frame,
    text="GUESS 🤖",
    command=predict,
    width=15,
    height=2,
    font=("Arial", 11, "bold")
)

guess_button.grid(
    row=0,
    column=1,
    padx=10
)


# =====================================================
# RESULT
# =====================================================

result_label = tk.Label(
    window,
    text="Draw something and click GUESS!",
    font=("Arial", 18, "bold")
)

result_label.pack(
    pady=(15, 5)
)


# =====================================================
# CONFIDENCE / TOP 3
# =====================================================

confidence_label = tk.Label(
    window,
    text="",
    font=("Arial", 11),
    justify="left"
)

confidence_label.pack(
    pady=5
)


# =====================================================
# START APPLICATION
# =====================================================

window.mainloop()
