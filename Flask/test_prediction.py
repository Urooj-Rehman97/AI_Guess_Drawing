import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("drawing_model.pkl")

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

# Select one drawing from each category
plt.figure(figsize=(12, 8))

for i, category in enumerate(categories):

    data = np.load(f"dataset/{category}.npy")

    image = data[100]

    # Prepare image for model
    image_for_model = image.reshape(1, -1) / 255.0

    # Predict
    prediction = model.predict(image_for_model)[0]

    print(
        "Actual:",
        category,
        "| AI Prediction:",
        prediction
    )

    # Show image
    plt.subplot(3, 3, i + 1)

    plt.imshow(image.reshape(28, 28), cmap="gray")

    plt.title(
        f"Actual: {category}\nAI: {prediction}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()