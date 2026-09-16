import numpy as np
import matplotlib.pyplot as plt

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

rows = len(categories)
columns = 5

plt.figure(figsize=(12, 20))

for row, category in enumerate(categories):

    data = np.load(f"dataset/{category}.npy")

    print(category, "dataset shape:", data.shape)

    for col in range(columns):

        image = data[col].reshape(28, 28)

        position = row * columns + col + 1

        plt.subplot(rows, columns, position)

        plt.imshow(image, cmap="gray")

        plt.title(category)

        plt.axis("off")

plt.tight_layout()
plt.show()