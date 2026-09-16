import numpy as np

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

for category in categories:

    data = np.load(f"dataset/{category}.npy")

    print(
        category,
        "min =", data[0].min(),
        "max =", data[0].max(),
        "mean =", data[0].mean()
    )